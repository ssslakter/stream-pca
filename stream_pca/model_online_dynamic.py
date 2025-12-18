#  online RPCA, assumed the background can be slowly changed OMWRPCA

import torch, numpy as np
from .utils import get_device

def svt(X, tau):
    """Singular Value Thresholding Operator D_tau(X)."""
    U, S, V = torch.svd(X)
    # Apply Soft-Thresholding
    S_thr = torch.maximum(S - tau, torch.zeros_like(S))
    # Reconstruct
    return U @ torch.diag(S_thr) @ V.transpose(-2, -1)


def soft_thresholding(X, tau):
    """Element-wise Soft Thresholding Operator S_tau(X)."""
    # S_tau[x] = sgn(x) * max(|x| - tau, 0)
    shrinkage = torch.maximum(torch.abs(X) - tau, torch.zeros_like(X))
    return torch.sign(X) * shrinkage

class OnlineMovingWindowRPCA:
    """
    Implements Online Moving Window RPCA (OMWRPCA).
    Uses a burn-in batch (Full RPCA) for initialization and then tracks 
    the subspace using a sliding window.
    """
    def __init__(self, rank=400, lam1=0.05, lam2=0.005, n_win=20, n_burnin=20, max_inner_iter=50, device=None):
        # Hyperparameters
        self.rank = rank              # r: Target rank (maximum rank)
        self.lam1 = lam1              # lambda1: Ridge regularization
        self.lam2 = lam2              # lambda2: Sparsity weight
        assert n_burnin >= n_win, "Burn-in size must be at least as large as window size."
        self.n_win = n_win            # n_win: Moving window size
        self.n_burnin = n_burnin      # n_burnin: Initialization batch size (must be >= n_win)
        self.max_inner_iter = max_inner_iter
        
        self.device = device or get_device()
        
        # OMWRPCA State
        self.U = None   # The Background Basis (D x r)
        self.A = None   # Accumulation Matrix A (r x r)
        self.B = None   # Accumulation Matrix B (D x r)
        
        # History Buffer (Stores the last n_win v_t and s_t vectors)
        # Needed for the subtraction step in Line 6 (v_{t-nwin} and s_{t-nwin})
        self.v_history = [] 
        self.s_history = []
        self.m_history = [] # Need m_{t-nwin} for the B update
        
        # Dimensions and tracking
        self.D = 0
        self.t = 0 # Frame counter
        self.I_r = None # Identity matrix

    def _init_internal_state(self, M_burnin, L_burnin, S_burnin, rank):
        """
        Initializes U0, A0, B0 from the L_burnin matrix.
        """
        self.D, T_burnin = L_burnin.shape
        self.rank = rank # Use the rank determined by Batch RPCA

        print(f"Initializing state with rank r={self.rank} determined by Batch RPCA.")

        # 1. Compute SVD on L_burnin (Batch RPCA result)
        # Lb = U_hat * Sigma_hat * V_hat.T
        # We perform truncated SVD to the determined rank 'r'
        U_hat, S_hat, V_hat = torch.svd(L_burnin)
        
        # Truncate to rank r
        U_hat = U_hat[:, :self.rank]
        S_hat = torch.diag(S_hat[:self.rank])
        V_hat = V_hat[:, :self.rank] # V_hat is (T_burnin x r)

        # 2. U0 = U_hat * Sigma_hat^(1/2) (m x r)
        # Note: The paper uses U0 = U_hat * Sigma_hat^(1/2). We implement it here.
        # This normalization is part of the original RPCA-STOC paper's derivation.
        U0 = U_hat @ torch.sqrt(S_hat)
        self.U = U0
        self.I_r = torch.eye(self.rank, device=self.device)

        # 3. Calculate v_i (coefficients) and initialize A0, B0
        # Since L_burnin = U0 * V0^T, the coefficient matrix V0 is V_hat * Sigma_hat^(1/2)
        # V0 is (T_burnin x r). v_i is the i-th row of V0 (r x 1)
        V0 = V_hat @ torch.sqrt(S_hat)
        
        # Initial A and B are summed over the last n_win frames of the burn-in period.
        # Start index i = T_burnin - n_win
        start_idx = T_burnin - self.n_win
        
        A_init = torch.zeros(self.rank, self.rank, device=self.device)
        B_init = torch.zeros(self.D, self.rank, device=self.device)

        # Loop over the initialization window
        for i in range(start_idx, T_burnin):
            v_i = V0[i, :].view(-1, 1)        # (r x 1)
            s_i = S_burnin[:, i].view(-1, 1)  # (D x 1)
            m_i = M_burnin[:, i].view(-1, 1)  # (D x 1)

            A_init += v_i @ v_i.T
            B_init += (m_i - s_i) @ v_i.T
            
            # Store history for the first n_win steps
            self.v_history.append(v_i)
            self.s_history.append(s_i)
            self.m_history.append(m_i)
            
        self.A = A_init
        self.B = B_init
        self.t = T_burnin # Start the frame counter where burn-in left off

    def init_basis(self, frames):
        """
        Performs the required Batch RPCA (PCP) on the burn-in samples.
        """
        # We need a temporary full RPCA solver here.
        # Assuming you defined the FullRobustPCA class earlier.
        
        if len(frames) < self.n_burnin:
             raise ValueError(f"Need at least {self.n_burnin} frames for burn-in.")
        
        # 1. Setup Data Matrix M for Batch RPCA
        M_list = [
            torch.tensor(f, dtype=torch.float32, device=self.device).view(-1) / 255.0 
            for f in frames[:self.n_burnin]
        ]
        M_burnin = torch.stack(M_list, dim=1) # (D x T_burnin)
        
        # D is set now
        D = M_burnin.shape[0]
        
        # 2. Run Batch RPCA (reusing the FullRobustPCA logic, adapted here)
        print("\n--- Step 1: Running Batch RPCA (PCP) on Burn-in Samples ---")
        
        # Simplified in-line ADMM for burn-in (for completeness)
        mu = 0.01  # Penalty parameter for ADMM
        lam = 1.0 / np.sqrt(max(D, self.n_burnin)) # Standard lambda
        
        L_k, S_k, Y_k = torch.zeros_like(M_burnin), torch.zeros_like(M_burnin), torch.zeros_like(M_burnin)
        
        for k in range(200): # Hardcoded iterations for stability
            # Corrected ADMM updates
            L_k_plus_1 = svt(M_burnin - S_k + (1 / mu) * Y_k, 1 / mu)
            S_k_plus_1 = soft_thresholding(M_burnin - L_k_plus_1 + (1 / mu) * Y_k, lam / mu)
            Y_k_plus_1 = Y_k + mu * (M_burnin - L_k_plus_1 - S_k_plus_1)
            L_k, S_k, Y_k = L_k_plus_1, S_k_plus_1, Y_k_plus_1
        
        # 3. Determine Rank 'r'
        # Get the singular values of the low-rank component L
        S_L = torch.svd(L_k)[1]
        
        # Estimate rank r: count singular values above a small threshold
        r_est = torch.sum(S_L > 1e-4).item()
        r_est = min(r_est, self.rank) # Ensure r_est doesn't exceed max allowed rank
        
        # 4. Initialize OMWRPCA state
        self._init_internal_state(M_burnin, L_k, S_k, r_est)


    def __call__(self, x_frame):
        """
        Online processing step (Line 4-7 in Algorithm 4).
        """
        if self.U is None:
             raise RuntimeError("Must call init_basis() first to perform burn-in.")

        self.t += 1
        H, W, C = x_frame.shape
        m = x_frame.view(-1, 1) # (D, 1)

        # 1. Project the new sample (vt, st) (Line 5)
        v, s = self._project_sample(m)

        # 2. Update A and B (Line 6) - Sliding Window Update
        # Note: This subtraction step only begins once the history buffer is full (t > n_win)
        
        # Get the oldest frame that is leaving the window
        # (t is 1-indexed for the loop, buffer index is 0-indexed)
        if len(self.v_history) >= self.n_win:
            v_old = self.v_history.pop(0)
            s_old = self.s_history.pop(0)
            m_old = self.m_history.pop(0)
            
            # Subtraction Step
            self.A -= v_old @ v_old.T
            self.B -= (m_old - s_old) @ v_old.T
        
        # Addition Step (new frame t)
        self.A += v @ v.T
        self.B += (m - s) @ v.T
        
        # Store the current frame's components
        self.v_history.append(v)
        self.s_history.append(s)
        self.m_history.append(m)
        
        # 3. Compute Ut (Line 7)
        # Ut = B (At + lam1 I)^-1
        lhs_U = self.A + self.lam1 * self.I_r
        self.U = torch.linalg.solve(lhs_U, self.B.T).T

        # 4. Return Result
        L_vec = (self.U @ v).view(H, W, C)
        S_vec = (s.view(H, W, C).abs() - 0.01).relu()

        return L_vec.clip(0, 1), S_vec.clip(0, 1)

    def _project_sample(self, m):
        """
        Inner loop to solve for (v, s) given fixed U.
        Identical to the inner loop in RPCA-STOC.
        """
        v = torch.zeros(self.rank, 1, device=self.device)
        s = torch.zeros_like(m)
        UtU = self.U.T @ self.U
        
        for _ in range(self.max_inner_iter):
            # v update: v = (U^T U + lam1 I)^-1 U^T (m - s)
            rhs = self.U.T @ (m - s)
            lhs = UtU + self.lam1 * self.I_r
            v = torch.linalg.solve(lhs, rhs)
            
            # s update: S_lam2(m - Uv)
            residual = m - (self.U @ v)
            s = soft_thresholding(residual, self.lam2)
            
        return v, s