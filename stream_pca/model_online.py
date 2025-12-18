import torch, numpy as np
from .utils import get_device

def soft_thresholding(X, tau):
    shrinkage = torch.maximum(torch.abs(X) - tau, torch.zeros_like(X))
    return torch.sign(X) * shrinkage

class OnlineStochasticRPCA:
    """
    Truly Online RPCA (Stochastic Optimization) RPCA-STOC
    Updates the background model (U) instantly with every single frame.
    No batch storage required.
    """
    def __init__(self, rank=200, lam1=0.5, lam2=0.01, max_inner_iter=50, device=None):
        self.rank = rank
        self.lam1 = lam1  # low rank U, V
        self.lam2 = lam2  # Sparsity S 
        self.max_inner_iter = max_inner_iter
        
        self.device = device or get_device()
        
        # Internal State (Persists across frames)
        self.U = None   # The Background Basis (D x r)
        self.A = None   # Accumulation Matrix A (r x r)
        self.B = None   # Accumulation Matrix B (D x r)
        
        self.D = 0      # Dimension of one frame
        self.I_r = None # Identity matrix (r x r)

    def init_basis(self, frames):
        """
        Optional: Initialize U using a small set of starting frames (e.g. first 10).
        If skipped, U will be initialized randomly on the first frame.
        """
        if not frames: return
        
        H, W, C = frames[0].shape
        self.D = H * W * C
        
        # TODO  maybe we should do 100 frame in init as warm up??
        # Stack frames to (D x T)
        M_list = [torch.tensor(f, dtype=torch.float32, device=self.device).view(-1) / 255.0 for f in frames]
        M_init = torch.stack(M_list, dim=1)
        
        self._init_internal_state()
        print(f"Online Model Initialized with {len(frames)} warmup frames.")

    def _init_internal_state(self):
        # Random Initialization of U
        self.U = torch.randn(self.D, self.rank, device=self.device)
        self.U = self.U / torch.norm(self.U, dim=0, keepdim=True) # Normalize columns
        
        # Initialize Summary Matrices A and B
        self.A = torch.zeros(self.rank, self.rank, device=self.device)
        self.B = torch.zeros(self.D, self.rank, device=self.device)
        self.I_r = torch.eye(self.rank, device=self.device)

    def __call__(self, x_frame):
        """
        Input: x_frame (H, W, 3) normalized 0-1
        Output: L (Background), S (Foreground)
        
        Side Effect: Updates self.U, self.A, self.B based on this new frame.
        """
        # 0. Setup on First Run (if init_basis was skipped)
        H, W, C = x_frame.shape
        D = H * W * C
        if self.U is None:
            self.D = D
            self._init_internal_state()
            print("Cold Start: Model initialized randomly.")

        # Flatten Input
        m = x_frame.view(-1, 1) # (D, 1)

        # --- Step 1: Project Sample (Find v, s) ---
        # Fix U, solve for coefficients v and sparse noise s
        v = torch.zeros(self.rank, 1, device=self.device)
        s = torch.zeros_like(m)
        UtU = self.U.T @ self.U
        
        # Inner loop (Alternating Minimization) to find v and s 
        for _ in range(self.max_inner_iter):
            # v update: (U^T U + lam1 I) v = U^T (m - s)
            rhs = self.U.T @ (m - s)
            lhs = UtU + self.lam1 * self.I_r
            v = torch.linalg.solve(lhs, rhs)
            
            # s update: S_lam2(m - Uv)
            residual = m - (self.U @ v)
            s = soft_thresholding(residual, self.lam2)

        # --- Step 2: Update Model (Update U) ---
        # Update summary matrices
        # A <- A + v v^T
        self.A += v @ v.T
        # B <- B + (m - s) v^T
        self.B += (m - s) @ v.T
        
        # # Update Basis U (algorithm 3)
        # A_tilde = self.A + self.lam1 * self.I_r
        # # Iterate over each column of U
        # for j in range(self.rank):
        #     # Get j-th column of B
        #     # b_j = self.B[:, j].unsqueeze(1)  # m × 1
            
        #     # Get j-th row of Ã (excluding diagonal)
        #     # a_tilde_j = A_tilde[j, :].unsqueeze(0)  # 1 × r

        #     # Compute U * ã_j (excluding the j-th column's contribution to itself)
        #     # We need to be careful: when we compute U * ã_j, we should use
        #     # the current U but with the j-th column set to zero for this operation
        #     U_temp = self.U.clone()
        #     U_temp[:, j] = 0  # Zero out the j-th column for this computation
        #     U_a_tilde = U_temp @ A_tilde[j, :].unsqueeze(0).T  # m × 1
            
        #     # Current j-th column of U
        #     # u_j = self.U[:, j].unsqueeze(1)  # m × 1

        #     # Update: ũ_j = (1/Ã[j,j]) * (b_j - U * ã_j) + u_j
        #     u_tilde = (1.0 / A_tilde[j, j]) * (self.B[:, j].unsqueeze(1) - U_a_tilde) + self.U[:, j].unsqueeze(1)
            
        #     # Normalize: u_j = ũ_j / max(||ũ_j||₂, 1)
        #     norm_u_tilde = torch.norm(u_tilde, p=2)
        #     if norm_u_tilde > 1:
        #         u_tilde = u_tilde / norm_u_tilde
            
        #     # Update the j-th column of U
        #     self.U[:, j] = u_tilde.squeeze(1)

        # U = B (A + lam1 I)^-1
        lhs_U = self.A + self.lam1 * self.I_r
        # Solve U @ lhs_U = B  =>  U = (lhs_U^-1 @ B^T)^T
        self.U = torch.linalg.solve(lhs_U, self.B.T).T

        # --- Step 3: Return Result ---
        L_vec = (self.U @ v).view(H, W, C)
        S_vec = (s.view(H, W, C).abs() - 0.01).relu()
        
        return L_vec.clip(0, 1), S_vec.clip(0, 1)