import torch, numpy as np
from .utils import get_device

# --- ADMM Helper Functions (Proximal Operators) ---

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


# --- The ADMM Model (Now performing Batch RPCA) ---
class FullRobustPCA:
    """
    run RPCA on full frame
    """
    def __init__(self, lam=0.006, mu=1, max_iter=50, device=None):
        self.lam = lam         # lambda (weight for sparsity)
        self.mu = mu           # mu (penalty parameter)
        self.max_iter = max_iter # Number of ADMM iterations for the whole batch
        self.device = device or get_device()
        
        self.H, self.W, self.C = 0, 0, 0  # image dims
        self.L_batch = None # for background
        self.S_batch = None # for foreground
        self.current_frame_idx = 0

    def init_basis(self, frames):
        """
        1. Forms the data matrix M (D x T).
        2. Executes the full ADMM algorithm on M.
        3. Stores the resulting low-rank (L) and sparse (S) matrices.
        """
        if not frames:
            raise ValueError("Frames list is empty for initialization.")
        
        T = len(frames) # Number of frames
        self.H, self.W, self.C = frames[0].shape
        self.D = self.H * self.W * self.C
        
        # 1. Form the data matrix M (D x T)
        M_list = [
            torch.tensor(f, dtype=torch.float32, device=self.device).view(-1) / 255.0 
            for f in frames
        ]
        M = torch.stack(M_list, dim=1) # hwc x num_frame
        print(f"\n--- Running Batch RPCA ---")
        print(f"Matrix M size: {self.D} x {T}. Running for max {self.max_iter} iterations...")
        
        # 2. ADMM
        L_k = torch.zeros_like(M)
        S_k = torch.zeros_like(M)
        Y_k = torch.zeros_like(M) # scaled dual var

        # thresholds
        tau_L = 1 / self.mu  # Threshold for SVT
        tau_S = self.lam / self.mu # Threshold for Soft-Thresholding
        
        norm_M = torch.linalg.norm(M, ord='fro')
        
        for k in range(self.max_iter):
            # A. L-update
            # W = M - S^k + Y^k
            W = M - S_k +  Y_k
            L_k_plus_1 = svt(W, tau_L)
            
            # B. S-update 
            # Z = M - L^{k+1} + Y^k
            Z = M - L_k_plus_1 + Y_k
            S_k_plus_1 = soft_thresholding(Z, tau_S)
            
            # C. Y-update
            # Y^{k+1} = Y^k + (M - L^{k+1} - S^{k+1})
            R_k_plus_1 = M - L_k_plus_1 - S_k_plus_1 # Residual
            Y_k_plus_1 = Y_k + R_k_plus_1
            
            # Convergence
            residual_norm = torch.linalg.norm(R_k_plus_1, ord='fro')
            if residual_norm / norm_M < 1e-7:
                print(f"\nADMM Converged at iteration {k}.")
                break
            
            if k % 10 == 0:
                print(f"ADMM Iteration {k}/{self.max_iter}: Residual = {residual_norm.item():.4f}", end='\r')
            L_k, S_k, Y_k = L_k_plus_1, S_k_plus_1, Y_k_plus_1

        #  final results
        self.L_batch = L_k.cpu()
        self.S_batch = S_k.cpu()

        # --- Sparsity Check ---
        S_np = self.S_batch.numpy()
        epsilon = 1e-4
        num_non_zero_elements = np.sum(np.abs(S_np) > epsilon)
        total_elements = S_np.size
        sparsity_ratio = num_non_zero_elements / total_elements * 100

        print(f"Total elements (pixels x frames): {total_elements}")
        print(f"Number of 'non-zero' elements in S (where |x| > {epsilon}): {num_non_zero_elements}")
        print(f"Sparsity Percentage: {sparsity_ratio:.4f}% of all elements are non-sparse.")
        
        print("\nBatch RPCA completed. Results stored.")

    def __call__(self, x_frame):
        """
        Retrieves the pre-calculated background (L) and foreground (S) 
        column for the current frame index (t).
        """
        if self.L_batch is None:
             raise RuntimeError("Batch RPCA must be run in init_basis first.")

        t = self.current_frame_idx
        
        # if out of range, use last frame
        if t >= self.L_batch.shape[1]:
            L_col = self.L_batch[:, -1]
            S_col = torch.zeros_like(L_col)
            
        else:
            # extract the splitted frame
            L_col = self.L_batch[:, t]
            S_col = self.S_batch[:, t]

        L_out = L_col.view(self.H, self.W, self.C).to(self.device).clip(0, 1)
        S_out = S_col.view(self.H, self.W, self.C).to(self.device).clip(0, 1)
        self.current_frame_idx += 1
        return L_out, S_out * 4