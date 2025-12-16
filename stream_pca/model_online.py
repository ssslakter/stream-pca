import torch, numpy as np
from .utils import get_device

# --- ADMM Helper Functions (Proximal Operators) ---

def svt(X, tau):
    """Singular Value Thresholding Operator D_tau(X)."""
    # 1. Compute SVD: X = U * Sigma * V^T
    U, S, V = torch.svd(X)
    
    # 2. Apply Soft-Thresholding to singular values (Sigma)
    # S_tau(sigma) = max(|sigma| - tau, 0)
    S_thr = torch.maximum(S - tau, torch.zeros_like(S))
    
    # 3. Reconstruct the matrix
    # Note: U and V are implicitly U[:, :S_thr.size(0)] and V[:, :S_thr.size(0)]
    # due to how torch.svd works on non-square matrices and how torch.diag handles vector input.
    # The reconstruction is U * S_thr_diag * V^T
    return U @ torch.diag(S_thr) @ V.transpose(-2, -1)


def soft_thresholding(X, tau):
    """Element-wise Soft Thresholding Operator S_tau(X)."""
    # S_tau[x] = sgn(x) * max(|x| - tau, 0)
    return torch.sign(X) * torch.maximum(torch.abs(X) - tau, torch.zeros_like(X))


# --- The ADMM Model ---

class OnlineRobustPCA:
    """
    Implements Online Robust Principal Component Analysis (PCP)
    using the ADMM/ALM formulation.
    """
    def __init__(self, lam=0.1, mu=0.01, max_iter=100, device=None):
        # Hyperparameters
        self.lam = lam         # lambda (weight for sparsity)
        self.mu = mu           # mu (penalty parameter)
        self.max_iter = max_iter # Number of ADMM iterations per frame

        self.device = device or get_device()
        self.H, self.W, self.C = 0, 0, 0 # Stores image dimensions
        
        # ADMM variables (initialized on first call)
        self.L, self.S, self.Y = None, None, None

    def init_basis(self, frames):
        """
        Initializes the model. In an online setting, we don't fully initialize 
        the matrices L, S, Y for the entire video, but we do get the dimensions 
        and an initial L/S for the first frame if needed.
        
        For simplicity, we only capture the dimensions here.
        The full matrices L, S, Y will be initialized on the first call to __call__.
        """
        if not frames:
            raise ValueError("Frames list is empty for initialization.")
        
        # Dimensions are H, W, 3 (RGB)
        frame_shape = frames[0].shape 
        self.H, self.W, self.C = frame_shape[0], frame_shape[1], frame_shape[2]
        
        # The flattened data dimension (D = H * W * C)
        self.D = self.H * self.W * self.C
        
        print(f"Model initialized. Data dimension D: {self.D}")

    def __call__(self, x_frame):
        """
        Processes a single frame x_frame (H, W, 3) tensor normalized 0-1.
        
        In the online approach, we process one frame at a time, where the
        input matrix M is just the *current* frame. This is a simplification
        of a true online/streaming ADMM, but implements the core PCP step 
        per frame.
        
        The input x_frame is converted to a vector M (D x 1) for matrix operations.
        """
        # Ensure dimensions are set
        if self.D == 0:
             raise RuntimeError("Run init_basis first to set dimensions")

        # M is the current frame, flattened to a column vector (D x 1)
        M = x_frame.view(self.D, 1)

        # 1. Initialize ADMM variables for the current frame if not already done
        if self.L is None:
            # Initialize L with the frame, S and Y with zeros
            self.L = M.clone().detach()
            self.S = torch.zeros_like(M)
            self.Y = torch.zeros_like(M)

        # We keep the previous L, S, Y from the last frame to warm-start
        L_k, S_k, Y_k = self.L, self.S, self.Y
        norm_err = None
        # 2. ADMM Iterations (inner loop for convergence)
        for _ in range(self.max_iter):
            # A. L-update (Line 4) - Proximal operator of the nuclear norm (SVT)
            # W = M - S^k + (1/mu) * Y^k
            W = M - S_k +  Y_k
            # L^{k+1} = D_{1/mu}(W)
            # NOTE: svt is defined for matrices. We're using a (D x 1) matrix, 
            # so the SVT operation is equivalent to soft-thresholding on the single value.
            # To get a true Low-Rank-like structure, we would need to un-flatten
            # the frames into a *matrix* where columns are frames, but for simplicity
            # of a per-frame update, we perform it on the (D x 1) 'matrix'.
            L_k_plus_1 = svt(W, 1 / self.mu)
            
            # B. S-update (Line 5) - Proximal operator of the L1 norm (Soft-Thresholding)
            # Z = M - L^{k+1} + (1/mu) * Y^k
            Z = M - L_k_plus_1 + Y_k
            # S^{k+1} = S_{lambda/mu}(Z)
            S_k_plus_1 = soft_thresholding(Z, self.lam / self.mu)
            
            # C. Y-update (Line 6) - Dual Variable Update
            # Y^{k+1} = Y^k + mu * (M - L^{k+1} - S^{k+1})
            Y_k_plus_1 = Y_k + (M - L_k_plus_1 - S_k_plus_1)
            norm_err = torch.linalg.norm((M - L_k_plus_1 - S_k_plus_1))

            # Update for next iteration
            L_k, S_k, Y_k = L_k_plus_1, S_k_plus_1, Y_k_plus_1
        print("the error ADMM for current frame is ", norm_err)
        # 3. Store final results and reshape for output
        self.L, self.S, self.Y = L_k, S_k, Y_k
        
        # Reshape the output back to (H, W, 3) image format
        L_out = L_k.view(self.H, self.W, self.C).clip(0, 1)
        S_out = S_k.view(self.H, self.W, self.C).clip(0, 1)

        return L_out, S_out