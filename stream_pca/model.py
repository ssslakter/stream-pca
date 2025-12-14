import torch, numpy as np
from .utils import get_device


class CompressiveModel:

    def __init__(self, subsample=0.1, device=None):
        self.p, self.device = subsample, device or get_device()
        self.U_flat = None

    def init_basis(self, frames):
        # Robust Init: Median of N frames -> GPU Tensor
        med = np.median(np.array(frames), axis=0).astype(np.float32) / 255.0
        self.U_flat = torch.tensor(med, device=self.device).view(-1, 3)

    def __call__(self, x):
        # x: (H, W, 3) Tensor, normalized 0-1
        if self.U_flat is None:
            raise RuntimeError("Run init_basis first")
        h, w, _ = x.shape
        m_flat = x.view(-1, 3)
        d = m_flat.shape[0]

        # 1. Compressive Mask (Simulate Single Pixel Camera)
        idx = torch.randperm(d, device=self.device)[: int(d * self.p)]

        l_out, s_out = torch.zeros_like(m_flat), torch.zeros_like(m_flat)
        for c in range(3):
            # Projection: v = (u . m) / (u . u)
            m_sub, u_sub = m_flat[idx, c], self.U_flat[idx, c]
            v = torch.dot(u_sub, m_sub) / (torch.dot(u_sub, u_sub) + 1e-5)

            # Reconstruction
            bg = self.U_flat[:, c] * v
            l_out[:, c], s_out[:, c] = bg, m_flat[:, c] - bg

        # 2. Post-Process (Digital Gate)
        L = l_out.view(h, w, 3).clip(0, 1)
        S = (s_out.view(h, w, 3).abs() - 0.15).relu() * 4.0
        return L, S.clip(0, 1)
