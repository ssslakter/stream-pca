import torch
from collections import deque
from .model_online_dynamic import OnlineMovingWindowRPCA
from .utils import get_device

class OnlineMovingWindowRPCA_CP(OnlineMovingWindowRPCA):
    def __init__(self, rank=200, lam1=0.5, lam2=0.005, n_win=50, n_burnin=100,
                 max_inner_iter=50, device=None,
                 Ncheck=50, alpha_prop=0.5, alpha=0.01, ntest=50,
                 ncp_burnin=None, npositive=3, ntol=0):
        super().__init__(rank, lam1, lam2, n_win, n_burnin, max_inner_iter, device)
        self.Ncheck = Ncheck
        self.alpha_prop = alpha_prop
        self.alpha = alpha
        self.ntest = ntest
        self.ncp_burnin = ncp_burnin or n_win
        self.npositive = npositive
        self.ntol = ntol

        self.Hc = None
        self.Bf = deque()
        self.Bc = deque()
        self.change_points = []
        self.restart_requested = False
        self.requested_t0 = None
        self.tstart = None

    def _init_internal_state(self, M_burnin, L_burnin, S_burnin, rank):
        super()._init_internal_state(M_burnin, L_burnin, S_burnin, rank)
        self.tstart = self.t
        # initialize histogram once D known
        self.Hc = torch.zeros(self.D + 1, dtype=torch.long)

    def __call__(self, x_frame):
        if self.U is None:
            raise RuntimeError("Must call init_basis() first to perform burn-in.")

        self.t += 1
        H, W, C = x_frame.shape
        m = x_frame.view(-1, 1)

        # project
        v, s = self._project_sample(m)

        # sliding window update (same order as base)
        if len(self.v_history) >= self.n_win:
            v_old = self.v_history.pop(0)
            s_old = self.s_history.pop(0)
            m_old = self.m_history.pop(0)
            self.A -= v_old @ v_old.T
            self.B -= (m_old - s_old) @ v_old.T

        self.A += v @ v.T
        self.B += (m - s) @ v.T
        self.v_history.append(v)
        self.s_history.append(s)
        self.m_history.append(m)

        # update basis
        lhs_U = self.A + self.lam1 * self.I_r
        self.U = torch.linalg.solve(lhs_U, self.B.T).T

        # compute support size
        ct = int((s.abs() > 1e-8).sum().item())

        # stages per Algorithm 5
        if self.t < (self.tstart + self.ncp_burnin):
            # still waiting for subspace to stabilize
            pass
        elif self.t < (self.tstart + self.ncp_burnin + self.ntest):
            # collect initial histogram without testing
            self.Hc[ct] += 1
        else:
            total = int(self.Hc.sum().item())
            if total == 0:
                p = 1.0
            else:
                lo = max(0, ct - self.ntol)
                p = float(self.Hc[lo:].sum().item()) / float(total)
            ft = 1 if p <= self.alpha else 0

            # buffers
            self.Bf.append(ft)
            self.Bc.append(ct)
            if len(self.Bf) > self.Ncheck:
                c = self.Bc.popleft()
                f = self.Bf.popleft()
                self.Hc[c] += 1

            # detection
            if len(self.Bf) == self.Ncheck:
                nabnormal = sum(self.Bf)
                if nabnormal >= self.alpha_prop * self.Ncheck:
                    # find first run of npositive consecutive abnormal flags
                    bf = list(self.Bf)
                    run = 0
                    for i, val in enumerate(bf):
                        if val == 1:
                            run += 1
                            if run >= self.npositive:
                                idx_in_buffer = i - (self.npositive - 1)
                                cp_time = self.t - (self.Ncheck - idx_in_buffer)
                                self.change_points.append(cp_time)
                                self.restart_requested = True
                                self.requested_t0 = cp_time
                                print(f"[OMWRPCA-CP] Change point detected at t={cp_time}")
                                break
                        else:
                            run = 0

        L_vec = (self.U @ v).view(H, W, C)
        S_vec = (s.view(H, W, C).abs() - 0.01).relu()
        return L_vec.clip(0, 1), S_vec.clip(0, 1)