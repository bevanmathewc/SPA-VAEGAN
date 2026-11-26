
"""
Gradient-based Adaptive Adversarial Weight for SPA-VAEGAN.

Computes the ratio of reconstruction and adversarial gradient norms
at the generator's final decoder layer.
"""

from typing import Optional, Tuple
import math
import torch


def raw_adaptive_weight(
    l_rec: torch.Tensor,
    l_adv: torch.Tensor,
    ref_weight: torch.nn.Parameter,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    delta: float = 0.1,
    eps: float = 1e-8
) -> Optional[torch.Tensor]:

    g_rec = torch.autograd.grad(
        l_rec, ref_weight, retain_graph=True
    )[0]

    g_adv = torch.autograd.grad(
        l_adv, ref_weight, retain_graph=True
    )[0]

    if not torch.isfinite(g_rec).all() or not torch.isfinite(g_adv).all():
        return None

    n_rec = torch.linalg.vector_norm(g_rec.detach().to(torch.float64))
    n_adv = torch.linalg.vector_norm(g_adv.detach().to(torch.float64))

    if not torch.isfinite(n_rec) or not torch.isfinite(n_adv):
        return None

    weight = delta * n_rec / (n_adv + eps)

    if not torch.isfinite(weight):
        return None

    return weight.detach().to(torch.float32)


class AdaptiveWeightTracker:
    """
    EMA adaptive adversarial weight:
    lambda_t = 0.9 * lambda_(t-1) + 0.1 * lambda_raw
    """

    def __init__(
        self,
        delta: float = 0.1,
        lambda_max: float = 1000.0,
        eps: float = 1e-8
    ):
        self.delta = delta
        self.lambda_max = lambda_max
        self.eps = eps
        self.lam: Optional[float] = None

    def step(
        self,
        l_rec: torch.Tensor,
        l_adv: torch.Tensor,
        ref_weight: torch.nn.Parameter,
        scaler: Optional[torch.cuda.amp.GradScaler] = None
    ) -> Tuple[float, Optional[float], float, float]:

        g_rec = torch.autograd.grad(
            l_rec, ref_weight, retain_graph=True
        )[0]

        g_adv = torch.autograd.grad(
            l_adv, ref_weight, retain_graph=True
        )[0]

        if not torch.isfinite(g_rec).all() or not torch.isfinite(g_adv).all():
            n_rec = float("nan")
            n_adv = float("nan")
            raw_val = None
        else:
            n_rec = torch.linalg.vector_norm(
                g_rec.detach().to(torch.float64)
            ).item()

            n_adv = torch.linalg.vector_norm(
                g_adv.detach().to(torch.float64)
            ).item()

            raw_val = None

            if math.isfinite(n_rec) and math.isfinite(n_adv):
                raw_val = self.delta * n_rec / (n_adv + self.eps)

                if math.isfinite(raw_val):
                    raw_val = min(max(raw_val, 0.0), self.lambda_max)

                    if self.lam is None:
                        self.lam = raw_val
                    else:
                        self.lam = 0.9 * self.lam + 0.1 * raw_val

        if self.lam is None or not math.isfinite(self.lam):
            lam_use = 0.0
        else:
            lam_use = min(max(self.lam, 0.0), self.lambda_max)

        return lam_use, raw_val, n_rec, n_adv
