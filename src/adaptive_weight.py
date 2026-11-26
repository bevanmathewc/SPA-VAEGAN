
"""
Gradient-based Adaptive Adversarial Weight for SPA-VAEGAN.

Computes the ratio of reconstruction and adversarial gradient norms
at the generator's final decoder layer.

Gradient norms are calculated WITHOUT GradScaler scaling.
GradScaler remains responsible for the actual training backward pass.
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

    # Calculate unscaled gradients for adaptive-weight estimation.
    g_rec = torch.autograd.grad(
        l_rec,
        ref_weight,
        retain_graph=True
    )[0]

    g_adv = torch.autograd.grad(
        l_adv,
        ref_weight,
        retain_graph=True
    )[0]

    # Use float64 for stable norm calculation.
    n_rec = torch.linalg.vector_norm(g_rec.detach().to(torch.float64))
    n_adv = torch.linalg.vector_norm(g_adv.detach().to(torch.float64))

    if not torch.isfinite(n_rec) or not torch.isfinite(n_adv):
        return None

    weight = delta * n_rec / (n_adv + eps)

    if not torch.isfinite(weight):
        return None

    return weight.detach().to(dtype=torch.float32)


class AdaptiveWeightTracker:
    """
    Tracks the adaptive adversarial weight using an exponential
    moving average:

    lambda_t = 0.9 * lambda_(t-1) + 0.1 * lambda_raw

    The result is clipped to [0, lambda_max].
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

        # Calculate gradients WITHOUT GradScaler multiplication.
        g_rec = torch.autograd.grad(
            l_rec,
            ref_weight,
            retain_graph=True
        )[0]

        g_adv = torch.autograd.grad(
            l_adv,
            ref_weight,
            retain_graph=True
        )[0]

        # Convert to float64 before calculating norms.
        n_rec_tensor = torch.linalg.vector_norm(
            g_rec.detach().to(torch.float64)
        )

        n_adv_tensor = torch.linalg.vector_norm(
            g_adv.detach().to(torch.float64)
        )

        n_rec = n_rec_tensor.item()
        n_adv = n_adv_tensor.item()

        raw_val = None

        # Update only when both gradient norms are finite.
        if (
            math.isfinite(n_rec)
            and math.isfinite(n_adv)
            and n_rec >= 0
            and n_adv >= 0
        ):

            raw_val = self.delta * n_rec / (n_adv + self.eps)

            # Prevent an excessively large raw weight from
            # destabilising the EMA.
            raw_val = min(raw_val, self.lambda_max)

            if math.isfinite(raw_val):
                if self.lam is None:
                    self.lam = raw_val
                else:
                    self.lam = 0.9 * self.lam + 0.1 * raw_val

        # Keep the final adaptive weight within configured bounds.
        if self.lam is None or not math.isfinite(self.lam):
            lam_use = 0.0
        else:
            lam_use = min(max(self.lam, 0.0), self.lambda_max)

        return lam_use, raw_val, n_rec, n_adv
