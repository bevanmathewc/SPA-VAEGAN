"""
Gradient-based Adaptive Adversarial Weight for SPA-VAEGAN.
Dynamically scales the adversarial loss gradient to be delta * ||grad_rec|| at the generator's final layer.
Prevents GAN collapse, stabilizes joint optimization, and prevents mode hallucination.
"""

from typing import Optional, Tuple
import torch


def raw_adaptive_weight(
    l_rec: torch.Tensor,
    l_adv: torch.Tensor,
    ref_weight: torch.nn.Parameter,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    delta: float = 0.1,
    eps: float = 1e-8
) -> Optional[torch.Tensor]:
    """
    Computes delta * ||grad(L_rec)||_2 / (||grad(L_adv)||_2 + eps) evaluated at ref_weight (G.dec.last.weight).
    Scale factor `s` from GradScaler is included to prevent fp16 gradient underflow,
    and cancels cleanly in the ratio.
    """
    s = scaler.get_scale() if (scaler is not None and scaler.is_enabled()) else 1.0
    
    # Compute gradients with respect to the generator's final conv layer
    g_rec = torch.autograd.grad(l_rec * s, ref_weight, retain_graph=True)[0]
    g_adv = torch.autograd.grad(l_adv * s, ref_weight, retain_graph=True)[0]
    
    n_rec = g_rec.float().norm()
    n_adv = g_adv.float().norm()
    
    # Return None if gradient norms are non-finite (e.g. overflow/underflow during fp16 scaling)
    if not (torch.isfinite(n_rec) and torch.isfinite(n_adv)):
        return None
        
    return (delta * n_rec / (n_adv + eps)).detach()


class AdaptiveWeightTracker:
    """
    Tracks and updates the adaptive adversarial weight lambda_t using exponential moving average (EMA).
    lambda_t = clip(0.9 * lambda_{t-1} + 0.1 * lambda_raw, 0, lambda_max)
    """
    def __init__(self, delta: float = 0.1, lambda_max: float = 1000.0, eps: float = 1e-8):
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
        """
        Updates the tracker state.
        Returns:
            (lambda_to_use, raw_lambda, norm_rec, norm_adv)
        """
        s = scaler.get_scale() if (scaler is not None and scaler.is_enabled()) else 1.0
        g_rec = torch.autograd.grad(l_rec * s, ref_weight, retain_graph=True)[0]
        g_adv = torch.autograd.grad(l_adv * s, ref_weight, retain_graph=True)[0]
        n_rec = g_rec.float().norm().item()
        n_adv = g_adv.float().norm().item()
        
        raw_val = None
        if torch.isfinite(torch.tensor(n_rec)) and torch.isfinite(torch.tensor(n_adv)):
            raw_val = self.delta * n_rec / (n_adv + self.eps)
            if self.lam is None:
                self.lam = raw_val
            else:
                self.lam = 0.9 * self.lam + 0.1 * raw_val
                
        lam_use = 0.0 if self.lam is None else min(max(self.lam, 0.0), self.lambda_max)
        return lam_use, raw_val, n_rec, n_adv
