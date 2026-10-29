"""Fair Cast: unbiased fp32 -> bf16 export of RL master weights, plus a helper to train on the served copy.

The whole method is two changes to an existing RL loop:

1. Export with an unbiased rounding rule instead of round-to-nearest
   (``sr_cast_bf16`` or ``ef_cast_bf16``), so that E[served] == master and no
   sub-half-ulp update is systematically withheld from the sampler.
2. Compute the policy-gradient loss on the exact served bf16 copy, with gradients
   passed straight through to the fp32 master (``ste_params`` / ``served_forward``).

Both halves are required (paper, Sec. 5 and 7.1).
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Dict, Iterator, Optional

import torch

__all__ = [
    "sr_cast_bf16",
    "ef_cast_bf16",
    "nearest_cast_bf16",
    "ema_update_",
    "export_bf16",
    "sync_to_sampler",
    "ste_params",
    "served_forward",
]


def nearest_cast_bf16(master: torch.Tensor) -> torch.Tensor:
    """Round-to-nearest export (the default in current RL frameworks). Kept for A/B comparisons."""
    return master.to(torch.bfloat16)


def sr_cast_bf16(master: torch.Tensor, gen: Optional[torch.Generator] = None) -> torch.Tensor:
    """Stochastic rounding fp32 -> bf16 with E[q] == master exactly and no state.

    Adds 16 uniformly distributed random bits to the fp32 bit pattern and truncates the low
    16 bits. The upper bf16 neighbour is selected with probability equal to the fractional
    position of ``master`` between its two bf16 neighbours. Sign bit is untouched, so the
    rule is correct for both signs; NaN/Inf inputs are passed through unchanged.
    """
    if master.dtype != torch.float32:
        raise TypeError(f"sr_cast_bf16 expects float32 master weights, got {master.dtype}")
    bits = master.contiguous().view(torch.int32)
    noise = torch.randint(0, 1 << 16, bits.shape, dtype=torch.int32, device=master.device, generator=gen)
    q = ((bits + noise) & -65536).view(torch.float32).to(torch.bfloat16)
    finite = torch.isfinite(master)
    if not bool(finite.all()):
        q = torch.where(finite, q, master.to(torch.bfloat16))
    return q


def ef_cast_bf16(master: torch.Tensor, resid: torch.Tensor) -> torch.Tensor:
    """Error-feedback export: q_t = round(theta_t + r_{t-1}), r_t = theta_t + r_{t-1} - q_t.

    Deterministic, no injected noise; the served copy tracks the master up to a residual
    bounded by half a bf16 spacing that never accumulates. ``resid`` is one fp32 tensor per
    parameter, updated in place, and must be zero-initialised before the first export.
    """
    if master.dtype != torch.float32:
        raise TypeError(f"ef_cast_bf16 expects float32 master weights, got {master.dtype}")
    y = master + resid
    q = y.to(torch.bfloat16)
    resid.copy_(y - q.float())
    return q


@torch.no_grad()
def ema_update_(ema: torch.Tensor, master: torch.Tensor, alpha: float) -> torch.Tensor:
    """m_t = (1 - alpha) m_{t-1} + alpha theta_t. Export ``ema`` with an unbiased rule to add an explicit lag of ~1/alpha steps (paper, Sec. 7.5)."""
    return ema.mul_(1.0 - alpha).add_(master, alpha=alpha)


@torch.no_grad()
def export_bf16(
    master: torch.Tensor,
    mode: str = "sr",
    gen: Optional[torch.Generator] = None,
    resid: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Dispatch on ``mode``: ``"nearest"`` (baseline), ``"sr"`` (stochastic rounding), ``"ef"`` (error feedback)."""
    if mode == "nearest":
        return nearest_cast_bf16(master)
    if mode == "sr":
        return sr_cast_bf16(master, gen)
    if mode == "ef":
        if resid is None:
            raise ValueError("mode='ef' needs a residual tensor")
        return ef_cast_bf16(master, resid)
    raise ValueError(f"unknown export mode {mode!r}; expected 'nearest', 'sr' or 'ef'")


@torch.no_grad()
def sync_to_sampler(
    master_params: Dict[str, torch.Tensor],
    served_params: Dict[str, torch.Tensor],
    mode: str = "sr",
    gen: Optional[torch.Generator] = None,
    resid: Optional[Dict[str, torch.Tensor]] = None,
) -> None:
    """Drop-in replacement for the ``served.copy_(master)`` line of a weight-sync hook.

    ``master_params`` / ``served_params`` map parameter names to fp32 / bf16 tensors. The served
    tensors must be the ones used by BOTH the rollout engine and the trainer's forward pass.
    """
    for name, w in master_params.items():
        served_params[name].copy_(export_bf16(w, mode, gen, None if resid is None else resid[name]))


def ste_params(model: torch.nn.Module, served: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Straight-through parameters for ``torch.func.functional_call``.

    Returns ``{name: (theta + sg(q - theta)).to(bf16)}``: equal to the served copy ``q`` in value,
    with d(w_eff)/d(theta) = I so the gradient lands on the fp32 master. This is the form used by
    the paper's harness (``forward_mode: "ste"``)::

        logits = functional_call(model, ste_params(model, served), (), dict(input_ids=ids)).logits
    """
    return {n: (p + (served[n].float() - p.detach())).to(torch.bfloat16) for n, p in model.named_parameters()}


@contextmanager
def served_forward(model: torch.nn.Module, served: Dict[str, torch.Tensor]) -> Iterator[None]:
    """Run ``model``'s forward on the served bf16 values while keeping fp32 parameter containers.

    Inside the block each parameter's ``.data`` is swapped for ``served[name].float()``; on exit the
    fp32 master is restored (before ``optimizer.step()``). Gradients computed inside land on the
    master's ``.grad``. Lighter than ``ste_params`` when ``functional_call`` is inconvenient::

        with served_forward(master_model, served):
            loss = policy_gradient_loss(master_model(ids).logits, ...)
            loss.backward()
        optimizer.step()
    """
    saved = {}
    try:
        for n, p in model.named_parameters():
            saved[n] = p.data
            p.data = served[n].data.float()
        yield
    finally:
        for n, p in model.named_parameters():
            if n in saved:
                p.data = saved[n]
