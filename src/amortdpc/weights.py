"""The weight space: the knob's sampling distribution P_w, the policy's encoding phi(w), and the
per-bin loss normalizer n(w).

The reference term has weight 1; the knob (here: track1, the tank-1 tracking weight) is sampled
log-uniform per training window and fed to the policy; other terms have fixed weights.

    phi(w)  = log10(w + lo/10), rescaled so the trained range maps to [-1, 1]
    n(w)    = running mean (EMA, rate 0.05) of J over the windows in w's bin; 12 log bins over [lo, hi]

Padding (`pad`): the knob is trained on [lo, hi] widened by `pad` bin widths at each end (the
paper's generalist uses pad = 1). The claimed range, the dev weights and the evaluation stay [lo, hi].

This is the 1-knob case of the paper's code. norm = "none" (n = 1) is the standard-DPC ablation.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

RUN_BINS, RUN_EMA = 12, 0.05


class WeightSpace:
    def __init__(self, terms: list[str], knob: str, lo: float, hi: float, fixed: dict[str, float],
                 norm: str = "running", pad: int = 0, zero_frac: float = 0.0):
        self.terms, self.knob, self.lo, self.hi = list(terms), knob, lo, hi
        self.zero_frac = zero_frac
        self.fixed = {terms[0]: 1.0, **fixed}
        assert set(self.terms) == set(self.fixed) | {knob}, "every term needs a weight"
        self.norm, self.pad = norm, pad
        self.nbins = RUN_BINS + 2 * pad
        self.running = torch.ones(self.nbins)

    def trained(self) -> tuple[float, float]:
        """The trained range: [lo, hi] widened by `pad` bin widths at each end."""
        r = 10 ** (self.pad * (math.log10(self.hi) - math.log10(self.lo)) / (RUN_BINS - 1))
        return self.lo / r, self.hi * r

    def full(self, w: Tensor) -> Tensor:
        """Knob values [B] -> the full weight matrix [B, nterms]."""
        return torch.stack([w if n == self.knob else torch.full_like(w, self.fixed[n]) for n in self.terms], 1)

    def sample(self, B: int, gen: torch.Generator, fixed: float | None = None, claimed: bool = False) -> Tensor:
        """Knob values [B], log-uniform on the trained range (or on [lo, hi] if `claimed`).
        `fixed` pins the knob: a specialist."""
        if fixed is not None:
            return torch.full((B,), float(fixed))
        lo, hi = (self.lo, self.hi) if claimed else self.trained()
        r = torch.rand(B, generator=gen)
        w = 10 ** (math.log10(lo) + r * (math.log10(hi) - math.log10(lo)))
        zero = torch.rand(B, generator=gen) < self.zero_frac      # optionally, a fraction at exactly w = 0
        return torch.where(zero, torch.zeros_like(w), w)

    def encode(self, w: Tensor) -> Tensor:
        """phi(w) [B, 1]."""
        lo, hi = self.trained()
        a, b = math.log10(lo / 10), math.log10(hi + lo / 10)
        return (2 * (torch.log10(w.clamp(min=0) + lo / 10) - a) / (b - a) - 1)[:, None]

    def _bin(self, w: Tensor) -> Tensor:
        lo, hi = self.trained()
        x = (torch.log10(w.clamp(min=lo)) - math.log10(lo)) / (math.log10(hi) - math.log10(lo))
        k = (x * (self.nbins - 1)).round().long().clamp(0, self.nbins - 1)
        return torch.where(w > 0, k, torch.full_like(k, self.pad))          # w = 0 shares the claimed low edge's bin

    def update_running(self, w: Tensor, J: Tensor):
        b = self._bin(w)
        for i in b.unique():
            self.running[i] = (1 - RUN_EMA) * self.running[i] + RUN_EMA * J[b == i].mean().detach()

    def n(self, w: Tensor) -> Tensor:
        """Per-sample normalizer n(w) [B]."""
        if self.norm == "running":
            return self.running[self._bin(w)]
        if self.norm == "none":
            return torch.ones_like(w)
        raise ValueError(self.norm)

    def dev_score(self, J: Tensor, w: Tensor) -> float:
        """Checkpoint score: the mean over knob bins of log(mean J in the bin). It is the objective
        the running normalizer descends, with no normalizer, so it is comparable across iterations.
        A specialist has one bin, so its score is log mean J."""
        b = self._bin(w)
        return sum(math.log(J[b == i].mean().item()) for i in b.unique()) / len(b.unique())
