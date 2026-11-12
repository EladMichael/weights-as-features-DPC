"""The policy: an MLP whose input is the observation with the weight encoding phi(w) and the
time-to-go encoding psi(tau) concatenated to it.

    u = head( MLP( [o_k, phi(w), psi(tau)] ) )      MLP: 3 hidden layers of 64, GELU
    head(z) = lo + (hi - lo) * sigmoid(z)             keeps u inside its bounds

That concatenation is the whole architectural change from standard DPC. A specialist is the same
network with its weight pinned.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class Head(nn.Module):
    """Bounded output: the formula of Neuromancer's MLP_bounds (re-implemented here)."""

    def __init__(self, lo: Tensor, hi: Tensor):
        super().__init__()
        self.register_buffer("lo", lo.float())
        self.register_buffer("hi", hi.float())

    def forward(self, z: Tensor) -> Tensor:
        return self.lo + (self.hi - self.lo) * torch.sigmoid(z)


class ConcatPolicy(nn.Module):
    def __init__(self, nf: int, nphi: int, nu: int, lo: Tensor, hi: Tensor, N: int,
                 hidden: int = 64, layers: int = 3):
        super().__init__()
        dims = [nf + nphi + 1] + [hidden] * layers          # +1: the time-to-go input
        mods = []
        for a, b in zip(dims[:-1], dims[1:]):
            mods += [nn.Linear(a, b), nn.GELU()]
        self.body = nn.Sequential(*mods, nn.Linear(hidden, nu))
        self.head = Head(lo, hi)
        self.N = N                                           # the training window tau is encoded against

    def tau_input(self, tau: float, B: int) -> Tensor:
        """psi(tau) = 2 log tau / log N - 1: tau in [1, N] maps to [-1, 1]."""
        return torch.full((B, 1), 2 * math.log(tau) / math.log(self.N) - 1)

    def forward(self, obs: Tensor, phi: Tensor, tau: float) -> Tensor:
        return self.head(self.body(torch.cat([obs, phi, self.tau_input(tau, len(obs))], -1)))


def actor(case, policy: ConcatPolicy, phi: Tensor, tau=None):
    """The closed-loop controller act(x, u_prev, sc, k). `tau(k)` gives the time-to-go at step k.
    The default is tau = N at every step: deployed as a receding-horizon controller with horizon N."""
    tau = tau or (lambda k: policy.N)
    return lambda x, up, sc, k: policy(case.features(x, up, sc, k), phi, tau(k))
