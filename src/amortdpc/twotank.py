"""The two-tank plant, its cost terms, and the closed-loop simulator shared by DPC and MPC.

    dh1/dt = c1 (1 - v) p - c2 sqrt(h1)
    dh2/dt = c1 v p + c2 sqrt(h1) - c2 sqrt(h2)          c1 = 0.08, c2 = 0.04

States: the tank levels h = (h1, h2). Inputs: pump p and valve v, both in [0, 0.5]. One control step is
DT = 5 s of explicit Euler in 1 s substeps. sqrt is smoothed so that it is differentiable at 0.

At steady state sqrt(h1) = (1 - v) sqrt(h2) and sqrt(h2) = 2p, so the reachable set is
h2/4 <= h1 <= h2 <= 1. Half of the reference pairs have r1 > r2 and cannot be held, so the tank-1
tracking weight w1 decides the compromise. That weight is the knob the policy is conditioned on.

Cost terms (each averaged over the window's steps):
    track2 = (h2 - r2)^2          the reference term, weight 1
    track1 = (h1 - r1)^2          weight w1, the knob, in [0.01, 100]
    smooth = |u_k - u_{k-1}|^2    fixed weight 0.1 (it does not move the steady state)

The objective is J(w) = sum_j w_j * mean_k term_j. The weights never live inside the plant: they come
in per sample, which is what lets one policy be trained on many of them at once.

`simulate` is the one closed-loop rollout used for DPC training (with autograd through the plant),
for scoring a trained policy, and for scoring MPC. Policies and references see identical dynamics and costs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import torch
from torch import Tensor

C1, C2 = 0.08, 0.04
DT, SUB = 5.0, 5
EPS = 1e-3
U_MAX = 0.5
R_LO, R_HI = 0.1, 0.9
EPISODE = 80                    # closed-loop steps per scored episode (400 s)
TERMS = ["track2", "track1", "smooth"]


# ---- the plant ---------------------------------------------------------------------------------
def ssqrt(h, lib=torch):
    """sqrt(smooth relu(h)): equals sqrt(h) away from 0 and stays differentiable at 0."""
    return lib.sqrt(0.5 * (h + lib.sqrt(h * h + EPS ** 2)))


def f(h1, h2, p, v, lib=torch):
    s1, s2 = ssqrt(h1, lib), ssqrt(h2, lib)
    return C1 * (1 - v) * p - C2 * s1, C1 * v * p + C2 * s1 - C2 * s2


def step_np(h1, h2, p, v, lib=torch):
    """One 5 s control step. `lib` is torch, numpy or casadi: the MPC uses the same map."""
    dt = DT / SUB
    for _ in range(SUB):
        d1, d2 = f(h1, h2, p, v, lib)
        h1, h2 = h1 + dt * d1, h2 + dt * d2
    return h1, h2


# ---- scenarios -----------------------------------------------------------------------------------
@dataclass
class Scenario:
    """A batch of initial conditions and references.
    x0 [B, 2] levels, u0 [B, 2] the input applied just before k = 0, exo {"r": [B, T, 2]} the references."""
    x0: Tensor
    u0: Tensor
    exo: dict[str, Tensor] = field(default_factory=dict)

    def __len__(self):
        return self.x0.shape[0]

    def index(self, idx) -> "Scenario":
        return Scenario(self.x0[idx], self.u0[idx], {k: v[idx] for k, v in self.exo.items()})


class TwoTank:
    nx, nu = 2, 2
    terms = TERMS

    def __init__(self, infeasible_frac: float = 0.5):
        self.u_lo, self.u_hi = torch.zeros(2), torch.full((2,), U_MAX)
        self.infeasible_frac = infeasible_frac

    def refs(self, n, gen):
        """Reference pairs; a fraction `infeasible_frac` has r1 > r2."""
        a = R_LO + (R_HI - R_LO) * torch.rand(n, generator=gen)
        b = R_LO + (R_HI - R_LO) * torch.rand(n, generator=gen)
        hi_first = torch.rand(n, generator=gen) < self.infeasible_frac
        r1 = torch.where(hi_first, torch.maximum(a, b), torch.minimum(a, b))
        r2 = torch.where(hi_first, torch.minimum(a, b), torch.maximum(a, b))
        return torch.stack([r1, r2], -1)

    def sample(self, n, N, gen):
        """Training windows: random levels and previous input, one reference pair held over the window."""
        x0 = 0.05 + 0.9 * torch.rand(n, 2, generator=gen)
        u0 = U_MAX * torch.rand(n, 2, generator=gen)
        r = self.refs(n, gen)[:, None].expand(n, N + 1, 2).clone()
        return Scenario(x0, u0, {"r": r})

    def features(self, x, u_prev, sc, k):
        """The policy's observation (8 numbers, roughly in [-1, 1])."""
        r = sc.exo["r"][:, k]
        return torch.cat([2 * x - 1, 2 * r - 1, 2 * (x - r), u_prev / U_MAX * 2 - 1], -1)

    @property
    def nf(self) -> int:
        return 8

    def step(self, x, u, sc, k):
        h1, h2 = step_np(x[:, 0], x[:, 1], u[:, 0], u[:, 1], torch)
        return torch.stack([h1, h2], -1)

    def stage(self, x, u, u_prev, x_next, sc, k):
        """Unweighted cost terms of step k, each [B]."""
        r = sc.exo["r"][:, k + 1] if sc.exo["r"].shape[1] > k + 1 else sc.exo["r"][:, k]
        e = x_next - r
        return {"track2": e[:, 1] ** 2, "track1": e[:, 0] ** 2, "smooth": ((u - u_prev) ** 2).sum(-1)}


Controller = Callable[[Tensor, Tensor, Scenario, int], Tensor]


def simulate(case: TwoTank, act: Controller, sc: Scenario, N: int):
    """Roll the closed loop for N steps: observe, act, step the plant, record every cost term.
    Returns X [B, N+1, nx], U [B, N, nu], terms name -> [B, N]. Differentiable when grad is enabled."""
    x, up = sc.x0, sc.u0
    X, U = [x], []
    T: dict[str, list[Tensor]] = {n: [] for n in case.terms}
    for k in range(N):
        u = act(x, up, sc, k)
        xn = case.step(x, u, sc, k)
        for n, v in case.stage(x, u, up, xn, sc, k).items():
            T[n].append(v)
        X.append(xn)
        U.append(u)
        x, up = xn, u
    return torch.stack(X, 1), torch.stack(U, 1), {n: torch.stack(v, 1) for n, v in T.items()}


def weighted(terms: dict[str, Tensor], W: Tensor, names: list[str] = TERMS) -> Tensor:
    """Per-sample objective sum_j W[:, j] * mean_k terms_j  -> [B]."""
    return sum(W[:, j] * terms[n].mean(1) for j, n in enumerate(names))


# ---- the held-out test episodes ----------------------------------------------------------------
def eval_set(n: int = 40, steps: int = EPISODE, pad: int = 80, seed: int = 7) -> Scenario:
    """The paper's held-out closed-loop episodes: n reference pairs (each infeasible with probability
    1/2), random initial levels, u_{-1} = 0, each reference held for `steps` steps (padded so an MPC
    horizon can look past the end). The draw does not depend on `steps` or `pad`."""
    gen = torch.Generator().manual_seed(seed)
    case = TwoTank(infeasible_frac=0.5)
    r = case.refs(n, gen)
    x0 = 0.05 + 0.9 * torch.rand(n, 2, generator=gen)
    return Scenario(x0, torch.zeros(n, 2), {"r": r[:, None].expand(n, steps + pad + 1, 2).clone()})


# ---- steady state (used to seed the optimum's solver) -----------------------------------------
def steady_optimum(r1: float, r2: float, w1: float, w2: float = 1.0) -> tuple[float, float]:
    """argmin w1 (h1 - r1)^2 + w2 (h2 - r2)^2 over the reachable set h2/4 <= h1 <= h2 <= 1."""
    def feas(h1, h2):
        return h2 / 4 - 1e-12 <= h1 <= h2 + 1e-12 and 0 <= h2 <= 1 + 1e-12

    cands = [(r1, r2), ((w1 * r1 + w2 * r2) / (w1 + w2),) * 2]
    t = (w1 * r1 / 4 + w2 * r2) / (w1 / 16 + w2)
    cands += [(t / 4, t), (min(max(r1, 0.25), 1.0), 1.0), (0.0, 0.0), (0.25, 1.0), (1.0, 1.0)]
    cands = [(min(max(a, 0), 1), min(max(b, 0), 1)) for a, b in cands]
    return min((c for c in cands if feas(*c)), key=lambda c: w1 * (c[0] - r1) ** 2 + w2 * (c[1] - r2) ** 2)


def steady_input(h1: float, h2: float) -> tuple[float, float]:
    """(p, v) holding the equilibrium (h1, h2)."""
    s1, s2 = np.sqrt(max(h1, 0)), np.sqrt(max(h2, 0))
    return C2 * s2 / C1, 1 - s1 / max(s2, 1e-9)
