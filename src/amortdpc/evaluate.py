"""Closed-loop scoring of a trained policy over a sweep of the knob.

Every score is closed loop, on the true plant, with the true weights, over the 40 held-out
80-step episodes of `twotank.eval_set`. Training loss is never used as a score.

Regret at weight w is J_ctrl(w) / J_ref(w) - 1, where J(w) is the weighted cost averaged over the
episodes, and J_ref is the reference's (the MPC, or the episode optimum) on the same episodes.
"""

from __future__ import annotations

import numpy as np
import torch

from .policy import actor
from .twotank import EPISODE, TwoTank, eval_set, simulate, weighted


@torch.no_grad()
def sweep(policy, ws, w1, sc=None, steps: int = EPISODE):
    """J [nw, ns]: each episode of `sc` (default: the 40 held-out ones) at each weight in w1."""
    case, sc = TwoTank(), sc or eval_set()
    w1 = torch.as_tensor(np.asarray(w1), dtype=torch.float32)
    nw, ns = len(w1), len(sc)
    from .twotank import Scenario
    big = Scenario(sc.x0.repeat(nw, 1), sc.u0.repeat(nw, 1), {k: v.repeat(nw, 1, 1) for k, v in sc.exo.items()})
    w = w1.repeat_interleave(ns)
    _, _, T = simulate(case, actor(case, policy, ws.encode(w)), big, steps)
    return weighted(T, ws.full(w)).view(nw, ns).double().numpy()


def regret(J: np.ndarray, J_ref: np.ndarray) -> np.ndarray:
    """Regret per weight: the episodes' mean cost against the reference's. J, J_ref [nw, ns] -> [nw]."""
    J, J_ref = np.asarray(J).mean(-1), np.asarray(J_ref).mean(-1)
    return J / J_ref - 1


def references(path):
    """The paper's references on the 40 held-out episodes, from data/twotank_references.npz:
    w1 [25] and J [25, 40] for the 80-step MPC and for the episode optimum."""
    z = np.load(path)
    J = lambda p: z[f"{p}_track2"] + z["w1"][:, None] * z[f"{p}_track1"] + float(z["w_smooth"]) * z[f"{p}_smooth"]
    return dict(w1=z["w1"], mpc=J("mpc"), opt=J("opt"), U_mpc=z["U_mpc"], U_opt=z["U_opt"])
