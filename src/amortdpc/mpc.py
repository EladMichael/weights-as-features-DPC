"""The references DPC is compared with: nonlinear MPC and the episode optimum (CasADi + IPOPT).

TankNLP is the H-step optimal control problem on the same discrete two-tank map, with the same
weights as the policy:

    min  sum_{k<H} (h2 - r2)^2 + w1 (h1 - r1)^2 + w_s |u_k - u_{k-1}|^2      u in [0, 0.5]^2

- As a controller (`__call__`) it is receding-horizon MPC: solved at every step, warm-started from its
  previous solution shifted by one. With H = 80 (the policy's window) it is the MPC that a policy
  deployed at tau = 80 targets.
- `optimum` solves it once with H = the whole episode. The plant is deterministic and known, so the
  best closed loop over an episode is the best open-loop input sequence. IPOPT is local, so the
  problem is solved from several initial guesses and the best is kept. In the paper the guesses
  were: hold the initial input, the steady-state input, 5 random piecewise-constant inputs, the MPC
  closed loops at 4 horizons, then every DPC closed loop as a further guess ("polish").
"""

from __future__ import annotations

import time

import numpy as np
import torch

from .twotank import EPISODE, U_MAX, TwoTank, eval_set, simulate, steady_input, steady_optimum, step_np, weighted


class TankNLP:
    def __init__(self, H: int, w1: float, w_smooth: float = 0.1):
        import casadi as ca
        t0 = time.perf_counter()
        self.H = H
        opti = ca.Opti()
        X, U = opti.variable(2, H + 1), opti.variable(2, H)
        self.x0, self.up, self.r = opti.parameter(2), opti.parameter(2), opti.parameter(2, H)
        opti.subject_to(X[:, 0] == self.x0)
        J = 0
        for k in range(H):
            h1, h2 = step_np(X[0, k], X[1, k], U[0, k], U[1, k], ca)
            opti.subject_to(X[0, k + 1] == h1)
            opti.subject_to(X[1, k + 1] == h2)
            du = U[:, k] - (self.up if k == 0 else U[:, k - 1])
            J += (X[1, k + 1] - self.r[1, k]) ** 2 + w1 * (X[0, k + 1] - self.r[0, k]) ** 2 + w_smooth * ca.sumsqr(du)
        opti.subject_to(opti.bounded(0, ca.vec(U), U_MAX))
        opti.minimize(J)
        opti.solver("ipopt", {"print_time": False, "ipopt.print_level": 0, "ipopt.sb": "yes",
                              "ipopt.tol": 1e-8, "ipopt.max_iter": 500})
        self.opti, self.X, self.U, self.J = opti, X, U, J
        self.prev = None
        self.build_sec, self.solve_sec, self.fails = time.perf_counter() - t0, [], 0

    def solve(self, x0, u_prev, r, X_init, U_init):
        """One solve from an initial guess. Returns (X [2, H+1], U [2, H], objective, ok)."""
        opti = self.opti
        opti.set_value(self.x0, x0)
        opti.set_value(self.up, u_prev)
        opti.set_value(self.r, r)
        opti.set_initial(self.X, X_init)
        opti.set_initial(self.U, U_init)
        t0 = time.perf_counter()
        try:
            sol = opti.solve()
            out = sol.value(self.X), sol.value(self.U), float(sol.value(self.J)), True
        except RuntimeError:                    # IPOPT stopped early: keep its last iterate
            out = opti.debug.value(self.X), opti.debug.value(self.U), float(opti.debug.value(self.J)), False
            self.fails += 1
        self.solve_sec.append(time.perf_counter() - t0)
        return out

    def __call__(self, x, u_prev, sc, k):
        """Receding-horizon MPC for a batch of one scenario (the `simulate` controller signature)."""
        H = self.H
        r = sc.exo["r"][0, k + 1:k + 1 + H].double().numpy().T
        x0, up = x[0].double().numpy(), u_prev[0].double().numpy()
        if self.prev is not None:
            Xp, Up = self.prev
            Xi, Ui = np.hstack([Xp[:, 1:], Xp[:, -1:]]), np.hstack([Up[:, 1:], Up[:, -1:]])
        else:
            Xi, Ui = np.repeat(x0[:, None], H + 1, 1), np.repeat(up[:, None], H, 1)
        Xs, Us, _, _ = self.solve(x0, up, r, Xi, Ui)
        self.prev = (Xs, Us)
        return torch.tensor(np.clip(Us[:, 0], 0, U_MAX)[None], dtype=torch.float32)


def rollout(x0, U):
    """States [2, T+1] of the discrete map under inputs U [2, T] (numpy)."""
    X = np.zeros((2, U.shape[1] + 1))
    X[:, 0] = x0
    for k in range(U.shape[1]):
        X[:, k + 1] = step_np(X[0, k], X[1, k], U[0, k], U[1, k], np)
    return X


def mpc_episode(w1: float, i: int, H: int = 80, w_smooth: float = 0.1):
    """Closed-loop MPC on held-out episode i. Returns (J, U [T, 2], median solve seconds)."""
    torch.set_num_threads(1)
    case, sc = TwoTank(), eval_set(pad=max(EPISODE, H)).index(slice(i, i + 1))
    mpc = TankNLP(H, w1, w_smooth)
    _, U, T = simulate(case, mpc, sc, EPISODE)
    W = torch.tensor([[1.0, w1, w_smooth]])
    return float(weighted(T, W)[0]), U[0].numpy(), float(np.median(mpc.solve_sec))


def optimum(w1: float, i: int, guesses: list[np.ndarray] = (), w_smooth: float = 0.1, n_random: int = 2):
    """The episode optimum of held-out episode i: one NLP over all 80 steps, solved from the
    hold and steady-state guesses, `n_random` random ones and any extra `guesses` (U [T, 2], e.g. a
    controller's closed loop). Returns (J, U [T, 2]); J is the replayed cost, as the controllers'."""
    torch.set_num_threads(1)
    sc = eval_set().index(slice(i, i + 1))
    x0, up = sc.x0[0].double().numpy(), sc.u0[0].double().numpy()
    r = sc.exo["r"][0, 1:EPISODE + 1].double().numpy().T
    p, v = steady_input(*steady_optimum(r[0, 0], r[1, 0], w1))
    starts = [np.repeat(up[:, None], EPISODE, 1), np.repeat(np.clip([[p], [v]], 0, U_MAX), EPISODE, 1)]
    rng = np.random.default_rng(i)
    starts += [np.repeat(rng.uniform(0, U_MAX, (2, EPISODE // 10)), 10, 1) for _ in range(n_random)]
    starts += [np.asarray(g, float).T for g in guesses]
    nlp = TankNLP(EPISODE, w1, w_smooth)
    best = None
    for U0 in starts:
        U0 = np.clip(U0, 0, U_MAX)
        _, Us, J, _ = nlp.solve(x0, up, r, rollout(x0, U0), U0)
        if best is None or J < best[0]:
            best = (J, np.clip(Us, 0, U_MAX).T)
    return replay_cost(best[1], w1, i, w_smooth), best[1]


def replay_cost(U, w1: float, i: int, w_smooth: float = 0.1) -> float:
    """The cost of an open-loop input sequence U [T, 2] on held-out episode i, through `simulate`."""
    case, sc = TwoTank(), eval_set().index(slice(i, i + 1))
    Ut = torch.as_tensor(np.asarray(U), dtype=torch.float32)
    _, _, T = simulate(case, lambda x, up, s, k: Ut[k][None], sc, EPISODE)
    return float(weighted(T, torch.tensor([[1.0, w1, w_smooth]]))[0])
