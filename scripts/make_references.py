"""Regenerate data/twotank_references.npz from scratch: the 80-step MPC and the episode optimum on the
40 held-out episodes at 25 tank-1 weights in [0.01, 100].

    python scripts/make_references.py [--procs 8] [--out data/twotank_references_new.npz]

About 1,000 MPC episodes at roughly 2.5 s each plus 1,000 optimum solves, so about an hour of
CPU-time; divide by --procs.

The shipped file's optimum was also seeded ("polished") with the paper's DPC closed loops as
initial guesses and kept wherever that lowered the cost; this script seeds it with the MPC closed loop
and its own guesses only. Expect the same numbers to within IPOPT's local optima (a few episodes may
come out slightly higher). Pass DPC closed loops through `optimum(..., guesses=[...])` to polish.
"""

import argparse
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from amortdpc.mpc import mpc_episode, optimum  # noqa: E402
from amortdpc.twotank import EPISODE, eval_set  # noqa: E402

W_SMOOTH = 0.1


def job(a):
    w1, i = a
    Jm, Um, _ = mpc_episode(w1, i, H=80, w_smooth=W_SMOOTH)
    Jo, Uo = optimum(w1, i, guesses=[Um], w_smooth=W_SMOOTH, n_random=5)
    return Um, Uo


def terms(U, w1, i):
    """Mean of each cost term for one input sequence, replayed through the simulator."""
    import torch
    from amortdpc.twotank import TwoTank, simulate
    sc = eval_set().index(slice(i, i + 1))
    Ut = torch.as_tensor(U, dtype=torch.float32)
    _, _, T = simulate(TwoTank(), lambda x, up, s, k: Ut[k][None], sc, EPISODE)
    return {n: float(v.mean()) for n, v in T.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--out", default="data/twotank_references_new.npz")
    a = ap.parse_args()
    w = np.logspace(-2, 2, 25)
    ns = len(eval_set())
    jobs = [(wi, i) for wi in w for i in range(ns)]
    t0 = time.time()
    with Pool(a.procs) as pool:
        res = pool.map(job, jobs, chunksize=4)
    out = dict(w1=w, w_smooth=np.float64(W_SMOOTH), steps=np.int64(EPISODE))
    for p, k in (("mpc", 0), ("opt", 1)):
        U = np.stack([r[k] for r in res]).reshape(len(w), ns, EPISODE, 2).astype(np.float32)
        out[f"U_{p}"] = U
        tt = [[terms(U[iw, i], w[iw], i) for i in range(ns)] for iw in range(len(w))]
        for n in ("track2", "track1", "smooth"):
            out[f"{p}_{n}"] = np.array([[t[n] for t in row] for row in tt])
    np.savez_compressed(a.out, **out)
    print(f"wrote {a.out} in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
