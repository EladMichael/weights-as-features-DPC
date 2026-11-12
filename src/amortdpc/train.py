"""Training one weight-conditioned DPC policy (or a specialist) on the two-tank.

Each iteration draws fresh scenarios and fresh weights (one per window), rolls the closed loop
through the known, differentiable plant for N steps, and descends mean_i J_i(w_i) / n(w_i):

    J_N(theta; xi, w) = sum_j w_j (1/N) sum_k l_j(x_k, u_k, ...),   u_k = pi_theta(o_k, phi(w), psi(tau_k))

with tau_k = N - k, the steps left in the window. Compared with standard DPC, the changes are
(1) w is sampled per window and is a policy input, and (2) the per-bin normalizer n(w).

The protocol, as in the paper:
  1. a dev set (fixed seed, shared by every run) for checkpoint selection;
  2. warm-up: `warmup_iters` unnormalized iterations at the knob's low end (a specialist: its weight);
     the running normalizer's bins are seeded with the warmed-up policy's dev cost;
  3. main loop: Adam, gradient clipping at 1.0, warm-up + cosine learning-rate schedule;
  4. every `eval_every` iterations, the dev score; the best checkpoint is kept;
  5. collapse rollback: if the dev score is more than `rewind_tol` above its best (or not finite) at two evaluations
     in a row, restore the best checkpoint (policy, Adam state, normalizer) and continue.

The paper trains each two-tank run for 36,000 iterations at batch 1024 (about 72 CPU-minutes).
The tutorial's `TUTORIAL` budget is far smaller: the same method, a much shorter run.
"""

from __future__ import annotations

import copy
import math
import time

import torch

from .policy import ConcatPolicy, actor
from .twotank import TwoTank, simulate, weighted
from .weights import WeightSpace

# The paper's two-tank configuration (configs/twotank/cond.yaml in the full code).
PAPER = dict(N=80, batch=1024, iters=36_000, lr=5e-3, lr_final=5e-5, warmup_frac=0.05, warmup_iters=200,
             dev_batch=2048, eval_every=100, grad_clip=1.0, rewind_tol=1.0, rewind_after=2, rewind_max=10,
             lo=0.01, hi=100.0, smooth=0.1, norm="running", pad=1, hidden=64, layers=3)


def make(cfg: dict):
    case = TwoTank(infeasible_frac=0.5)
    ws = WeightSpace(case.terms, knob="track1", lo=cfg["lo"], hi=cfg["hi"], fixed={"smooth": cfg["smooth"]},
                     norm=cfg["norm"], pad=cfg["pad"])
    policy = ConcatPolicy(case.nf, 1, case.nu, case.u_lo, case.u_hi, N=cfg["N"],
                          hidden=cfg["hidden"], layers=cfg["layers"])
    return case, ws, policy


def objective(case, ws, policy, sc, w, N, normalize=True, update=False):
    """Per-sample J(w) / n(w) over an N-step closed-loop window, tau counting down N, N-1, ..., 1."""
    _, _, T = simulate(case, actor(case, policy, ws.encode(w), tau=lambda k: N - k), sc, N)
    J = weighted(T, ws.full(w), case.terms)
    if update and ws.norm == "running":
        ws.update_running(w, J)
    return (J / ws.n(w) if normalize else J), T


def cosine(opt, cfg):
    """Linear warm-up to lr over warmup_frac of the iterations, then cosine decay to lr_final."""
    n, lr, lo = cfg["iters"], cfg["lr"], cfg["lr_final"]
    w = max(int(cfg["warmup_frac"] * n), 1)
    f = lambda i: (i + 1) / w if i < w else lo / lr + (1 - lo / lr) * 0.5 * (1 + math.cos(math.pi * (i - w) / max(n - w, 1)))
    return torch.optim.lr_scheduler.LambdaLR(opt, f)


def train(cfg: dict, seed: int = 0, spec: float | None = None, verbose: bool = False):
    """Train one policy. spec: pin the knob to this weight (a specialist); None: the generalist.
    Returns (policy at its best dev checkpoint, its WeightSpace, log rows)."""
    cfg = {**PAPER, **cfg}
    if spec is not None:
        cfg["pad"] = 0                                     # a specialist has nothing to pad
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    dev_gen = torch.Generator().manual_seed(10_000)        # the dev set is the same for every run
    case, ws, policy = make(cfg)
    N, B = cfg["N"], cfg["batch"]
    dev_sc = case.sample(cfg["dev_batch"], N, dev_gen)
    dev_w = ws.sample(cfg["dev_batch"], dev_gen, spec, claimed=True)
    opt = torch.optim.Adam(policy.parameters(), lr=cfg["lr"])
    sched = cosine(opt, cfg)
    t0 = time.time()

    # 2. warm-up at one weight, unnormalized; then seed the normalizer's bins
    w0 = torch.full((B,), spec if spec is not None else cfg["lo"])
    for _ in range(cfg["warmup_iters"]):
        loss, _ = objective(case, ws, policy, case.sample(B, N, gen), w0, N, normalize=False)
        opt.zero_grad()
        loss.mean().backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg["grad_clip"])
        opt.step()
    with torch.no_grad():
        J0, _ = objective(case, ws, policy, dev_sc, w0[:1].expand(len(dev_sc)), N, normalize=False)
    ws.running.fill_(J0.mean().item())

    # 3-5. the main loop
    log, best, bad, rewinds = [], math.inf, 0, []
    for it in range(1, cfg["iters"] + 1):
        sc, w = case.sample(B, N, gen), ws.sample(B, gen, spec)
        loss, _ = objective(case, ws, policy, sc, w, N, update=True)
        opt.zero_grad()
        loss.mean().backward()
        gn = torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg["grad_clip"])
        opt.step()
        sched.step()
        if it % cfg["eval_every"] == 0 or it == cfg["iters"]:
            with torch.no_grad():
                dJ, _ = objective(case, ws, policy, dev_sc, dev_w, N, normalize=False)
                dev = ws.dev_score(dJ.clamp(min=1e-12), dev_w)
            if dev < best:
                best, best_state = dev, copy.deepcopy(policy.state_dict())
                best_opt, best_running = copy.deepcopy(opt.state_dict()), ws.running.clone()
            bad = bad + 1 if dev > best + cfg["rewind_tol"] or not math.isfinite(dev) else 0      # NaN counts as collapsed
            if bad >= cfg["rewind_after"] and len(rewinds) < cfg["rewind_max"]:     # collapsed: roll back
                lrs = [g["lr"] for g in opt.param_groups]                          # the schedule keeps its place
                policy.load_state_dict(best_state)
                opt.load_state_dict(best_opt)
                for g, lr in zip(opt.param_groups, lrs):
                    g["lr"] = lr
                ws.running.copy_(best_running)
                rewinds.append(it)
                bad = 0
            log.append(dict(it=it, train=loss.mean().item(), dev=dev, best=best, grad=gn.item(),
                            lr=opt.param_groups[0]["lr"], sec=time.time() - t0, rewinds=len(rewinds)))
            if verbose:
                print(f"it {it:5d}  dev {dev:.4f}  best {best:.4f}  {time.time() - t0:.0f}s", flush=True)
    policy.load_state_dict(best_state)
    policy.eval()
    return policy, ws, log


def _job(args):
    """One training run in a worker process (for train_many)."""
    cfg, seed, spec = args
    torch.set_num_threads(1)
    policy, ws, log = train(cfg, seed, spec)
    return spec, policy.state_dict(), log


def train_many(cfg: dict, specs: list, seed: int = 0, procs: int | None = None):
    """Train several runs at once, one process each (spec None = the generalist, a number = a specialist).
    Yields (spec, policy, ws, log) as each run finishes."""
    import multiprocessing as mp
    procs = procs or min(len(specs), mp.cpu_count())
    with mp.get_context("spawn").Pool(procs) as pool:
        for spec, state, log in pool.imap_unordered(_job, [(cfg, seed, s) for s in specs]):
            c = {**PAPER, **cfg, **({"pad": 0} if spec is not None else {})}
            _, ws, policy = make(c)
            policy.load_state_dict(state)
            policy.eval()
            yield spec, policy, ws, log
