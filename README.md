# One policy, the whole knob: weight-conditioned DPC (two-tank tutorial)

Elad Michael, Cary Faulkner, Himanshu Sharma, Yan Chen, Draguna Vrabie<br>
Pacific Northwest National Laboratory · contact: elad.michael@pnnl.gov

Companion code for **Multi-Objective DPC: Predictive Control Across the Pareto Front** (submitted to L4DC 2027).

Differentiable predictive control (DPC) normally fixes the objective's weights as scalars, so every
trade-off needs its own trained policy. Here the weights become a **policy input**. They are sampled per
training window, applied per sample in the loss, and the loss is normalized per weight bin. One
training run then gives a controller for every weight on a four-decade knob, within a few hundredths of a
percent of the MPC it targets in the paper's two-tank case.

This repository is a **small, self-contained tutorial**, not the paper's full codebase. It covers the
two-tank case only. Its notebook trains the paper's controllers from scratch and reproduces the two-tank
regret-vs-optimum figure at reduced scale.

![regret against the optimum](figures/regret_opt.png)

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate      # Python 3.10+
pip install -r requirements.txt
jupyter notebook two_tank_tutorial.ipynb
```

CPU only. The notebook trains four policies (a generalist and three specialists) in parallel, one process each.
It runs in about 3–5 minutes on a laptop with 4 or more cores. We measured 3.1 min on a 12-core Apple M4 Pro
with other jobs running, and 4.9 min under heavy load.

What to expect: the generalist's regret against the 80-step MPC it targets, as a mean / worst value over the knob's 25 weights
(seed 0):

| budget | per run | generalist vs MPC, mean / worst |
|:--|:--|:--|
| `RUN = "fast"` (default) | 512 × 2,000 iterations | 0.32% / 0.73% |
| `RUN = "longer"` | 1024 × 6,000 iterations | 0.14% / 0.26% |
| the paper (5 seeds) | 1024 × 36,000 iterations | 0.016% / 0.040% |

## Reduced scale: what is and is not the paper's

| | paper | this notebook (`RUN = "fast"`) |
|:--|:--|:--|
| plant, objective, test episodes | two-tank, 40 held-out 80-step episodes, w₁ ∈ [0.01, 100] | same |
| policy | concat MLP 3×64, φ(w) and time-to-go inputs, bounded head | same |
| training | per-window weights, per-bin running normalizer, padded box, warm-up + cosine LR, rollback, dev checkpointing | same |
| budget per run | batch 1024 × 36,000 iterations (about 72 CPU-min) | batch 512 × 2,000 iterations |
| seeds | 5 | 1 |
| MPC and optimum | CasADi/IPOPT | same code; the paper's values ship in `data/`, a few are re-solved live |

The training code is a trimmed copy of the paper's. On a short run with the same seed it produces dev
scores identical to the paper's code. The **curves therefore have the paper's shape, but they sit above its
numbers.** The policies' gap above the MPC is training error that a longer run closes. Set
`RUN = "longer"` or `RUN = "paper"` in the notebook to see it shrink. The full-scale results are the
figures in [`figures/`](figures/).

## Layout

```
two_tank_tutorial.ipynb      the walkthrough: problem → method → train → references → regret plot
src/amortdpc/
  twotank.py                 the plant, its cost terms, the held-out episodes, the closed-loop simulator
  policy.py                  the concat policy and its time-to-go input
  weights.py                 weight sampling, the encoding φ(w), the per-bin loss normalizer
  train.py                   the training loop (the paper's protocol); PAPER = the paper's configuration
  evaluate.py                closed-loop knob sweeps and regret
  mpc.py                     nonlinear MPC and the episode optimum (CasADi + IPOPT)
scripts/make_references.py   regenerate data/twotank_references.npz (about 1 CPU-hour)
data/twotank_references.npz  the paper's MPC (H = 80) and optimum on the 40 episodes × 25 weights
figures/                     every figure of the paper (PNG, and PDF in figures/pdf/)
```

## Adapting it to your system

Everything plant-specific is in `src/amortdpc/twotank.py`: a differentiable `step`, the cost as named
unweighted terms (`stage`), a scenario sampler and an observation (`features`). Replace these, choose which
terms get a sampled weight and over what range (`WeightSpace`), and the training loop runs unchanged.
Section 8 of the notebook lists the steps.

## Figures

| file | content |
|:--|:--|
| `pipeline` | the training pipeline |
| `frontiers` | the two traded terms along the knob: generalist, specialists, MPC |
| `regret_mpc`, `regret_opt` | 1-D regret against the same-horizon MPC and against the optimum (LQ, two-tank, building) |
| `regret2d_mpc`, `regret2d_opt` | the same with a second knob (the move penalty) |
| `normalizers` | the per-bin normalizer against the alternatives, including none |
| `convergence`, `convergence_median`, `time_to_converge` | training cost against iterations and CPU time |

## Citation

```bibtex
@inproceedings{michael2027multiobjective,
  title     = {Multi-Objective {DPC}: Predictive Control Across the {Pareto} Front},
  author    = {Michael, Elad and Faulkner, Cary and Sharma, Himanshu and Chen, Yan and Vrabie, Draguna},
  booktitle = {Learning for Dynamics and Control (L4DC)},
  year      = {2027},
  note      = {Under review}
}
```

## License

BSD 2-Clause, Copyright Battelle Memorial Institute 2026, with the U.S. Department of Energy / Battelle
disclaimer. See `LICENSE`.
