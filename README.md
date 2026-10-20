# Noise-Aware Geometry-Guided GNN Initialization for VQE + QNGD

FYP implementation (Batch 51). See `README.txt` for the submission-format
setup/run instructions, or `review2/report.pdf` for the full write-up. The
original planning documents are in the repo root
(`reviereport (2).pdf`, `VQE_QNGD_Novelty_Methodology_Proposal.docx`,
`Plan_A_Plan_B_VQE_Dataset_Strategy.docx`, `work_implementation.docx`).

## Setup

```bash
python -m venv .venv
.venv/Scripts/pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/Scripts/pip install -r requirements.txt
```

## Reference repos (not tracked here — re-clone as needed)

```bash
mkdir external && cd external
git clone https://github.com/chizhang24/VQEzy.git   # dataset: Hamiltonians + optimal trajectories
git clone https://github.com/chizhang24/Qracle.git  # prior-art GNN VQE initializer (comparison baseline)
```

`VQEzy/qmanybody/*.h5` (xyz/fh/ti families) and `VQEzy/qchem/*.h5` (h2/hehp/nh3)
are plain data, loadable with h5py directly — no need for VQEzy's own `uv`/
TorchQuantum environment. Only the `qasmbench/random_vqe` family needs
TorchQuantum.

## Layout

```
run_build_dataset.py          # Stage 1 entry point (see "Training on a GPU machine")
run_train_gpu.py               # Stage 3 entry point
src/
├── data/
│   ├── vqezy_loader.py        # reads VQEzy .h5 instances (qmanybody family)
│   ├── build_dataset.py       # Stage 1: extract+cache (G_H,G_A,G_D) + fixed train/val/test split
│   └── dump_h5_sample.py      # human-readable dump of a .h5 file's raw contents
├── quantum/
│   ├── hamiltonians.py        # rebuilds PennyLane Hamiltonians from stored couplings; exact ground energy
│   ├── ansatz.py               # CZRXRY hardware-efficient ansatz (matches VQEzy's own)
│   ├── vqe_baseline.py         # Phase 1: plain VQE+QNGD baseline, real circuit-eval counts
│   ├── run_baseline_batch.py   # Phase 1 baseline across many instances
│   ├── vqe_torch.py            # differentiable (torch-interface) VQE energy, for training the GNN
│   ├── train_phase2.py         # small-scale/CPU training + held-out eval (direct or unrolled-GD loss)
│   ├── train_gpu.py             # Stage 3: GPU-ready training off the cached dataset
│   └── compare_init_strategies.py  # THE real test: GNN-init vs random-init circuit evals to threshold
├── graphs/                     # Hamiltonian / ansatz / hardware+noise graph builders + visualization
└── models/                     # GNN encoders, cross-attention fusion, parameter decoder (f_theta -> theta_0)
```

## QNGD, noise modeling, and the 5 trained variants

Later work (building on Stage 3 above) adds Quantum Natural Gradient Descent
as the inner optimizer, a hardware-driven noise model, and a custom
differentiable simulator so training can backprop through many unrolled,
noisy QNGD steps at GPU-batch scale -- PennyLane's own noisy simulator
(`default.mixed`) is validated against but not fast enough to train with
directly.

```
src/data/build_noisy_dataset.py    # derives data_cache_noisy/ from data_cache/: same split,
                                    # each instance gets a freshly sampled noise level + topology
src/quantum/noise_model.py          # turns G_D's own numbers into depolarizing/amplitude-damping
                                    # channel parameters (topology-aware CZ routing cost included)
src/quantum/torch_sim.py            # batched, differentiable density-matrix simulator + QNGD
                                    # update, exactly matching qml.QNGOptimizer(approx="block-diag")
src/quantum/validate_torch_sim.py   # checks torch_sim.py against PennyLane: energies, QNGD steps,
                                    # and the unrolled meta-gradient vs finite differences
src/quantum/train_qngd.py           # trains GNN-A / GNN-B: unrolled QNGD-step loss, clean or noisy
src/quantum/run_qngd.py             # runs full VQE+QNGD (random or GNN init) on a cached split,
                                    # counting circuit executions to a target energy
src/quantum/eval_sim_qngd.py        # evaluates one or more checkpoints side by side (used for the
                                    # 5-variant comparison table)
src/quantum/eval_noisy_qngd.py      # clean-vs-noisy evaluation for a single checkpoint
src/quantum/compare_init_cached.py  # GD-unrolled variants (GNN-GD/-A/-B): GNN-init vs random-init,
                                    # both GD and QNGD inner loops, off the cached dataset
src/quantum/run_vanilla_vqe.py      # reproduces the dataset's own plain-VQE/Adam procedure, for
                                    # the S1 (original VQE) vs S2 (VQE+QNGD) baseline comparison
```

Exact ground energies (the evaluation target everywhere, not any dataset's
self-reported optimum) come from full diagonalization, feasible up to
`EXACT_DIAG_MAX_QUBITS = 12` qubits (`src/quantum/hamiltonians.py`).

The checkpoints in `checkpoints/` correspond to:

| Checkpoint | Variant | Trained via | Noise during training |
| --- | --- | --- | --- |
| `gnn_gd.pt` | GNN-GD | `compare_init_cached.py` (GD-unrolled) | none (fixed medium/ring graph) |
| `gnn_gd_a.pt` | GNN-GD-A | `compare_init_cached.py` (GD-unrolled) | none (hardware graph removed) |
| `gnn_gd_b.pt` | GNN-GD-B | `compare_init_cached.py` (GD-unrolled) | yes, varied per instance |
| `qngd_clean.pt` | GNN-A | `train_qngd.py --variant clean` | none (hardware graph removed) |
| `qngd_noisy.pt` | GNN-B | `train_qngd.py --variant noisy` | yes, varied per instance |
| `gpu_initializer*.pt`, `pilot_clean_K15_traj.pt` | early/pilot runs | `train_gpu.py` | see Stage 3 above |

## Phase 1 baseline

```bash
.venv/Scripts/python -m src.quantum.vqe_baseline --h5 external/VQEzy/qmanybody/xyz_4_qubit.h5 --sample sample_0
```

Circuit-eval counts come from `qml.Tracker` (true device-level executions),
not a call counter on the QNode itself — parameter-shift gradients and QNGD's
metric-tensor estimation both trigger many device executions per optimizer
step without re-invoking the decorated circuit function, so a naive counter
undercounts by ~30x.

## Training the GNN initializer on a GPU machine

Two stages, run in order. `data_cache/`, `checkpoints/`, and `results/*.json`
ARE tracked in git (all small -- a checkpoint is ~110KB, the whole cache a
few MB) -- this is the intended round trip: the laptop generates
`data_cache/` and pushes it, the GPU machine pulls it, trains, and pushes
`checkpoints/` + `results/` back, and the laptop continues evaluation/write-up
without needing the GPU machine again. `external/` (the ~250MB cloned
VQEzy/Qracle repos) is the only thing gitignored -- re-clone it locally per
the commands above if you need to rebuild `data_cache/` from scratch.

```bash
# 0. Setup: same as above, but install CUDA-enabled torch instead of the CPU wheel, e.g.
.venv/Scripts/pip install torch --index-url https://download.pytorch.org/whl/cu121  # match your CUDA version
.venv/Scripts/pip install pennylane pennylane-lightning torch_geometric h5py networkx numpy scipy matplotlib pandas tqdm
mkdir external && cd external && git clone https://github.com/chizhang24/VQEzy.git && cd ..

# 1. Extract + cache (G_H, G_A, G_D) for a fixed pool of instances, and write a
#    persisted train/val/test split manifest. CPU-bound, cheap, run once.
.venv/Scripts/python run_build_dataset.py --train-per-file 200 --val-per-file 30 --test-per-file 30 --unseen-size-count 40

# 2. Train. GNN runs on GPU; quantum circuit simulation (PennyLane lightning.qubit)
#    stays CPU-bound regardless -- see train_gpu.py's module docstring for why,
#    and for how to opt into lightning.gpu if that's installed.
.venv/Scripts/python run_train_gpu.py --epochs 30 --hidden 64 --gnn-layers 3
```

IMPORTANT: always invoke `run_build_dataset.py` directly (not
`python -m src.data.build_dataset`) — the latter makes Python treat that file
as `__main__`, which breaks pickling its `CachedInstance` class for later
loading by a different script (a classic Python gotcha: a class's pickled
module path depends on how the *defining* file was executed, not how it's
later imported). `run_train_gpu.py` doesn't have this restriction since it
doesn't define any pickled classes itself, but using the launcher for both
keeps usage consistent.

`--unroll-steps K` (both `train_phase2.py` and `train_gpu.py`) trains against
the energy after `K` unrolled gradient-descent steps instead of the energy at
theta_0 directly — see git history for why: minimizing E(theta_0) directly was
empirically shown (`compare_init_strategies.py`) to produce starting points
that look good on paper but sit in flat/plateau regions, so a real optimizer
often does *worse* starting from there than from a random point. This fix is
technically validated (gradients do flow through the unrolled steps) but not
yet validated at real scale — that's what running it here is for.
