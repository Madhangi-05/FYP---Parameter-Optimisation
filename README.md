# Noise-Aware Geometry-Guided GNN Initialization for VQE + QNGD

FYP implementation. See the project documents in the repo root for the full
research plan (`reviereport (2).pdf`, `VQE_QNGD_Novelty_Methodology_Proposal.docx`,
`Plan_A_Plan_B_VQE_Dataset_Strategy.docx`, `work_implementation.docx`).

## Setup

```bash
python -m venv .venv
.venv/Scripts/pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/Scripts/pip install pennylane pennylane-lightning torch_geometric h5py networkx numpy scipy matplotlib pandas tqdm
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
src/
├── data/vqezy_loader.py      # reads VQEzy .h5 instances (qmanybody family)
├── quantum/
│   ├── hamiltonians.py       # rebuilds PennyLane Hamiltonians from stored couplings
│   ├── ansatz.py             # CZRXRY hardware-efficient ansatz (matches VQEzy's own)
│   └── vqe_baseline.py       # Phase 1: plain VQE+QNGD baseline, real circuit-eval counts
├── graphs/                   # Phase 2: Hamiltonian / ansatz / hardware+noise graph builders
└── models/                   # Phase 2+: GNN encoders, fusion, decoder, geometry head
```

## Phase 1 baseline

```bash
.venv/Scripts/python -m src.quantum.vqe_baseline --h5 external/VQEzy/qmanybody/xyz_4_qubit.h5 --sample sample_0
```

Circuit-eval counts come from `qml.Tracker` (true device-level executions),
not a call counter on the QNode itself — parameter-shift gradients and QNGD's
metric-tensor estimation both trigger many device executions per optimizer
step without re-invoking the decorated circuit function, so a naive counter
undercounts by ~30x.
