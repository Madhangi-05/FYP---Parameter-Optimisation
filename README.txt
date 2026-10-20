================================================================================
 Noise-Aware Geometry-Guided GNN Initialization for VQE + QNGD
 Batch 51 -- Source Code Submission
================================================================================

Team      : R.K. Barath (3122235001029), Madhangi Karimanal (3122235001073),
            Nithyasree K (3122235001092)
Guide     : Dr. Sneka Nandhini, Assistant Professor, Dept. of Computer Science
Institute : Sri Sivasubramaniya Nadar College of Engineering


--------------------------------------------------------------------------------
 a. URL / SOURCE FOR DATASET
--------------------------------------------------------------------------------

Primary dataset: VQEzy
  Repository : https://github.com/chizhang24/VQEzy
  License / usage : public GitHub repository, used for the Hamiltonian
  families and problem instances; we do NOT use VQEzy's own stored
  "optimal" parameters as ground truth -- see note below.

  Files actually used (under VQEzy/qmanybody/, plain HDF5, read directly
  with h5py -- no need for VQEzy's own environment):
    xyz_4_qubit.h5    (Heisenberg-XYZ, 4 qubits)
    fh_4_qubit.h5     (Fermi-Hubbard, 4 qubits)
    fh_6_qubit.h5     (Fermi-Hubbard, 6 qubits)
    fh_8_qubit.h5     (Fermi-Hubbard, 8 qubits)
    ti_8_qubit.h5     (Transverse-field Ising, 8 qubits)
    xyz_12_qubit.h5   (Heisenberg-XYZ, 12 qubits -- held out entirely for
                        the "unseen qubit count" generalization test)

Ground-truth energies: we do NOT trust any dataset's self-reported "optimal"
energy (several were found to be far from the true minimum). Instead, every
ground-state energy used for evaluation is computed independently by exact
diagonalization (feasible up to 12 qubits -- see
src/quantum/hamiltonians.py, EXACT_DIAG_MAX_QUBITS).

Reference / comparison baseline (optional, not required to run our
pipeline -- only needed if you want to inspect the prior-art baseline we
compared against in the report):
  Qracle repository : https://github.com/chizhang24/Qracle


--------------------------------------------------------------------------------
 b. SOFTWARE AND HARDWARE REQUIREMENTS
--------------------------------------------------------------------------------

Operating system
  Windows 10/11, Linux, or macOS. Developed on Windows 11; GNN training was
  run on a Linux lab machine with an NVIDIA GPU. Both work with the same
  code -- only the CUDA vs CPU torch wheel differs (see Setup, step 2).

Python
  Python 3.10 or later (developed and tested on Python 3.13).

Python packages (pinned versions in requirements.txt; install via pip)
  torch             >= 2.1      (PyTorch -- GNN + the custom simulator)
  torch_geometric   >= 2.4      (GATConv graph-attention encoders)
  pennylane         >= 0.35     (reference quantum circuit simulation)
  pennylane-lightning >= 0.35   (lightning.qubit fast statevector device)
  h5py              >= 3.9      (reads the VQEzy .h5 dataset files)
  networkx          >= 3.1      (graph construction utilities)
  numpy             >= 1.24
  pandas            >= 2.0      (result tables / CSV output)
  matplotlib        >= 3.7      (graph/result visualization)

Hardware
  CPU only is sufficient for: dataset/graph construction, the VQE/QNGD
  baselines, and small-scale evaluation -- nothing here strictly requires
  a GPU.
  GPU (CUDA-capable, e.g. NVIDIA, tested with CUDA 12.1) is recommended,
  not required, for training the GNN initializers -- training falls back
  to CPU automatically if no GPU is available, just slower.
  RAM: 8 GB or more recommended.
  Disk: ~300 MB free -- about 250 MB for the cloned VQEzy (+ optional
  Qracle) reference repositories, plus a few MB for the cached dataset and
  checkpoints (both already included in this submission).


--------------------------------------------------------------------------------
 c. DETAILED INSTRUCTIONS TO EXECUTE THE SOURCE CODE
--------------------------------------------------------------------------------

All commands below are run from the repository root (the folder this file
is in) and assume a Windows path style (.venv/Scripts/...); on Linux/macOS
use .venv/bin/... instead. Every script can also be run from anywhere once
the virtual environment is active, as long as your current directory is the
repository root.

--------------------------------------------------------------------------------
STEP 1 -- Create the environment and install dependencies
--------------------------------------------------------------------------------

    python -m venv .venv

    REM CPU-only machine:
    .venv\Scripts\pip install torch --index-url https://download.pytorch.org/whl/cpu
    .venv\Scripts\pip install -r requirements.txt

    REM GPU machine (match the CUDA version you have installed, e.g. cu121):
    .venv\Scripts\pip install torch --index-url https://download.pytorch.org/whl/cu121
    .venv\Scripts\pip install -r requirements.txt

--------------------------------------------------------------------------------
STEP 2 -- Download the dataset
--------------------------------------------------------------------------------

    mkdir external
    cd external
    git clone https://github.com/chizhang24/VQEzy.git
    cd ..

This creates external/VQEzy/qmanybody/*.h5, which Step 3 reads directly.
(This step can be skipped if you only want to inspect the already-cached
graphs/checkpoints included in this submission -- see Step 6.)

--------------------------------------------------------------------------------
STEP 3 -- Build the cached dataset (graphs + train/val/test split)
--------------------------------------------------------------------------------

Converts each VQEzy instance into its three graphs (Hamiltonian G_H, ansatz
G_A, hardware/noise G_D), computes its exact ground energy, and writes a
fixed, reusable train/val/test split to data_cache/. This step is CPU-only
and needs to be run once.

    .venv\Scripts\python run_build_dataset.py --train-per-file 200 --val-per-file 30 --test-per-file 30 --unseen-size-count 8

IMPORTANT: always invoke run_build_dataset.py directly like this, never as
"python -m src.data.build_dataset" -- the latter changes how Python records
the cached class's module path and breaks loading the cache from any other
script afterward (a standard Python pickling gotcha). All other scripts in
this project are safe to invoke either way.

Then build the noise-varied copy of the same split (needed for the
noise-aware training variants, GNN-GD-B / GNN-A / GNN-B):

    .venv\Scripts\python -m src.data.build_noisy_dataset

This writes data_cache_noisy/ -- the same instances and split as
data_cache/, but each instance gets a freshly sampled hardware noise level
(low/medium/high) and topology (linear/ring/grid) instead of one fixed
setting.

NOTE: a pre-built data_cache/ and data_cache_noisy/ are already included in
this submission, so Steps 2-3 are only needed if you want to rebuild the
dataset from scratch (e.g. with different split sizes).

--------------------------------------------------------------------------------
STEP 4 -- Reproduce the baselines (S1: original VQE, S2: VQE+QNGD)
--------------------------------------------------------------------------------

S1 -- the dataset's own plain VQE procedure (Adam optimizer, no QNGD):

    .venv\Scripts\python -m src.quantum.run_vanilla_vqe --split test

S2 -- VQE+QNGD from a random start (the real baseline every GNN variant is
measured against):

    .venv\Scripts\python -m src.quantum.run_qngd --init random --data-dir data_cache --split test
    .venv\Scripts\python -m src.quantum.run_qngd --init random --data-dir data_cache_noisy --split test

The first line reproduces the noiseless baseline numbers, the second the
noisy baseline numbers. Both scripts print circuit-execution counts (from
PennyLane's qml.Tracker, true device-level counts) and target-reached
success rates, and write a results/*.csv file.

--------------------------------------------------------------------------------
STEP 5 -- Train the GNN initializer (5 variants)
--------------------------------------------------------------------------------

Each variant isolates two choices: the inner optimizer unrolled during
training (plain gradient descent, "GD", vs Quantum Natural Gradient Descent,
"QNGD"), and whether the hardware/noise graph is used and trained with
varied noise. Training is label-free throughout -- no pre-computed "optimal"
parameters are used as targets.

  GNN-GD   (GD-unrolled, fixed noiseless hardware graph):
    .venv\Scripts\python run_train_gpu.py --unroll-steps 3 --ckpt-name gnn_gd.pt

  GNN-GD-A (GD-unrolled, hardware/noise graph removed -- ablation):
    .venv\Scripts\python -m src.quantum.train_qngd --variant clean --unroll-opt gd --stepsize 0.1 --ckpt-name gnn_gd_a.pt

  GNN-GD-B (GD-unrolled, noise-aware -- trained on data_cache_noisy):
    .venv\Scripts\python -m src.quantum.train_qngd --variant noisy --unroll-opt gd --stepsize 0.1 --ckpt-name gnn_gd_b.pt

  GNN-A    (QNGD-unrolled, hardware/noise graph removed -- ablation):
    .venv\Scripts\python -m src.quantum.train_qngd --variant clean --ckpt-name qngd_clean.pt

  GNN-B    (QNGD-unrolled, noise-aware -- trained on data_cache_noisy):
    .venv\Scripts\python -m src.quantum.train_qngd --variant noisy --ckpt-name qngd_noisy.pt

Each command saves its checkpoint to checkpoints/<ckpt-name> and its
training history to results/train_history_<name>.json. On a GPU this takes
roughly 15-30 minutes per variant at the default 30 epochs; on CPU
considerably longer -- reduce --epochs for a quick smoke test.

NOTE: all 5 trained checkpoints are already included in checkpoints/, so
Step 5 is only needed if you want to retrain from scratch.

Before trusting any training run that unrolls optimizer steps, you can
sanity-check the custom simulator itself against PennyLane:

    .venv\Scripts\python -m src.quantum.validate_torch_sim

--------------------------------------------------------------------------------
STEP 6 -- Evaluate a trained variant against the random-start baseline
--------------------------------------------------------------------------------

General form (works for any of the 5 checkpoints above; add --no-hardware
for GNN-GD-A, GNN-A, which were trained without the hardware graph):

    .venv\Scripts\python -m src.quantum.run_qngd --init gnn --ckpt checkpoints\gnn_gd.pt --data-dir data_cache --split test
    .venv\Scripts\python -m src.quantum.run_qngd --init gnn --ckpt checkpoints\qngd_clean.pt --no-hardware --data-dir data_cache --split test

Run the same command with --data-dir data_cache_noisy to get the under-noise
numbers for that checkpoint. Each run prints and saves (results/*.csv) the
circuit-execution count vs. the S2 random-start baseline and the
target-reached success rate -- this is how every row of the results tables
in review2/report.pdf was produced.

For comparing several checkpoints in one call (used for quick
pilot/ablation checks on the validation split):

    .venv\Scripts\python -m src.quantum.eval_sim_qngd --models gnn-gd:checkpoints\gnn_gd.pt:hw gnn-a:checkpoints\qngd_clean.pt:nohw gnn-b:checkpoints\qngd_noisy.pt:hw --split val

    .venv\Scripts\python -m src.quantum.compare_init_cached --ckpt checkpoints\gnn_gd.pt --data-dir data_cache --split test

See the top-of-file docstring in each of these two scripts for their full
option list.

--------------------------------------------------------------------------------
STEP 7 -- Visualize the graphs (optional)
--------------------------------------------------------------------------------

    .venv\Scripts\python -m src.graphs.visualize_batch

Renders the three constructed graphs (G_H, G_A, G_D) for a sample of
instances to results/*.png, for visual inspection.

--------------------------------------------------------------------------------
Project layout (see README.md in this same folder for the full annotated
module-by-module breakdown)
--------------------------------------------------------------------------------

  run_build_dataset.py, run_train_gpu.py   entry-point launcher scripts
  requirements.txt                          pinned Python dependencies
  src/data/        dataset loading + graph-cache construction
  src/graphs/      Hamiltonian / ansatz / hardware-noise graph builders
  src/models/      GNN encoders, cross-attention fusion, parameter decoder
  src/quantum/     VQE/QNGD circuits, noise model, custom simulator,
                   training loops, evaluation/comparison scripts
  checkpoints/     the 5 trained GNN initializer checkpoints
  data_cache/, data_cache_noisy/    pre-built cached graph datasets
  results/, logs/  outputs and logs from our own experiment runs
  review2/         the full written report, presentation, and poster
================================================================================
