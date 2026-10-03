"""Stage 3: GPU-ready training loop over the cached dataset from
build_dataset.py (run that first).

Device split, and why: the GNN (encoders/fusion/decoder) moves to GPU and
benefits from it -- that's most of the trainable-parameter compute. The
quantum circuit itself (computing the actual VQE energy) stays on CPU by
default: PennyLane's lightning.qubit is a CPU statevector simulator, so
GPU-vs-CPU for the rest of the model doesn't make quantum simulation faster.
--qdevice lightning.gpu is supported IF that plugin (needs NVIDIA cuQuantum)
is installed on the target machine; otherwise leave the default.

Same two training objectives as train_phase2.py, ported to work off the
cached dataset:
  --unroll-steps 0 (default): loss = E(theta_0) directly.
  --unroll-steps K > 0: loss = E(theta_K) after K unrolled GD steps -- the
    attempted fix for the "low start energy but flat gradient" problem found
    empirically (see compare_init_strategies.py / git history). Still
    unvalidated at real scale as of this script's creation -- that's
    precisely what running this on a GPU machine is for.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from pennylane import numpy as pnp

from src.data.build_dataset import DATASET_DIR
from src.models.initializer import VQEInitializer
from src.quantum.hamiltonians import build_hamiltonian_from_params
from src.quantum.run_baseline_batch import REPO_ROOT
from src.quantum.train_phase2 import random_baseline_energy
from src.quantum.vqe_torch import make_torch_vqe


def _to_device(cached_instance, device):
    cached_instance.hamiltonian_graph = cached_instance.hamiltonian_graph.to(device)
    cached_instance.ansatz_graph = cached_instance.ansatz_graph.to(device)
    cached_instance.hardware_graph = cached_instance.hardware_graph.to(device)
    return cached_instance


def _build_circuit(cached_instance, qdevice, diff_method):
    """Rebuilds the Hamiltonian from the few cached numbers (family, coupling,
    n_qubits) only -- never touches the raw .h5 file, so this (and the rest
    of training) works from data_cache/ alone with no external/VQEzy present
    on this machine at all."""
    hamiltonian, n_qubits = build_hamiltonian_from_params(
        cached_instance.family, cached_instance.coupling, cached_instance.n_qubits
    )
    import pennylane as qml

    dev = qml.device(qdevice, wires=n_qubits)
    circuit = make_torch_vqe(hamiltonian, n_qubits, cached_instance.n_layers, dev=dev, diff_method=diff_method)
    return circuit


def evaluate_split(model, data, device, qdevice, n_trials=5, seed=42, tag=""):
    model.eval()
    rows = []
    for ci in data:
        circuit = _build_circuit(ci, qdevice, diff_method="best")
        with torch.no_grad():
            theta0 = model(ci)
        gnn_energy = circuit(theta0.to("cpu")).item()
        rand_energy = random_baseline_energy(circuit, ci.n_layers, ci.n_qubits, n_trials=n_trials, seed=seed)
        rows.append(
            {
                "family": ci.family,
                "sample": ci.sample,
                "n_qubits": ci.n_qubits,
                "unseen_qubit_count": ci.unseen_qubit_count,
                "ground_energy": ci.ground_energy,
                "gnn_init_energy": gnn_energy,
                "random_init_energy": rand_energy,
                "gnn_better": gnn_energy < rand_energy,
            }
        )
    model.train()
    wins = sum(r["gnn_better"] for r in rows)
    print(f"  [{tag}] GNN beat random init on {wins}/{len(rows)} instances")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(DATASET_DIR))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--gnn-layers", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--unroll-steps", type=int, default=0)
    ap.add_argument("--unroll-lr", type=float, default=0.1)
    ap.add_argument("--qdevice", default="lightning.qubit", help="PennyLane device; use lightning.gpu if installed")
    ap.add_argument("--device", default="auto", help="auto|cuda|cpu -- device for the GNN")
    ap.add_argument("--val-every", type=int, default=5)
    ap.add_argument("--ckpt-name", default="gpu_initializer.pt")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"PennyLane device for circuit simulation: {args.qdevice} (CPU-bound unless this is a GPU-native plugin)")

    data_dir = Path(args.data_dir)
    train_data = torch.load(data_dir / "train.pt", weights_only=False)
    val_data = torch.load(data_dir / "val.pt", weights_only=False)
    test_data = torch.load(data_dir / "test.pt", weights_only=False)
    print(f"Loaded cached dataset: train={len(train_data)}  val={len(val_data)}  test={len(test_data)}")

    train_data = [_to_device(ci, device) for ci in train_data]
    val_data = [_to_device(ci, device) for ci in val_data]
    test_data = [_to_device(ci, device) for ci in test_data]

    diff_method = "parameter-shift" if args.unroll_steps > 0 else "best"
    t_circ0 = time.perf_counter()
    train_circuits = [_build_circuit(ci, args.qdevice, diff_method) for ci in train_data]
    print(f"Built {len(train_circuits)} training circuits in {time.perf_counter()-t_circ0:.1f}s")

    torch.manual_seed(args.seed)
    model = VQEInitializer(hidden=args.hidden, gnn_layers=args.gnn_layers).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    loss_label = f"E(theta_{args.unroll_steps})" if args.unroll_steps > 0 else "E(theta_0)"
    print(f"\nTraining on {len(train_data)} instances, {args.epochs} epochs, loss={loss_label}")

    history = []
    t_train0 = time.perf_counter()
    for epoch in range(args.epochs):
        epoch_losses = []
        t_epoch0 = time.perf_counter()
        for ci, circuit in zip(train_data, train_circuits):
            opt.zero_grad()
            if args.unroll_steps > 0:
                theta0 = model(ci)
                theta = theta0.to("cpu")
                for _ in range(args.unroll_steps):
                    energy = circuit(theta)
                    grad = torch.autograd.grad(energy, theta, create_graph=True)[0]
                    theta = theta - args.unroll_lr * grad
                energy = circuit(theta)
            else:
                theta0 = model(ci)
                energy = circuit(theta0.to("cpu"))
            energy.backward()
            opt.step()
            epoch_losses.append(energy.item())
        mean_loss = sum(epoch_losses) / len(epoch_losses)
        dt = time.perf_counter() - t_epoch0
        history.append({"epoch": epoch, "mean_loss": mean_loss, "epoch_seconds": dt})
        print(f"  epoch {epoch:3d}: mean train {loss_label} = {mean_loss:9.4f}  ({dt:.1f}s)")

        if (epoch + 1) % args.val_every == 0 or epoch == args.epochs - 1:
            evaluate_split(model, val_data, device, args.qdevice, tag=f"val@epoch{epoch}")

    print(f"\nTotal training time: {time.perf_counter()-t_train0:.1f}s")

    print("\n=== Final test evaluation ===")
    same_size = [ci for ci in test_data if not ci.unseen_qubit_count]
    unseen_size = [ci for ci in test_data if ci.unseen_qubit_count]
    results = {}
    if same_size:
        results["unseen_sample"] = evaluate_split(model, same_size, device, args.qdevice, tag="test/unseen_sample")
    if unseen_size:
        results["unseen_qubit_count"] = evaluate_split(model, unseen_size, device, args.qdevice, tag="test/unseen_qubit_count")

    out_dir = REPO_ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / "gpu_train_history.json", "w") as f:
        json.dump(history, f, indent=2)
    with open(out_dir / "gpu_test_results.json", "w") as f:
        json.dump(results, f, indent=2)

    ckpt_dir = REPO_ROOT / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    torch.save(model.state_dict(), ckpt_dir / args.ckpt_name)
    print(f"\nSaved checkpoint -> {ckpt_dir / args.ckpt_name}")
    print(f"Saved training history -> {out_dir / 'gpu_train_history.json'}")
    print(f"Saved test results -> {out_dir / 'gpu_test_results.json'}")


if __name__ == "__main__":
    main()
