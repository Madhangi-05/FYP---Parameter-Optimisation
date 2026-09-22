"""The actual test: does starting VQE+GD / VQE+QNGD from the trained GNN's
theta_0 reach the true ground energy in fewer circuit evaluations than
starting from a random theta_0? (Not "is the starting energy lower" --
that was Phase 2's decision gate. This runs the full optimizer from both
starting points and compares real circuit-eval counts, same accounting as
Phase 1's baseline: qml.Tracker device-level executions.)

Reuses the held-out split from train_phase2.py (same seed -> same instances,
guaranteed not in the checkpoint's training set) and takes 2 instances from
each of the 5 <=8-qubit categories (12-qubit skipped here to keep runtime
bounded -- the GNN's initial-energy win rate was already weakest there, a
natural follow-up once this core comparison is in).
"""
from __future__ import annotations

import argparse

import pandas as pd
import pennylane as qml
import torch
from pennylane import numpy as pnp

from src.data.vqezy_loader import load_instance
from src.graphs.build_instance import build_graph_instance
from src.models.initializer import VQEInitializer
from src.quantum.hamiltonians import build_hamiltonian, true_ground_energy
from src.quantum.run_baseline_batch import REPO_ROOT
from src.quantum.train_phase2 import UNSEEN_SIZE_FILE, build_split
from src.quantum.vqe_baseline import _make_qnode, run_optimizer


def select_comparison_instances(n_per_category: int = 2, seed: int = 0):
    _, held_out = build_split(
        train_per_file=40, heldout_per_file=8, heldout_unseen_size=10, seed=seed
    )
    by_file: dict[str, list] = {}
    for h5_rel, sample in held_out:
        if h5_rel == UNSEEN_SIZE_FILE:
            continue  # 12-qubit: skipped here, see module docstring
        by_file.setdefault(h5_rel, []).append(sample)
    selected = []
    for h5_rel, samples in by_file.items():
        selected += [(h5_rel, s) for s in samples[:n_per_category]]
    return selected


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(REPO_ROOT / "checkpoints" / "phase2_initializer_scaled.pt"))
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--gnn-layers", type=int, default=2)
    ap.add_argument("--n-layers", type=int, default=2)
    ap.add_argument("--n-steps", type=int, default=50)
    ap.add_argument("--gd-stepsize", type=float, default=0.1)
    ap.add_argument("--qngd-stepsize", type=float, default=0.02)
    ap.add_argument("--tol", type=float, default=0.05)
    ap.add_argument("--n-per-category", type=int, default=2)
    ap.add_argument("--random-seed", type=int, default=123)  # distinct from training/eval seeds used so far
    args = ap.parse_args()

    model = VQEInitializer(hidden=args.hidden, gnn_layers=args.gnn_layers)
    model.load_state_dict(torch.load(args.ckpt))
    model.eval()
    print(f"Loaded checkpoint: {args.ckpt}")

    instances = select_comparison_instances(args.n_per_category)
    print(f"Comparing on {len(instances)} held-out instances (not in training set)\n")

    rows = []
    for h5_rel, sample in instances:
        instance = load_instance(REPO_ROOT / h5_rel, sample)
        gi = build_graph_instance(instance, n_layers=args.n_layers, topology="ring", noise_level="medium", seed=0)
        hamiltonian, n_qubits = build_hamiltonian(instance)
        ground = true_ground_energy(hamiltonian, n_qubits)
        target = ground if ground is not None else float(instance.reference_loss_history[-1])

        with torch.no_grad():
            theta_gnn_t = model(gi)
        theta_gnn = pnp.array(theta_gnn_t.numpy(), requires_grad=True)

        rng = pnp.random.default_rng(args.random_seed)
        theta_random = pnp.array(
            rng.uniform(0, 2 * pnp.pi, size=(args.n_layers, n_qubits, 2)), requires_grad=True
        )

        dev = qml.device("lightning.qubit", wires=n_qubits)
        print(f"=== {instance.family}/{sample} ({n_qubits}q, target={target:.3f}) ===")
        for init_name, theta_init in [("gnn", theta_gnn), ("random", theta_random)]:
            for opt_name, opt in [
                ("gradient_descent", qml.GradientDescentOptimizer(stepsize=args.gd_stepsize)),
                ("qngd", qml.QNGOptimizer(stepsize=args.qngd_stepsize)),
            ]:
                circuit = _make_qnode(hamiltonian, n_qubits, args.n_layers, dev)
                res = run_optimizer(circuit, dev, theta_init, opt, args.n_steps, target, args.tol)
                print(
                    f"  init={init_name:6s} opt={opt_name:>17s}: "
                    f"final_energy={res['final_energy']:9.4f}  "
                    f"circuit_evals={res['circuit_evals']:6d}  "
                    f"steps_to_thr={str(res['steps_to_threshold']):>4s}"
                )
                # circuit_evals is the TOTAL spent over the fixed n_steps budget --
                # nearly identical regardless of starting point (same optimizer,
                # same circuit -> same per-step cost). The metric that actually
                # depends on the starting point is evals SPENT TO REACH THE
                # THRESHOLD, derived from steps_to_threshold (NaN if never reached
                # within the budget -- a real outcome, not a missing value).
                evals_per_step = res["circuit_evals"] / args.n_steps
                evals_to_threshold = (
                    evals_per_step * res["steps_to_threshold"]
                    if res["steps_to_threshold"] is not None
                    else float("nan")
                )
                rows.append(
                    {
                        "family": instance.family,
                        "sample": sample,
                        "n_qubits": n_qubits,
                        "init": init_name,
                        "optimizer": opt_name,
                        "target_energy": target,
                        "final_energy": res["final_energy"],
                        "final_energy_error": res["final_energy"] - target,
                        "circuit_evals_total_budget": res["circuit_evals"],
                        "steps_to_threshold": res["steps_to_threshold"],
                        "evals_to_threshold": evals_to_threshold,
                    }
                )
        print()

    df = pd.DataFrame(rows)
    out_path = REPO_ROOT / "results" / "gnn_vs_random_init_comparison.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved -> {out_path}\n")

    print(df.to_string(index=False))

    print("\n--- Reached threshold within budget? (count out of total rows) ---")
    print(df.groupby(["optimizer", "init"])["steps_to_threshold"].apply(lambda s: s.notna().sum()).to_string())

    print("\n--- Among cases where it reached threshold: mean evals_to_threshold ---")
    print(df.dropna(subset=["evals_to_threshold"]).groupby(["optimizer", "init"])["evals_to_threshold"].mean().to_string())

    print("\n--- Per-instance-pair comparison: did GNN-init reach threshold in FEWER evals than random-init? ---")
    pivot = df.pivot_table(
        index=["family", "sample", "n_qubits", "optimizer"], columns="init", values="evals_to_threshold"
    )

    def _outcome(row):
        gnn_reached, rand_reached = pd.notna(row["gnn"]), pd.notna(row["random"])
        if gnn_reached and rand_reached:
            return "gnn_fewer_evals" if row["gnn"] < row["random"] else "random_fewer_evals"
        if gnn_reached and not rand_reached:
            return "gnn_only_reached"
        if rand_reached and not gnn_reached:
            return "random_only_reached"
        return "neither_reached"

    pivot["outcome"] = pivot.apply(_outcome, axis=1)
    print(pivot.to_string())
    print("\nOutcome counts:")
    print(pivot["outcome"].value_counts().to_string())


if __name__ == "__main__":
    main()
