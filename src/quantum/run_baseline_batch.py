"""Phase 1 baseline, batched across 10 real VQEzy instances spanning all
three qmanybody families (xyz/fh/ti) and multiple qubit counts (4/6/8),
so the GD-vs-QNGD comparison isn't a single data point.
"""
from __future__ import annotations

import argparse
import time

import pandas as pd
import pennylane as qml
from pennylane import numpy as pnp

from src.data.vqezy_loader import load_instance
from src.quantum.ansatz import n_params
from src.quantum.hamiltonians import build_hamiltonian, true_ground_energy
from src.quantum.vqe_baseline import REPO_ROOT, _make_qnode, run_optimizer

INSTANCES = [
    ("external/VQEzy/qmanybody/xyz_4_qubit.h5", "sample_0"),
    ("external/VQEzy/qmanybody/xyz_4_qubit.h5", "sample_1"),
    ("external/VQEzy/qmanybody/xyz_4_qubit.h5", "sample_2"),
    ("external/VQEzy/qmanybody/fh_4_qubit.h5", "sample_0"),
    ("external/VQEzy/qmanybody/fh_4_qubit.h5", "sample_1"),
    ("external/VQEzy/qmanybody/fh_6_qubit.h5", "sample_0"),
    ("external/VQEzy/qmanybody/fh_6_qubit.h5", "sample_1"),
    ("external/VQEzy/qmanybody/fh_8_qubit.h5", "sample_0"),
    ("external/VQEzy/qmanybody/ti_8_qubit.h5", "sample_0"),
    ("external/VQEzy/qmanybody/ti_8_qubit.h5", "sample_1"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-layers", type=int, default=2)
    ap.add_argument("--n-steps", type=int, default=50)
    ap.add_argument("--gd-stepsize", type=float, default=0.1)
    ap.add_argument("--qngd-stepsize", type=float, default=0.02)
    ap.add_argument("--tol", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows = []
    t_start = time.perf_counter()
    for h5_rel, sample in INSTANCES:
        h5_path = REPO_ROOT / h5_rel
        instance = load_instance(h5_path, sample)
        hamiltonian, n_qubits = build_hamiltonian(instance)
        vqezy_reference = float(instance.reference_loss_history[-1])
        exact_ground = true_ground_energy(hamiltonian, n_qubits)
        target_energy = exact_ground if exact_ground is not None else vqezy_reference
        p = n_params(n_qubits, args.n_layers)
        rng = pnp.random.default_rng(args.seed)
        init = pnp.array(
            rng.uniform(0, 2 * pnp.pi, size=(args.n_layers, n_qubits, 2)), requires_grad=True
        )
        dev = qml.device("lightning.qubit", wires=n_qubits)

        target_kind = "exact" if exact_ground is not None else "vqezy_ref(unverified)"
        print(
            f"=== {instance.family} {sample} (n_qubits={n_qubits}, n_params={p}, "
            f"target={target_energy:.4f} [{target_kind}], vqezy_ref={vqezy_reference:.4f}) ===",
            flush=True,
        )
        for name, opt in [
            ("gradient_descent", qml.GradientDescentOptimizer(stepsize=args.gd_stepsize)),
            ("qngd", qml.QNGOptimizer(stepsize=args.qngd_stepsize)),
        ]:
            circuit = _make_qnode(hamiltonian, n_qubits, args.n_layers, dev)
            res = run_optimizer(circuit, dev, init, opt, args.n_steps, target_energy, args.tol)
            print(
                f"  {name:>17s}: final_energy={res['final_energy']:9.4f}  "
                f"circuit_evals={res['circuit_evals']:6d}  "
                f"steps_to_thr={str(res['steps_to_threshold']):>4s}  "
                f"time={res['wall_clock_s']:.1f}s",
                flush=True,
            )
            rows.append(
                {
                    "family": instance.family,
                    "sample": sample,
                    "n_qubits": n_qubits,
                    "n_params": p,
                    "optimizer": name,
                    "target_energy": target_energy,
                    "target_kind": target_kind,
                    "vqezy_reference": vqezy_reference,
                    "final_energy": res["final_energy"],
                    "final_energy_error": res["final_energy"] - target_energy,
                    "circuit_evals": res["circuit_evals"],
                    "steps_to_threshold": res["steps_to_threshold"],
                    "wall_clock_s": res["wall_clock_s"],
                }
            )

    df = pd.DataFrame(rows)
    out_dir = REPO_ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    df.to_csv(out_dir / "phase1_baseline_batch.csv", index=False)
    df.to_json(out_dir / "phase1_baseline_batch.json", orient="records", indent=2)

    print(f"\nTotal wall clock: {time.perf_counter() - t_start:.1f}s")
    print(f"Saved -> {out_dir / 'phase1_baseline_batch.csv'}")
    print()
    print(df.to_string(index=False))

    # quick aggregate: win rate + mean evals by optimizer
    summary = (
        df.groupby("optimizer")[["circuit_evals", "final_energy"]]
        .mean()
        .rename(columns={"circuit_evals": "mean_circuit_evals", "final_energy": "mean_final_energy"})
    )
    print("\n--- Aggregate over all instances ---")
    print(summary.to_string())


if __name__ == "__main__":
    main()
