"""VQE+QNGD on the cached held-out split, from either:
  --init random : the SAME random starts the vanilla-VQE run used
                  (results/random_starts_<split>.npz, written by
                  run_vanilla_vqe.py) -- so vs vanilla, the optimizer is the
                  only difference. This is the baseline for later studies.
  --init gnn    : theta_0 predicted by a trained GNN checkpoint (--ckpt).

QNGD = qml.QNGOptimizer (block-diag metric tensor, pinv, lam=0), stepsize
0.02 as in Phase 1 (vqe_baseline.py: 0.1 diverges under QNGD). Circuit evals
come from qml.Tracker over the whole run -- gradient (parameter-shift) AND
metric-tensor executions, per commuting measurement group -- the same
device-level accounting run_vanilla_vqe.py charges Adam.
"""
from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.data.build_dataset import DATASET_DIR
from src.quantum.run_baseline_batch import REPO_ROOT
from src.quantum.run_vanilla_vqe import start_key, steps_to


def _run_one(job):
    import pennylane as qml
    from pennylane import numpy as pnp

    from src.quantum.hamiltonians import build_hamiltonian_from_params
    from src.quantum.vqe_baseline import _make_qnode

    hamiltonian, n = build_hamiltonian_from_params(job["family"], job["coupling"], job["n_qubits"])
    dev = qml.device("lightning.qubit", wires=n)
    circuit = _make_qnode(hamiltonian, n, job["n_layers"], dev)
    opt = qml.QNGOptimizer(stepsize=job["stepsize"])
    params = pnp.array(job["theta0"], requires_grad=True)
    history = np.zeros(job["n_steps"])
    with qml.Tracker(dev) as tracker:
        for i in range(job["n_steps"]):
            params, energy = opt.step_and_cost(circuit, params)
            history[i] = float(energy)  # energy BEFORE step i's update, same convention as vanilla
    evals = tracker.totals.get("executions", 0)
    return {
        **{k: job[k] for k in ["family", "sample", "n_qubits", "unseen_qubit_count", "init", "ground"]},
        "init_energy": float(history[0]),
        "final_energy": float(history[-1]),
        "evals_per_step": evals / job["n_steps"],
        "history": history.tolist(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(DATASET_DIR))
    ap.add_argument("--split", default="test")
    ap.add_argument("--init", choices=["random", "gnn"], default="random")
    ap.add_argument("--ckpt", default=None, help="GNN checkpoint, required for --init gnn")
    # Must match the checkpoint's training config (train_gpu.py defaults: 64/3).
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--gnn-layers", type=int, default=3)
    ap.add_argument("--no-hardware", action="store_true", help="checkpoint was trained without the hardware/noise graph")
    ap.add_argument("--n-random", type=int, default=3)
    ap.add_argument("--n-steps", type=int, default=200)
    ap.add_argument("--stepsize", type=float, default=0.02)
    ap.add_argument("--tol", type=float, default=0.05)
    ap.add_argument("--rel-tol", type=float, default=0.01)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = ap.parse_args()

    data = torch.load(Path(args.data_dir) / f"{args.split}.pt", weights_only=False)
    out_dir = REPO_ROOT / "results"

    if args.init == "random":
        starts = dict(np.load(out_dir / f"random_starts_{args.split}.npz"))
        inits = [(ci, f"random{r}", starts[start_key(ci, r)]) for ci in data for r in range(args.n_random)]
        tag = args.tag or "random"
    else:
        from src.models.initializer import VQEInitializer

        if not args.ckpt:
            ap.error("--init gnn needs --ckpt")
        model = VQEInitializer(hidden=args.hidden, gnn_layers=args.gnn_layers, use_hardware=not args.no_hardware)
        model.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
        model.eval()
        with torch.no_grad():
            inits = [(ci, "gnn", model(ci).numpy().astype(np.float64)) for ci in data]
        tag = args.tag or f"gnn_{Path(args.ckpt).stem}"

    jobs = [
        {
            "family": ci.family,
            "sample": ci.sample,
            "coupling": np.asarray(ci.coupling),
            "n_qubits": ci.n_qubits,
            "n_layers": ci.n_layers,
            "unseen_qubit_count": bool(ci.unseen_qubit_count),
            "ground": float(ci.ground_energy),
            "init": name,
            "theta0": theta0,
            "n_steps": args.n_steps,
            "stepsize": args.stepsize,
        }
        for ci, name, theta0 in inits
    ]
    print(f"VQE+QNGD (stepsize={args.stepsize}, {args.n_steps} steps), init={tag}: {len(jobs)} runs")

    rows = []
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context("spawn")) as ex:
        for i, row in enumerate(ex.map(_run_one, jobs, chunksize=2)):
            rows.append(row)
            if (i + 1) % 50 == 0 or i + 1 == len(jobs):
                print(f"  {i + 1}/{len(jobs)} runs done", flush=True)

    df = pd.DataFrame(rows)
    df["steps_exact"] = [steps_to(h, g + args.tol) for h, g in zip(df.history, df.ground)]
    df["steps_rel"] = [steps_to(h, g + args.rel_tol * abs(g)) for h, g in zip(df.history, df.ground)]
    df["evals_exact"] = df.steps_exact * df.evals_per_step
    df["evals_rel"] = df.steps_rel * df.evals_per_step
    df["evals_total_budget"] = args.n_steps * df.evals_per_step

    out = df.copy()
    out["history"] = out.history.apply(json.dumps)
    out_path = out_dir / f"qngd_{tag}_{args.split}.csv"
    out.to_csv(out_path, index=False)
    print(f"\nSaved -> {out_path}")

    df["final_gap"] = df.final_energy - df.ground
    g = df.groupby(["family", "n_qubits"])
    print(f"\nVQE+QNGD init={tag}, per group -- thresholds: exact = ground+{args.tol}, rel = within {args.rel_tol:.0%} of ground")
    print(
        pd.DataFrame(
            {
                "runs": g.size(),
                "evals_per_step": g.evals_per_step.mean(),
                "mean_final_gap": g.final_gap.mean(),
                "reach_exact": g.steps_exact.apply(lambda s: s.notna().mean()),
                "reach_rel": g.steps_rel.apply(lambda s: s.notna().mean()),
                "median_evals_rel(reached)": g.evals_rel.median(),
                "evals_full_budget": g.evals_total_budget.mean(),
            }
        ).round(3).to_string()
    )


if __name__ == "__main__":
    main()
