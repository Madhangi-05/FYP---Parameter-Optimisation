"""The real test, on the cached held-out split: does starting VQE+GD /
VQE+QNGD from the trained GNN's theta_0 reach the ground energy in fewer
circuit evaluations than starting from a random theta_0?

Differs from compare_init_strategies.py in two ways:
  - Uses data_cache/test.pt (the exact split train_gpu.py held out), so it
    needs no external/VQEzy checkout and is guaranteed disjoint from the
    checkpoint's training set. The old script draws from train_phase2.py's
    split, which can overlap the cached train split.
  - Compares against several random starts per instance (not one), so the
    baseline isn't a single lucky/unlucky draw.

Same accounting as Phase 1 (vqe_baseline.run_optimizer): qml.Tracker
device-level executions, parameter-shift gradients.

Two thresholds are reported, because a 2-layer CZRXRY ansatz cannot always
reach the exact ground energy -- with only the exact target, "neither
reached" would dominate and hide real differences in convergence speed:
  - exact:  E <= E_ground + tol
  - best:   within rel_tol of the best energy ANY run reached on that
            instance (i.e. what this ansatz can actually achieve)
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
from src.models.initializer import VQEInitializer
from src.quantum.run_baseline_batch import REPO_ROOT


def _run_one(job):
    # Imported here: workers are spawned, and keeping torch/CUDA out of them
    # avoids every worker grabbing GPU context or oversubscribing threads.
    import pennylane as qml
    from pennylane import numpy as pnp

    from src.quantum.hamiltonians import build_hamiltonian_from_params
    from src.quantum.vqe_baseline import _make_qnode, run_optimizer

    hamiltonian, n_qubits = build_hamiltonian_from_params(job["family"], job["coupling"], job["n_qubits"])
    dev = qml.device("lightning.qubit", wires=n_qubits)
    circuit = _make_qnode(hamiltonian, n_qubits, job["n_layers"], dev)
    if job["optimizer"] == "gradient_descent":
        opt = qml.GradientDescentOptimizer(stepsize=job["gd_stepsize"])
    else:
        opt = qml.QNGOptimizer(stepsize=job["qngd_stepsize"])
    theta = pnp.array(job["theta0"], requires_grad=True)
    e0 = float(circuit(theta))
    res = run_optimizer(circuit, dev, theta, opt, job["n_steps"], job["ground"], job["tol"])
    return {
        **{k: job[k] for k in ["family", "sample", "n_qubits", "unseen_qubit_count", "init", "optimizer", "ground"]},
        "init_energy": e0,
        "final_energy": res["final_energy"],
        "circuit_evals_total": res["circuit_evals"],
        "evals_per_step": res["circuit_evals"] / job["n_steps"],
        "history": res["history"],
    }


def _steps_to(history, target):
    for i, e in enumerate(history):
        if e <= target:
            return i + 1
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(DATASET_DIR))
    ap.add_argument("--ckpt", default=str(REPO_ROOT / "checkpoints" / "gpu_initializer.pt"))
    # Must match the checkpoint's training config (train_gpu.py defaults: 64/3).
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--gnn-layers", type=int, default=3)
    ap.add_argument("--split", default="test")
    ap.add_argument("--n-steps", type=int, default=100)
    ap.add_argument("--gd-stepsize", type=float, default=0.1)
    ap.add_argument("--qngd-stepsize", type=float, default=0.02)
    ap.add_argument("--tol", type=float, default=0.05)
    ap.add_argument("--rel-tol", type=float, default=0.01)
    ap.add_argument("--n-random", type=int, default=3)
    ap.add_argument("--random-seed", type=int, default=123)
    ap.add_argument("--per-group", type=int, default=0, help="cap instances per (family,n_qubits); 0 = all")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--tag", default="gpu_initializer")
    args = ap.parse_args()

    data = torch.load(Path(args.data_dir) / f"{args.split}.pt", weights_only=False)
    if args.per_group:
        seen: dict = {}
        kept = []
        for ci in data:
            k = (ci.family, ci.n_qubits)
            if seen.get(k, 0) < args.per_group:
                kept.append(ci)
                seen[k] = seen.get(k, 0) + 1
        data = kept

    model = VQEInitializer(hidden=args.hidden, gnn_layers=args.gnn_layers)
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
    model.eval()
    print(f"Loaded checkpoint: {args.ckpt}")
    print(f"Comparing on {len(data)} {args.split} instances, {args.n_random} random starts each, {args.n_steps} steps")

    rng = np.random.default_rng(args.random_seed)
    jobs = []
    for ci in data:
        with torch.no_grad():
            theta_gnn = model(ci).numpy().astype(np.float64)
        inits = [("gnn", theta_gnn)] + [
            (f"random{r}", rng.uniform(0, 2 * np.pi, size=(ci.n_layers, ci.n_qubits, 2)))
            for r in range(args.n_random)
        ]
        for init_name, theta0 in inits:
            for opt_name in ["gradient_descent", "qngd"]:
                jobs.append(
                    {
                        "family": ci.family,
                        "sample": ci.sample,
                        "coupling": np.asarray(ci.coupling),
                        "n_qubits": ci.n_qubits,
                        "n_layers": ci.n_layers,
                        "unseen_qubit_count": bool(ci.unseen_qubit_count),
                        "ground": float(ci.ground_energy),
                        "init": init_name,
                        "theta0": theta0,
                        "optimizer": opt_name,
                        "n_steps": args.n_steps,
                        "gd_stepsize": args.gd_stepsize,
                        "qngd_stepsize": args.qngd_stepsize,
                        "tol": args.tol,
                    }
                )

    rows = []
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context("spawn")) as ex:
        for i, row in enumerate(ex.map(_run_one, jobs, chunksize=4)):
            rows.append(row)
            if (i + 1) % 100 == 0 or i + 1 == len(jobs):
                print(f"  {i + 1}/{len(jobs)} runs done", flush=True)

    df = pd.DataFrame(rows)
    # "best achievable" per instance: lowest energy any run (any init, any optimizer) reached.
    # Keyed with n_qubits: sample names repeat across fh_4/fh_6/fh_8.
    best = df.groupby(["family", "n_qubits", "sample"])["history"].apply(lambda hs: min(min(h) for h in hs)).rename("best_reached")
    df = df.join(best, on=["family", "n_qubits", "sample"])
    df["target_best"] = df["best_reached"] + args.rel_tol * df["best_reached"].abs()
    df["steps_exact"] = [_steps_to(h, g + args.tol) for h, g in zip(df["history"], df["ground"])]
    df["steps_best"] = [_steps_to(h, t) for h, t in zip(df["history"], df["target_best"])]
    df["evals_exact"] = df["steps_exact"] * df["evals_per_step"]
    df["evals_best"] = df["steps_best"] * df["evals_per_step"]
    df["init_kind"] = np.where(df["init"] == "gnn", "gnn", "random")

    out_dir = REPO_ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    df_out = df.copy()
    df_out["history"] = df_out["history"].apply(json.dumps)
    out_path = out_dir / f"convergence_{args.tag}_{args.split}.csv"
    df_out.to_csv(out_path, index=False)
    print(f"\nSaved -> {out_path}")

    summarize(df, args.n_steps)


def summarize(df, n_steps):
    """Per (optimizer, family, n_qubits): reach rate and median evals for GNN vs
    random, plus the per-instance head-to-head (GNN vs the mean random start).
    Unreached runs count as n_steps+1 steps in the median, so a method that
    often fails can't look fast by only being scored on its easy cases."""
    for thr in ["exact", "best"]:
        steps_col, evals_col = f"steps_{thr}", f"evals_{thr}"
        d = df.copy()
        d["evals_pen"] = d[steps_col].fillna(n_steps + 1) * d["evals_per_step"]
        print(f"\n=== Threshold: {thr} ===")
        g = d.groupby(["optimizer", "family", "n_qubits", "init_kind"])
        tab = pd.DataFrame(
            {
                "reach_rate": g[steps_col].apply(lambda s: s.notna().mean()),
                "median_evals_pen": g["evals_pen"].median(),
                "mean_final_gap": g.apply(lambda x: (x["final_energy"] - x["ground"]).mean(), include_groups=False),
            }
        ).unstack("init_kind")
        print(tab.round(3).to_string())

        per_inst = d.groupby(["optimizer", "family", "n_qubits", "sample", "init_kind"])["evals_pen"].mean().unstack("init_kind")
        per_inst["gnn_fewer"] = per_inst["gnn"] < per_inst["random"]
        per_inst["ratio"] = per_inst["gnn"] / per_inst["random"]
        h2h = per_inst.groupby(["optimizer", "family", "n_qubits"]).agg(
            gnn_fewer=("gnn_fewer", "sum"), n=("gnn_fewer", "count"), median_eval_ratio=("ratio", "median")
        )
        print("\nHead-to-head (GNN evals vs mean of random starts; ratio<1 = GNN cheaper):")
        print(h2h.round(3).to_string())
        overall = per_inst.groupby("optimizer").agg(
            gnn_fewer=("gnn_fewer", "sum"), n=("gnn_fewer", "count"), median_eval_ratio=("ratio", "median")
        )
        print("\nOverall:")
        print(overall.round(3).to_string())


if __name__ == "__main__":
    main()
