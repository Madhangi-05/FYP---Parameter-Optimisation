"""Vanilla VQE exactly as the VQEzy paper ran it (external/VQEzy/data_sampling.py):
Adam(lr=1e-3), 2000 fixed steps, params ~ torch.randn (N(0,1)), same
2-layer CZRXRY ansatz -- on the cached held-out test split.

This is the "no learned init, no QNGD" reference that every later study
(VQE+QNGD random init, VQE+QNGD GNN init, ...) is compared against.

Circuit-eval accounting: VQEzy used diff_method="adjoint", which only exists
on a simulator -- real hardware needs parameter-shift. So the optimization
itself runs with adjoint (fast, identical trajectory: both are exact
gradients), but each step is CHARGED the parameter-shift cost, measured with
qml.Tracker on one real parameter-shift step of that instance. That cost is
(2P+1) x (number of commuting measurement groups in H) -- the same
device-level accounting the GD/QNGD runs in this repo use (vqe_baseline.py).

The random starts are saved (results/random_starts_<split>.npz) so the
VQE+QNGD random-init baseline reuses the IDENTICAL starting points, making
optimizer the only difference between the two.
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


def _run_one(job):
    import pennylane as qml
    import torch

    from src.quantum.ansatz import czrxry_ansatz
    from src.quantum.hamiltonians import build_hamiltonian_from_params

    torch.set_num_threads(1)
    hamiltonian, n = build_hamiltonian_from_params(job["family"], job["coupling"], job["n_qubits"])
    L = job["n_layers"]
    dev = qml.device("lightning.qubit", wires=n)

    # Per-step hardware cost: one real parameter-shift step, tracked.
    @qml.qnode(dev, interface="torch", diff_method="parameter-shift")
    def ps_circuit(t):
        czrxry_ansatz(t, n, L)
        return qml.expval(hamiltonian)

    probe = torch.tensor(job["theta0"], dtype=torch.float32, requires_grad=True)
    with qml.Tracker(dev) as tracker:
        ps_circuit(probe).backward()
    evals_per_step = tracker.totals["executions"]

    @qml.qnode(dev, interface="torch", diff_method="adjoint")
    def circuit(t):
        czrxry_ansatz(t, n, L)
        return qml.expval(hamiltonian)

    # float32 like VQEzy's torch.randn params.
    params = torch.tensor(job["theta0"], dtype=torch.float32, requires_grad=True)
    opt = torch.optim.Adam([params], lr=job["lr"])
    history = np.zeros(job["n_steps"])
    for i in range(job["n_steps"]):
        opt.zero_grad()
        loss = circuit(params)
        loss.backward()
        opt.step()
        history[i] = loss.item()  # energy BEFORE step i's update, as in VQEzy
    return {
        **{k: job[k] for k in ["family", "sample", "n_qubits", "unseen_qubit_count", "init", "ground"]},
        "init_energy": float(history[0]),
        "final_energy": float(history[-1]),
        "evals_per_step": evals_per_step,
        "history": history.tolist(),
    }


def start_key(ci, r):
    # Sample names repeat across fh_4/fh_6/fh_8 files, so n_qubits must be
    # part of the key or different instances silently share a start.
    return f"{ci.family}/{ci.n_qubits}/{ci.sample}/{r}"


def make_random_starts(data, n_random, seed):
    rng = np.random.default_rng(seed)
    return {
        start_key(ci, r): rng.standard_normal((ci.n_layers, ci.n_qubits, 2))
        for ci in data
        for r in range(n_random)
    }


def steps_to(history, target):
    idx = np.nonzero(np.asarray(history) <= target)[0]
    return int(idx[0]) + 1 if len(idx) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(DATASET_DIR))
    ap.add_argument("--split", default="test")
    ap.add_argument("--n-steps", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--n-random", type=int, default=3)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--tol", type=float, default=0.05)
    ap.add_argument("--rel-tol", type=float, default=0.01)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = ap.parse_args()

    data = torch.load(Path(args.data_dir) / f"{args.split}.pt", weights_only=False)
    out_dir = REPO_ROOT / "results"
    out_dir.mkdir(exist_ok=True)

    starts_path = out_dir / f"random_starts_{args.split}.npz"
    if starts_path.exists():
        starts = dict(np.load(starts_path))
        print(f"Reusing random starts from {starts_path}")
    else:
        starts = make_random_starts(data, args.n_random, args.seed)
        np.savez(starts_path, **starts)
        print(f"Saved random starts -> {starts_path}")

    jobs = [
        {
            "family": ci.family,
            "sample": ci.sample,
            "coupling": np.asarray(ci.coupling),
            "n_qubits": ci.n_qubits,
            "n_layers": ci.n_layers,
            "unseen_qubit_count": bool(ci.unseen_qubit_count),
            "ground": float(ci.ground_energy),
            "init": f"random{r}",
            "theta0": starts[start_key(ci, r)],
            "n_steps": args.n_steps,
            "lr": args.lr,
        }
        for ci in data
        for r in range(args.n_random)
    ]
    print(f"Vanilla VQE (Adam lr={args.lr}, {args.n_steps} steps): {len(data)} instances x {args.n_random} starts = {len(jobs)} runs")

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
    out_path = out_dir / f"vanilla_vqe_{args.split}.csv"
    out.to_csv(out_path, index=False)
    print(f"\nSaved -> {out_path}")

    df["final_gap"] = df.final_energy - df.ground
    g = df.groupby(["family", "n_qubits"])
    print(f"\nVanilla VQE (paper's method), per group -- thresholds: exact = ground+{args.tol}, rel = within {args.rel_tol:.0%} of ground")
    print(
        pd.DataFrame(
            {
                "runs": g.size(),
                "evals_per_step": g.evals_per_step.first(),
                "mean_final_gap": g.final_gap.mean(),
                "reach_exact": g.steps_exact.apply(lambda s: s.notna().mean()),
                "reach_rel": g.steps_rel.apply(lambda s: s.notna().mean()),
                "median_evals_rel(reached)": g.evals_rel.median(),
                "evals_full_budget": g.evals_total_budget.first(),
            }
        ).round(3).to_string()
    )


if __name__ == "__main__":
    main()
