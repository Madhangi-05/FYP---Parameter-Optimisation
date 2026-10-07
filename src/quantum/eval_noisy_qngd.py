"""Noisy evaluation: VQE+QNGD under each test instance's own noise
(data_cache_noisy/test.pt), from
  - the same 3 random starts as every other study (random_starts_test.npz),
  - GNN-A (no hardware/noise graph, trained noiseless),
  - GNN-B (noise-aware: hardware graph + trained under noise),
so GNN-A vs GNN-B isolates what noise awareness buys.

Trajectories come from torch_sim.py (validated against qml.QNGOptimizer on
default.mixed to ~1e-13), batched on the GPU -- default.mixed itself would
take hours per run. Circuit cost per step is NOT re-derived here: the noisy
circuit has the same gates/measurement groups as the noiseless one, so each
step is charged the qml.Tracker count measured for that same instance in the
noiseless QNGD baseline (results/qngd_random_test.csv).

There is no exact "ground energy" under noise (the state is mixed), so the
target is relative: within rel_tol of the lowest noisy energy any method
reached on that instance. 12-qubit instances are skipped (a 4096x4096
density matrix per run is too heavy for this sweep).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.data.build_noisy_dataset import NOISY_DATASET_DIR
from src.models.initializer import VQEInitializer
from src.quantum.hamiltonians import build_hamiltonian_from_params
from src.quantum.noise_model import noise_from_hardware_graph
from src.quantum.run_baseline_batch import REPO_ROOT
from src.quantum.run_vanilla_vqe import start_key
from src.quantum.torch_sim import CZRXRYSim, hamiltonian_matrix

KEY = ["family", "n_qubits", "sample"]


def load_model(path, use_hardware, hidden, gnn_layers):
    m = VQEInitializer(hidden=hidden, gnn_layers=gnn_layers, use_hardware=use_hardware)
    m.load_state_dict(torch.load(path, map_location="cpu"))
    return m.eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-a", default=str(REPO_ROOT / "checkpoints" / "qngd_clean.pt"))
    ap.add_argument("--ckpt-b", default=str(REPO_ROOT / "checkpoints" / "qngd_noisy.pt"))
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--gnn-layers", type=int, default=3)
    ap.add_argument("--n-random", type=int, default=3)
    ap.add_argument("--n-steps", type=int, default=200)
    ap.add_argument("--stepsize", type=float, default=0.02)
    ap.add_argument("--rel-tol", type=float, default=0.01)
    ap.add_argument("--max-qubits", type=int, default=8)
    ap.add_argument("--batch-elems", type=int, default=2**21)
    ap.add_argument("--tag", default="qngd_clean_vs_noisy")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = [ci for ci in torch.load(NOISY_DATASET_DIR / "test.pt", weights_only=False) if ci.n_qubits <= args.max_qubits]
    starts = dict(np.load(REPO_ROOT / "results" / "random_starts_test.npz"))
    cost = (
        pd.read_csv(REPO_ROOT / "results" / "qngd_random_test.csv").groupby(KEY).evals_per_step.mean().to_dict()
    )
    model_a = load_model(args.ckpt_a, False, args.hidden, args.gnn_layers)
    model_b = load_model(args.ckpt_b, True, args.hidden, args.gnn_layers)

    runs = []  # (ci, init_name, theta0)
    with torch.no_grad():
        for ci in data:
            for r in range(args.n_random):
                runs.append((ci, f"random{r}", starts[start_key(ci, r)]))
            runs.append((ci, "gnnA_clean", model_a(ci).numpy().astype(np.float64)))
            runs.append((ci, "gnnB_noisy", model_b(ci).numpy().astype(np.float64)))

    rows = []
    by_n: dict[int, list] = {}
    for run in runs:
        by_n.setdefault(run[0].n_qubits, []).append(run)
    for n, group in sorted(by_n.items()):
        bs = max(1, args.batch_elems // 4**n)
        for i in range(0, len(group), bs):
            chunk = group[i : i + bs]
            H = torch.stack(
                [
                    torch.tensor(
                        hamiltonian_matrix(build_hamiltonian_from_params(ci.family, ci.coupling, n)[0], n),
                        dtype=torch.complex64,
                    )
                    for ci, _, _ in chunk
                ]
            ).to(device)
            noise = [noise_from_hardware_graph(ci.hardware_graph, n) for ci, _, _ in chunk]
            sim = CZRXRYSim(n, chunk[0][0].n_layers, H, noise=noise)
            theta0 = torch.tensor(np.stack([t for _, _, t in chunk]), dtype=torch.float32, device=device)
            with torch.enable_grad():
                energies, _ = sim.qngd_unroll(theta0, args.n_steps, args.stepsize, create_graph=False)
            energies = energies.detach().cpu().numpy()
            for (ci, name, _), hist in zip(chunk, energies):
                rows.append(
                    {
                        "family": ci.family,
                        "n_qubits": n,
                        "sample": ci.sample,
                        "noise_level": ci.noise_level,
                        "topology": ci.topology,
                        "init": name,
                        "init_kind": "random" if name.startswith("random") else name,
                        "evals_per_step": cost[(ci.family, n, ci.sample)],
                        "history": hist[:-1],  # energy BEFORE each step, as in the other studies
                    }
                )
            print(f"  n={n}: {min(i + bs, len(group))}/{len(group)} runs", flush=True)

    df = pd.DataFrame(rows)
    df["best_reached"] = df.groupby(KEY).history.transform(lambda hs: min(h.min() for h in hs))
    df["target"] = df.best_reached + args.rel_tol * df.best_reached.abs()

    def steps_to(h, t):
        i = np.nonzero(h <= t)[0]
        return int(i[0]) + 1 if len(i) else None

    df["steps"] = [steps_to(h, t) for h, t in zip(df.history, df.target)]
    df["reached"] = df.steps.notna()
    # Honest cost: evals until target, or the whole budget if never reached.
    df["spent"] = np.where(df.reached, df.steps.fillna(0) * df.evals_per_step, args.n_steps * df.evals_per_step)
    df["init_energy"] = df.history.apply(lambda h: float(h[0]))
    df["final_energy"] = df.history.apply(lambda h: float(h.min()))

    out = df.copy()
    out["history"] = out.history.apply(lambda h: json.dumps(h.tolist()))
    out_path = REPO_ROOT / "results" / f"noisy_eval_{args.tag}_test.csv"
    out.to_csv(out_path, index=False)
    print(f"\nSaved -> {out_path}")

    per = df.groupby(KEY + ["noise_level", "topology", "init_kind"]).agg(
        spent=("spent", "mean"), reached=("reached", "mean"), final=("final_energy", "mean")
    ).unstack("init_kind")

    def table(by):
        rows = []
        for key, g in per.groupby(level=by) if by else [("ALL", per)]:
            r = {"group": key if isinstance(key, str) else " ".join(map(str, key)), "n": len(g)}
            for k in ["random", "gnnA_clean", "gnnB_noisy"]:
                r[f"reach_{k}"] = g["reached"][k].mean()
            for k in ["gnnA_clean", "gnnB_noisy"]:
                r[f"{k}_wins_vs_random"] = f'{int((g["spent"][k] < g["spent"]["random"]).sum())}/{len(g)}'
                r[f"{k}_saving_%"] = 100 * (1 - g["spent"][k].sum() / g["spent"]["random"].sum())
            r["B_beats_A"] = f'{int((g["spent"]["gnnB_noisy"] < g["spent"]["gnnA_clean"]).sum())}/{len(g)}'
            rows.append(r)
        return pd.DataFrame(rows).round(2).to_string(index=False)

    pd.set_option("display.width", 250)
    for by in [["family", "n_qubits"], ["noise_level"], ["topology"], None]:
        print(f"\n--- by {by or 'overall'} ---")
        print(table(by))


if __name__ == "__main__":
    main()
