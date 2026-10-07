"""Fast model-selection evaluator: VQE+QNGD (200 steps, eta=0.02) on the
VALIDATION split, simulated with torch_sim.py on the GPU (validated against
qml.QNGOptimizer). Compares any set of GNN checkpoints against the same
random starts, so pilot/ablation decisions never touch the test split.

Circuit cost: every init of the same instance pays the same per-step cost,
so per-step cost only weights instances against each other. It is taken as
(2P+1) x (measurement groups), the parameter-shift part of the qml.Tracker
count (the block-diag metric adds a few % on top, equally for all inits).

Usage:
  python -m src.quantum.eval_sim_qngd --models old_gd3:checkpoints/gpu_initializer_unroll3.pt:hw \
      gnnA:checkpoints/qngd_clean.pt:nohw  pilot:checkpoints/pilot_clean_K15_traj.pt:nohw
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.data.build_dataset import DATASET_DIR
from src.data.build_noisy_dataset import NOISY_DATASET_DIR
from src.models.initializer import VQEInitializer
from src.quantum.hamiltonians import build_hamiltonian_from_params
from src.quantum.noise_model import noise_from_hardware_graph
from src.quantum.torch_sim import CZRXRYSim, hamiltonian_matrix


def per_step_cost(hamiltonian, n_qubits, n_layers):
    groups = len(qml_groups(hamiltonian))
    return (2 * n_layers * n_qubits * 2 + 1) * groups


def qml_groups(hamiltonian):
    import pennylane as qml

    _, ops = hamiltonian.terms()
    return qml.pauli.group_observables(ops, grouping_type="qwc")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True, help="name:path:hw|nohw")
    ap.add_argument("--split", default="val")
    ap.add_argument("--noisy", action="store_true")
    ap.add_argument("--n-random", type=int, default=3)
    ap.add_argument("--seed", type=int, default=99)
    ap.add_argument("--n-steps", type=int, default=200)
    ap.add_argument("--stepsize", type=float, default=0.02)
    ap.add_argument("--rel-tol", type=float, default=0.01)
    ap.add_argument("--max-qubits", type=int, default=8)
    ap.add_argument("--batch-elems", type=int, default=2**21)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ddir = NOISY_DATASET_DIR if args.noisy else DATASET_DIR
    data = [ci for ci in torch.load(Path(ddir) / f"{args.split}.pt", weights_only=False) if ci.n_qubits <= args.max_qubits]

    models = {}
    for spec in args.models:
        name, path, hw = spec.split(":")
        m = VQEInitializer(use_hardware=(hw == "hw"))
        m.load_state_dict(torch.load(path, map_location="cpu"))
        models[name] = m.eval()

    rng = np.random.default_rng(args.seed)
    runs = []
    with torch.no_grad():
        for ci in data:
            for r in range(args.n_random):
                runs.append((ci, f"random{r}", rng.uniform(0, 2 * np.pi, (ci.n_layers, ci.n_qubits, 2))))
            for name, m in models.items():
                runs.append((ci, name, m(ci).numpy()))

    cache = {}
    rows = []
    by_n: dict[int, list] = {}
    for run in runs:
        by_n.setdefault(run[0].n_qubits, []).append(run)
    for n, group in sorted(by_n.items()):
        bs = max(1, args.batch_elems // 4**n)
        for i in range(0, len(group), bs):
            chunk = group[i : i + bs]
            Hs = []
            for ci, _, _ in chunk:
                k = (ci.family, n, ci.sample)
                if k not in cache:
                    H, _ = build_hamiltonian_from_params(ci.family, ci.coupling, n)
                    cache[k] = (torch.tensor(hamiltonian_matrix(H, n), dtype=torch.complex64), per_step_cost(H, n, ci.n_layers))
                Hs.append(cache[k][0])
            noise = [noise_from_hardware_graph(ci.hardware_graph, n) for ci, _, _ in chunk] if args.noisy else None
            sim = CZRXRYSim(n, chunk[0][0].n_layers, torch.stack(Hs).to(device), noise=noise)
            theta0 = torch.tensor(np.stack([t for _, _, t in chunk]), dtype=torch.float32, device=device)
            with torch.enable_grad():
                E, _ = sim.qngd_unroll(theta0, args.n_steps, args.stepsize, create_graph=False)
            for (ci, name, _), h in zip(chunk, E.detach().cpu().numpy()[:, :-1]):
                rows.append({"family": ci.family, "n_qubits": n, "sample": ci.sample,
                             "init": "random" if name.startswith("random") else name,
                             "cost": cache[(ci.family, n, ci.sample)][1], "history": h})
        print(f"  n={n} done", flush=True)

    df = pd.DataFrame(rows)
    K = ["family", "n_qubits", "sample"]
    best = df.groupby(K).history.transform(lambda hs: min(h.min() for h in hs))
    target = best + args.rel_tol * best.abs()
    steps = [(np.nonzero(h <= t)[0][:1] + 1).tolist() for h, t in zip(df.history, target)]
    df["reached"] = [bool(s) for s in steps]
    df["spent"] = [(s[0] if s else args.n_steps) * c for s, c in zip(steps, df.cost)]
    df["final"] = df.history.apply(lambda h: h.min())

    per = df.groupby(K + ["init"]).agg(spent=("spent", "mean"), reached=("reached", "mean"), final=("final", "mean")).unstack("init")
    names = list(models)
    out = []
    for key, g in list(per.groupby(level=[0, 1])) + [(("ALL", ""), per)]:
        r = {"group": f"{key[0]} {key[1]}", "n": len(g), "reach_random": g["reached"]["random"].mean()}
        for nm in names:
            r[f"reach_{nm}"] = g["reached"][nm].mean()
            r[f"{nm}_saving%"] = 100 * (1 - g["spent"][nm].sum() / g["spent"]["random"].sum())
            r[f"{nm}_wins"] = f'{int((g["spent"][nm] < g["spent"]["random"]).sum())}/{len(g)}'
        out.append(r)
    pd.set_option("display.width", 300)
    print(f"\nVQE+QNGD on {args.split} ({'noisy' if args.noisy else 'noiseless'}), savings vs random starts:")
    print(pd.DataFrame(out).round(2).to_string(index=False))


if __name__ == "__main__":
    main()
