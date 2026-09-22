"""One combined figure: for each of the 10 instances used in the Phase 1
GD-vs-QNGD baseline (run_baseline_batch.INSTANCES), show the raw instance
data + baseline results alongside its three graphs (G_H, G_A, G_D).

Requires results/phase1_baseline_batch.csv to already exist (produced by
`python -m src.quantum.run_baseline_batch`).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data.vqezy_loader import load_instance
from src.graphs.build_instance import build_graph_instance
from src.graphs.visualize import _draw_ansatz, _draw_hamiltonian, _draw_hardware
from src.quantum.run_baseline_batch import INSTANCES

REPO_ROOT = Path(__file__).resolve().parents[2]

_COUPLING_LABEL = {"xyz": "(J1,J2,J3)", "fh": "(t,U)", "ti": "(j,h)"}


def _format_data_panel(ax, instance, gi, batch_df):
    rows = batch_df[
        (batch_df.family == instance.family)
        & (batch_df["sample"] == instance.sample_key)
        & (batch_df.n_qubits == instance.n_qubits)
    ]
    gd_row = rows[rows.optimizer == "gradient_descent"].iloc[0] if len(rows) else None
    qngd_row = rows[rows.optimizer == "qngd"].iloc[0] if len(rows) else None

    coupling_label = _COUPLING_LABEL.get(instance.family, "coupling")
    coupling_str = np.array2string(instance.coupling, precision=3, separator=", ")

    lines = [
        f"{instance.family.upper()} / {instance.sample_key}",
        f"n_qubits = {instance.n_qubits}",
        f"{coupling_label} = {coupling_str}",
        "",
    ]
    if gd_row is not None:
        lines += [
            f"target (exact) = {gd_row.target_energy:.3f}",
            f"VQEzy reference = {gd_row.vqezy_reference:.3f}",
            "",
            f"GD:   final={gd_row.final_energy:.3f}  err={gd_row.final_energy_error:.3f}",
            f"      evals={int(gd_row.circuit_evals)}",
            f"QNGD: final={qngd_row.final_energy:.3f}  err={qngd_row.final_energy_error:.3f}",
            f"      evals={int(qngd_row.circuit_evals)}",
        ]
    else:
        lines.append("(no baseline row found)")

    ax.text(
        0.02, 0.98, "\n".join(lines), transform=ax.transAxes, va="top", ha="left",
        fontsize=8.5, family="monospace",
    )
    ax.axis("off")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-layers", type=int, default=2)
    ap.add_argument("--topology", default="ring")
    ap.add_argument("--noise-level", default="medium")
    ap.add_argument(
        "--batch-csv", default=str(REPO_ROOT / "results" / "phase1_baseline_batch.csv")
    )
    ap.add_argument(
        "--out", default=str(REPO_ROOT / "results" / "all_10_instances_graphs.png")
    )
    args = ap.parse_args()

    batch_df = pd.read_csv(args.batch_csv)

    n = len(INSTANCES)
    fig, axes = plt.subplots(n, 4, figsize=(24, 4.3 * n))

    for row_i, (h5_rel, sample) in enumerate(INSTANCES):
        instance = load_instance(REPO_ROOT / h5_rel, sample)
        gi = build_graph_instance(
            instance,
            n_layers=args.n_layers,
            topology=args.topology,
            noise_level=args.noise_level,
            seed=row_i,
        )
        _format_data_panel(axes[row_i, 0], instance, gi, batch_df)
        _draw_hamiltonian(axes[row_i, 1], gi, show_title=(row_i == 0))
        _draw_ansatz(axes[row_i, 2], gi, show_title=(row_i == 0))
        _draw_hardware(fig, axes[row_i, 3], gi, show_title=(row_i == 0))
        print(f"[{row_i+1}/{n}] {instance.family}/{sample} done")

    axes[0, 0].set_title("Instance data + Phase-1 baseline results", fontsize=10, loc="left")
    fig.suptitle(
        "10 VQEzy instances: raw data + baseline results + (G_H, G_A, G_D)", fontsize=15
    )
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    out_path = Path(args.out)
    out_path.parent.mkdir(exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
