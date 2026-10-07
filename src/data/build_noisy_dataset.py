"""Derives data_cache_noisy/ from data_cache/: same instances and the same
train/val/test split, but each instance gets a freshly sampled hardware/noise
graph G_D with noise_level ~ {low, medium, high} and topology ~ {linear,
ring, grid} (uniform, seeded per instance). The original cache used one fixed
medium/ring setting, which gives a noise-aware model nothing to learn from.

The sampled preset/topology are stored on the instance (noise_level,
topology) for analysis; the noise actually simulated is always derived from
the graph's own numbers (noise_model.noise_from_hardware_graph), so input
features and simulated noise cannot disagree.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch

from src.data.build_dataset import DATASET_DIR
from src.graphs.hardware_graph import build_hardware_graph
from src.quantum.run_baseline_batch import REPO_ROOT

NOISY_DATASET_DIR = REPO_ROOT / "data_cache_noisy"
LEVELS = ["low", "medium", "high"]
TOPOLOGIES = ["linear", "ring", "grid"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(DATASET_DIR))
    ap.add_argument("--out", default=str(NOISY_DATASET_DIR))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = {}
    for split_i, split in enumerate(["train", "val", "test"]):
        data = torch.load(Path(args.src) / f"{split}.pt", weights_only=False)
        rng = random.Random(f"{args.seed}/{split}")
        counts = {}
        for i, ci in enumerate(data):
            level, topo = rng.choice(LEVELS), rng.choice(TOPOLOGIES)
            ci.hardware_graph = build_hardware_graph(
                ci.n_qubits, topology=topo, noise_level=level, seed=args.seed * 1_000_003 + split_i * 100_000 + i
            )
            ci.noise_level, ci.topology = level, topo
            counts[f"{level}/{topo}"] = counts.get(f"{level}/{topo}", 0) + 1
        torch.save(data, out / f"{split}.pt")
        summary[split] = dict(sorted(counts.items()))
        print(f"{split}: {len(data)} instances  {summary[split]}")
    with open(out / "config.json", "w") as f:
        json.dump({"src": str(args.src), "seed": args.seed, "levels": LEVELS, "topologies": TOPOLOGIES, "counts": summary}, f, indent=2)


if __name__ == "__main__":
    main()
