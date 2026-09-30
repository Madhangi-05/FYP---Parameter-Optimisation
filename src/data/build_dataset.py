"""Stage 1: extract (G_H, G_A, G_D) for a fixed pool of VQEzy instances once,
and write a fixed train/val/test split -- so graph construction (and, for
val/test instances, exact-diagonalization ground energy) happens a single
time, not on every training run.

Run this once (on CPU is fine -- it's cheap, see timing note below), then
train_gpu.py just loads the cached tensors.

Notes:
  - Training instances do NOT get ground_energy computed (it's not used by
    the training loss -- the live circuit energy is) -- only val/test do,
    since they need it to score how good an initialization is.
  - The Hamiltonian object itself (a PennyLane operator, not a tensor) is
    NOT cached -- only h5_rel/sample are, so train_gpu.py can cheaply
    rebuild it via build_hamiltonian() when it's actually needed for a
    circuit. That reconstruction is fast (reads one small h5 entry + builds
    a qml.Hamiltonian from a formula); the graphs are what's expensive
    enough to be worth caching.
  - xyz_12_qubit is reserved entirely for test, as the one genuine
    unseen-qubit-count generalization check (all training instances stay
    <=8 qubits).
"""
from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from torch_geometric.data import Data

from src.data.vqezy_loader import iter_sample_keys, load_instance
from src.graphs.build_instance import build_graph_instance
from src.quantum.hamiltonians import build_hamiltonian, true_ground_energy
from src.quantum.run_baseline_batch import REPO_ROOT

TRAIN_FILES = [
    "external/VQEzy/qmanybody/xyz_4_qubit.h5",
    "external/VQEzy/qmanybody/fh_4_qubit.h5",
    "external/VQEzy/qmanybody/fh_6_qubit.h5",
    "external/VQEzy/qmanybody/fh_8_qubit.h5",
    "external/VQEzy/qmanybody/ti_8_qubit.h5",
]
UNSEEN_SIZE_FILE = "external/VQEzy/qmanybody/xyz_12_qubit.h5"

DATASET_DIR = REPO_ROOT / "data_cache"


@dataclass
class CachedInstance:
    family: str
    sample: str
    h5_rel: str  # so train_gpu.py can cheaply rebuild the Hamiltonian
    n_qubits: int
    n_layers: int
    hamiltonian_graph: Data
    ansatz_graph: Data
    hardware_graph: Data
    ground_energy: float | None  # only set for val/test
    unseen_qubit_count: bool = False


def _extract(h5_rel, sample, n_layers, topology, noise_level, seed, need_ground) -> CachedInstance:
    instance = load_instance(REPO_ROOT / h5_rel, sample)
    gi = build_graph_instance(
        instance, n_layers=n_layers, topology=topology, noise_level=noise_level, seed=seed
    )
    ground = None
    if need_ground:
        hamiltonian, n_qubits = build_hamiltonian(instance)
        ground = true_ground_energy(hamiltonian, n_qubits)
    return CachedInstance(
        family=instance.family,
        sample=sample,
        h5_rel=h5_rel,
        n_qubits=gi.n_qubits,
        n_layers=gi.n_layers,
        hamiltonian_graph=gi.hamiltonian_graph,
        ansatz_graph=gi.ansatz_graph,
        hardware_graph=gi.hardware_graph,
        ground_energy=ground,
    )


def build_and_cache(
    train_per_file: int,
    val_per_file: int,
    test_per_file: int,
    unseen_size_count: int,
    n_layers: int,
    topology: str,
    noise_level: str,
    seed: int,
    out_dir: Path = DATASET_DIR,
):
    rng = random.Random(seed)
    manifest: dict[str, list] = {"train": [], "val": [], "test": []}
    cached: dict[str, list] = {"train": [], "val": [], "test": []}

    t0 = time.perf_counter()
    for h5_rel in TRAIN_FILES:
        keys = iter_sample_keys(REPO_ROOT / h5_rel)
        rng.shuffle(keys)
        splits = {
            "train": keys[:train_per_file],
            "val": keys[train_per_file : train_per_file + val_per_file],
            "test": keys[
                train_per_file + val_per_file : train_per_file + val_per_file + test_per_file
            ],
        }
        for split_name, sample_keys in splits.items():
            need_ground = split_name != "train"
            for sample in sample_keys:
                inst_seed = hash((h5_rel, sample)) % (2**31)
                ci = _extract(h5_rel, sample, n_layers, topology, noise_level, inst_seed, need_ground)
                cached[split_name].append(ci)
                manifest[split_name].append(
                    {"h5": h5_rel, "sample": sample, "family": ci.family, "n_qubits": ci.n_qubits}
                )
        print(f"  {h5_rel}: train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])}")

    keys = iter_sample_keys(REPO_ROOT / UNSEEN_SIZE_FILE)
    rng.shuffle(keys)
    for sample in keys[:unseen_size_count]:
        inst_seed = hash((UNSEEN_SIZE_FILE, sample)) % (2**31)
        ci = _extract(UNSEEN_SIZE_FILE, sample, n_layers, topology, noise_level, inst_seed, need_ground=True)
        ci.unseen_qubit_count = True
        cached["test"].append(ci)
        manifest["test"].append(
            {
                "h5": UNSEEN_SIZE_FILE,
                "sample": sample,
                "family": ci.family,
                "n_qubits": ci.n_qubits,
                "unseen_qubit_count": True,
            }
        )
    print(f"  {UNSEEN_SIZE_FILE} (reserved, test-only): {unseen_size_count} instances")

    out_dir.mkdir(exist_ok=True, parents=True)
    for split_name in ["train", "val", "test"]:
        torch.save(cached[split_name], out_dir / f"{split_name}.pt")
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    with open(out_dir / "config.json", "w") as f:
        json.dump(
            {
                "train_per_file": train_per_file,
                "val_per_file": val_per_file,
                "test_per_file": test_per_file,
                "unseen_size_count": unseen_size_count,
                "n_layers": n_layers,
                "topology": topology,
                "noise_level": noise_level,
                "seed": seed,
            },
            f,
            indent=2,
        )

    elapsed = time.perf_counter() - t0
    print(
        f"\nTotal: train={len(cached['train'])}  val={len(cached['val'])}  test={len(cached['test'])}  "
        f"({elapsed:.1f}s)"
    )
    print(f"Saved -> {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-per-file", type=int, default=200)
    ap.add_argument("--val-per-file", type=int, default=30)
    ap.add_argument("--test-per-file", type=int, default=30)
    ap.add_argument("--unseen-size-count", type=int, default=40)
    ap.add_argument("--n-layers", type=int, default=2)
    ap.add_argument("--topology", default="ring")
    ap.add_argument("--noise-level", default="medium")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    build_and_cache(
        args.train_per_file,
        args.val_per_file,
        args.test_per_file,
        args.unseen_size_count,
        args.n_layers,
        args.topology,
        args.noise_level,
        args.seed,
    )


if __name__ == "__main__":
    main()
