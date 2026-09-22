"""Dumps a human-readable view of a VQEzy .h5 file: overall structure +
full contents of a few concrete samples. .h5 is a binary format, so this
is the "let me actually look at the data" counterpart to vqezy_loader.py.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]


def dump(h5_path: str, n_samples: int, out_path: str):
    lines = []
    with h5py.File(h5_path, "r") as f:
        groups = list(f.keys())
        n_total = len(f[groups[0]].keys())
        lines.append(f"File: {h5_path}")
        lines.append(f"Top-level groups: {groups}")
        lines.append(f"Total samples in file: {n_total}")
        lines.append("")

        sample_keys = list(f[groups[0]].keys())[:n_samples]
        for key in sample_keys:
            lines.append(f"=== {key} ===")
            for g in groups:
                val = f[g][key][()]
                if isinstance(val, np.ndarray) and val.size > 20:
                    lines.append(f"{g}: shape={val.shape} dtype={val.dtype}")
                    lines.append(f"  first 5: {val.flat[:5]}")
                    lines.append(f"  last 5:  {val.flat[-5:]}")
                else:
                    lines.append(f"{g}: {val}")
            lines.append("")

    out_text = "\n".join(lines)
    Path(out_path).write_text(out_text, encoding="utf-8")
    print(out_text)
    print(f"\nSaved -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", default=str(REPO_ROOT / "external/VQEzy/qmanybody/xyz_4_qubit.h5"))
    ap.add_argument("--n-samples", type=int, default=3)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out_path = args.out or str(
        REPO_ROOT / "results" / f"h5_dump_{Path(args.h5).stem}.txt"
    )
    dump(args.h5, args.n_samples, out_path)


if __name__ == "__main__":
    main()
