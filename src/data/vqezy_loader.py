"""Loader for the qmanybody split of the VQEzy dataset (xyz / fh / ti .h5 files).

Each .h5 file has four top-level groups (coupling_const|t_U|j_h, loss_history,
n_qubits, opt_params), each containing one dataset per sample, keyed
"sample_<i>". This loader reads those groups directly with h5py -- no
TorchQuantum / uv environment from the original VQEzy repo is required.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

# Which coupling group name each qmanybody file family uses.
_COUPLING_GROUP = {
    "xyz": "coupling_const",
    "fh": "t_U",
    "ti": "j_h",
}


def _family_from_filename(path: Path) -> str:
    name = path.stem
    for family in _COUPLING_GROUP:
        if name.startswith(family):
            return family
    raise ValueError(f"Cannot infer family (xyz/fh/ti) from filename: {path.name}")


@dataclass
class VQEzyInstance:
    family: str          # "xyz" | "fh" | "ti"
    sample_key: str
    n_qubits: int
    coupling: np.ndarray          # J1,J2,J3 (xyz) | t,U (fh) | j,h (ti)
    reference_loss_history: np.ndarray
    reference_opt_params: np.ndarray


def iter_sample_keys(h5_path: str | Path, limit: int | None = None):
    with h5py.File(h5_path, "r") as f:
        keys = list(f["n_qubits"].keys())
    if limit is not None:
        keys = keys[:limit]
    return keys


def load_instance(h5_path: str | Path, sample_key: str) -> VQEzyInstance:
    h5_path = Path(h5_path)
    family = _family_from_filename(h5_path)
    coupling_group = _COUPLING_GROUP[family]
    with h5py.File(h5_path, "r") as f:
        n_qubits = int(f["n_qubits"][sample_key][()])
        coupling = np.array(f[coupling_group][sample_key][()])
        loss_history = np.array(f["loss_history"][sample_key][()])
        opt_params = np.array(f["opt_params"][sample_key][()])
    return VQEzyInstance(
        family=family,
        sample_key=sample_key,
        n_qubits=n_qubits,
        coupling=coupling,
        reference_loss_history=loss_history,
        reference_opt_params=opt_params,
    )
