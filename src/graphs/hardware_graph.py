"""Builds the hardware/noise graph (G_D, N) (Section 5.1 / 7.2 of the project docs).

None of the VQEzy instances carry real device/noise metadata, so this is the
"Plan A" synthetic augmentation the dataset-strategy doc calls for: hardware
topology and noise level are sampled/parameterized independently of the
Hamiltonian, letting later experiments sweep them for the noise-robustness
and generalization study without needing real backend calibration data.

  Nodes = physical qubits.
  Node feat = [readout_error, T1_normalized, single_qubit_gate_error]
  Edges = coupling map, depends on `topology`:
          linear: i -- i+1
          ring:   linear + wraparound
          grid:   2D nearest-neighbour lattice (n_qubits factored as close to
                  square as possible)
  Edge feat = [two_qubit_gate_error]

Noise level presets are loosely calibrated to typical superconducting-qubit
NISQ error magnitudes (order-of-magnitude only, not any specific backend).
"""
from __future__ import annotations

import math

import numpy as np
import torch
from torch_geometric.data import Data

_NOISE_PRESETS = {
    # (readout_err, single_qubit_err, two_qubit_err, T1_us) base values
    "low": (0.01, 0.0005, 0.003, 150.0),
    "medium": (0.025, 0.002, 0.01, 80.0),
    "high": (0.06, 0.006, 0.03, 30.0),
}


def _coupling_pairs(n_qubits: int, topology: str) -> list[tuple[int, int]]:
    if topology == "linear":
        return [(i, i + 1) for i in range(n_qubits - 1)]
    if topology == "ring":
        pairs = [(i, i + 1) for i in range(n_qubits - 1)]
        if n_qubits > 2:
            pairs.append((n_qubits - 1, 0))
        return pairs
    if topology == "grid":
        rows = int(math.floor(math.sqrt(n_qubits)))
        rows = max(rows, 1)
        while n_qubits % rows != 0:
            rows -= 1
        cols = n_qubits // rows
        pairs = []
        for r in range(rows):
            for c in range(cols):
                idx = r * cols + c
                if c + 1 < cols:
                    pairs.append((idx, idx + 1))
                if r + 1 < rows:
                    pairs.append((idx, idx + cols))
        return pairs
    raise ValueError(f"Unknown topology: {topology}")


def build_hardware_graph(
    n_qubits: int,
    topology: str = "linear",
    noise_level: str | float = "medium",
    seed: int | None = None,
) -> Data:
    rng = np.random.default_rng(seed)

    if isinstance(noise_level, str):
        readout0, sq0, tq0, t1_0 = _NOISE_PRESETS[noise_level]
    else:
        # continuous scalar in [0, 1]: interpolate low -> high
        lo = np.array(_NOISE_PRESETS["low"])
        hi = np.array(_NOISE_PRESETS["high"])
        readout0, sq0, tq0, t1_0 = lo + noise_level * (hi - lo)

    # +/-20% per-qubit device variability, matching real hardware heterogeneity.
    jitter = lambda base: base * rng.uniform(0.8, 1.2, size=n_qubits)
    readout = jitter(readout0)
    single_q_err = jitter(sq0)
    t1 = jitter(t1_0)
    t1_norm = t1 / _NOISE_PRESETS["low"][3]  # normalize against the "low noise" T1

    x = torch.tensor(
        np.stack([readout, t1_norm, single_q_err], axis=1), dtype=torch.float32
    )

    pairs = _coupling_pairs(n_qubits, topology)
    edge_index, edge_attr = [], []
    for (a, b) in pairs:
        tq_err = float(tq0 * rng.uniform(0.8, 1.2))
        edge_index += [[a, b], [b, a]]
        edge_attr += [[tq_err], [tq_err]]

    edge_index_t = (
        torch.tensor(edge_index, dtype=torch.long).T
        if edge_index
        else torch.zeros((2, 0), dtype=torch.long)
    )
    edge_attr_t = (
        torch.tensor(edge_attr, dtype=torch.float32)
        if edge_attr
        else torch.zeros((0, 1), dtype=torch.float32)
    )

    data = Data(x=x, edge_index=edge_index_t, edge_attr=edge_attr_t)
    data.topology = topology
    data.noise_level = noise_level
    return data
