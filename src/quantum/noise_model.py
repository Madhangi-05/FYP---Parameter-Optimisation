"""Noise model driven by the hardware/noise graph G_D -- so the numbers the
GNN sees as input are exactly the numbers that corrupt the circuit.

Per instance, read from G_D (src/graphs/hardware_graph.py):
  node feat [readout_err, T1/150us, single_qubit_err], edge feat [two_qubit_err],
  edges = physical coupling map (linear / ring / grid).

Channels inserted into czrxry_ansatz (same order in the PennyLane circuit
below and in torch_sim.py -- validate_torch_sim.py checks they agree):
  - after every CZ(a,b): DepolarizingChannel(p_cz) on a and on b. If (a,b)
    is not a physical edge, the CZ needs routing: each extra hop costs a SWAP
    there and back (2 x 3 CNOT-equivalents), so
        p_cz = 1 - (1 - p_edge_mean)^(1 + 6*(d-1)),  d = hop distance.
    This is what makes topology physically matter (the ansatz always uses a
    CZ *ring*, so a linear chain pays heavily for the wrap-around CZ).
  - after every RX / RY on qubit q: DepolarizingChannel(single_qubit_err_q).
  - end of each ansatz layer: AmplitudeDamping(gamma_q),
        gamma_q = 1 - exp(-t_layer / T1_q),
    t_layer = CZ rounds * 300ns + 2 * 35ns (order-of-magnitude superconducting
    gate times; 2 parallel CZ rounds cover an even ring, 3 an odd one).
  - before measurement: DepolarizingChannel(1.5 * readout_err_q). For
    expectation values of Pauli strings this is EXACTLY a symmetric classical
    readout flip with probability r: both scale each Pauli factor on q by
    (1 - 2r) = (1 - 4p/3). (A BitFlip channel would not be: it leaves X-basis
    measurements untouched.)
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import networkx as nx
import numpy as np
import pennylane as qml

T_1Q_US = 0.035
T_2Q_US = 0.300
T1_REF_US = 150.0  # hardware_graph.py normalizes T1 by the "low" preset's 150us
MAX_DEPOL = 0.75  # fully depolarizing; probabilities above this are not channels we want


@dataclass
class NoiseParams:
    cz_pairs: list  # ansatz CZ-ring pairs, in circuit order
    cz_p: np.ndarray  # (n_cz,) depolarizing prob applied to BOTH qubits after that CZ
    sq_p: np.ndarray  # (n,) depolarizing prob after each RX / RY
    gamma: np.ndarray  # (n,) amplitude damping per ansatz layer
    readout_p: np.ndarray  # (n,) final depolarizing prob equivalent to readout error


def ansatz_cz_pairs(n_qubits: int) -> list:
    # Same pairs and order as czrxry_layer in src/quantum/ansatz.py.
    return [(i, i + 1) for i in range(n_qubits - 1)] + [(n_qubits - 1, 0)]


def noise_from_hardware_graph(hw, n_qubits: int) -> NoiseParams:
    x = hw.x.detach().cpu().numpy().astype(np.float64)
    readout, t1, sq = x[:, 0], x[:, 1] * T1_REF_US, x[:, 2]

    g = nx.Graph()
    g.add_nodes_from(range(n_qubits))
    ei = hw.edge_index.detach().cpu().numpy()
    ea = hw.edge_attr.detach().cpu().numpy()[:, 0]
    for (a, b), p in zip(ei.T, ea):
        g.add_edge(int(a), int(b), p=float(p))

    pairs = ansatz_cz_pairs(n_qubits)
    cz_p = []
    for a, b in pairs:
        path = nx.shortest_path(g, a, b)
        d = len(path) - 1
        p_mean = float(np.mean([g.edges[u, v]["p"] for u, v in zip(path[:-1], path[1:])]))
        cz_p.append(min(MAX_DEPOL, 1.0 - (1.0 - p_mean) ** (1 + 6 * (d - 1))))

    rounds = 2 if n_qubits % 2 == 0 else 3
    t_layer = rounds * T_2Q_US + 2 * T_1Q_US
    gamma = 1.0 - np.exp(-t_layer / t1)
    return NoiseParams(
        cz_pairs=pairs,
        cz_p=np.array(cz_p),
        sq_p=np.minimum(sq, MAX_DEPOL),
        gamma=gamma,
        readout_p=np.minimum(1.5 * readout, MAX_DEPOL),
    )


def noisy_czrxry_ansatz(params, n_qubits: int, n_layers: int, noise: NoiseParams):
    """PennyLane reference implementation (needs a mixed-state device, e.g.
    default.mixed). Used to validate torch_sim.py, not for bulk runs."""
    for l in range(n_layers):
        for (a, b), p in zip(noise.cz_pairs, noise.cz_p):
            qml.CZ(wires=[a, b])
            qml.DepolarizingChannel(p, wires=a)
            qml.DepolarizingChannel(p, wires=b)
        for i in range(n_qubits):
            qml.RX(params[l, i, 0], wires=i)
            qml.DepolarizingChannel(noise.sq_p[i], wires=i)
            qml.RY(params[l, i, 1], wires=i)
            qml.DepolarizingChannel(noise.sq_p[i], wires=i)
        for i in range(n_qubits):
            qml.AmplitudeDamping(noise.gamma[i], wires=i)
    for i in range(n_qubits):
        qml.DepolarizingChannel(noise.readout_p[i], wires=i)
