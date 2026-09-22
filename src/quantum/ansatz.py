"""Hardware-efficient CZRXRY ansatz, matching external/VQEzy/ansatz.py.

Reusing VQEzy's own ansatz (rather than inventing a new one) means the
Hamiltonian graph / ansatz graph / circuits we build in later phases stay
directly comparable to the dataset's reference trajectories.
"""
from __future__ import annotations

import pennylane as qml


def czrxry_layer(params, n_qubits):
    """One CZ-ring + RX/RY layer. params shape: (n_qubits, 2)."""
    for i in range(n_qubits - 1):
        qml.CZ(wires=[i, i + 1])
    qml.CZ(wires=[n_qubits - 1, 0])

    for i in range(n_qubits):
        qml.RX(params[i, 0], wires=i)
        qml.RY(params[i, 1], wires=i)


def czrxry_ansatz(params, n_qubits, n_layers):
    """params shape: (n_layers, n_qubits, 2)."""
    for l in range(n_layers):
        czrxry_layer(params[l], n_qubits)


def n_params(n_qubits: int, n_layers: int) -> int:
    return n_layers * n_qubits * 2
