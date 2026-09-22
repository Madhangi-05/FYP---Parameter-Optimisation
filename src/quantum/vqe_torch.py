"""Differentiable VQE energy: a torch-interface PennyLane QNode so E(theta0)
is a plain torch scalar that autograd can backprop through, all the way into
the GNN that produced theta0. This is what makes the "quantum-in-the-loop"
training objective (Eq. 5.3, energy term) possible without ever needing
pre-optimized parameter labels.
"""
from __future__ import annotations

import pennylane as qml

from src.quantum.ansatz import czrxry_ansatz


def make_torch_vqe(hamiltonian, n_qubits: int, n_layers: int, dev=None):
    if dev is None:
        dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="torch", diff_method="best")
    def circuit(theta):
        czrxry_ansatz(theta, n_qubits, n_layers)
        return qml.expval(hamiltonian)

    return circuit
