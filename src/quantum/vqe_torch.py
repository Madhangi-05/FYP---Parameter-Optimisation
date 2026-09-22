"""Differentiable VQE energy: a torch-interface PennyLane QNode so E(theta0)
is a plain torch scalar that autograd can backprop through, all the way into
the GNN that produced theta0. This is what makes the "quantum-in-the-loop"
training objective (Eq. 5.3, energy term) possible without ever needing
pre-optimized parameter labels.
"""
from __future__ import annotations

import pennylane as qml

from src.quantum.ansatz import czrxry_ansatz


def make_torch_vqe(hamiltonian, n_qubits: int, n_layers: int, dev=None, diff_method: str = "best"):
    """diff_method="best" (adjoint, on default.qubit) is fine for a single
    backward pass -- e.g. the direct E(theta0) loss, or evaluation. It does
    NOT reliably support a second backward pass (needed to unroll optimizer
    steps inside training, see train_phase2.py's --unroll-steps), so that
    path explicitly requests diff_method="parameter-shift" instead: the
    parameter-shift rule is itself just a combination of ordinary forward
    circuit evaluations, so it stays differentiable under create_graph=True.
    """
    if dev is None:
        dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="torch", diff_method=diff_method)
    def circuit(theta):
        czrxry_ansatz(theta, n_qubits, n_layers)
        return qml.expval(hamiltonian)

    return circuit
