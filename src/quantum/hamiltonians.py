"""Reconstructs PennyLane Hamiltonians for the VQEzy qmanybody families.

Mirrors the generation code in external/VQEzy/data_sampling.py exactly, so a
Hamiltonian rebuilt here from an instance's stored coupling constants matches
the one VQEzy used to produce its reference loss_history / opt_params.
"""
from __future__ import annotations

import numpy as np
import pennylane as qml

from src.data.vqezy_loader import VQEzyInstance

# Exact diagonalization is O(2^(3n)); 12 qubits (4096x4096) is a couple of
# seconds, 14 is tens of seconds, 16+ becomes impractical on a laptop CPU.
EXACT_DIAG_MAX_QUBITS = 12


def true_ground_energy(hamiltonian, n_qubits: int) -> float | None:
    """Exact ground-state energy via full diagonalization, when feasible.

    Needed because VQEzy's own stored reference optimum (loss_history[-1])
    is NOT always well-converged -- e.g. fh_8_qubit/sample_0's reference is
    -5.26 while the true ground energy is -16.25 (their 2000-step Adam run
    got stuck). Using an under-converged reference as an optimization target
    silently makes "steps to threshold" comparisons meaningless.
    """
    if n_qubits > EXACT_DIAG_MAX_QUBITS:
        return None
    mat = qml.matrix(hamiltonian, wire_order=range(n_qubits))
    return float(np.linalg.eigvalsh(mat)[0].real)


def build_hamiltonian(instance: VQEzyInstance):
    """Returns (hamiltonian, n_qubits) for a VQEzy qmanybody instance."""
    if instance.family == "xyz":
        n_qubits = instance.n_qubits
        J = instance.coupling  # (J1, J2, J3)
        h = qml.spin.heisenberg("chain", [n_qubits], coupling=J)
        return h, n_qubits

    if instance.family == "fh":
        n_qubits = instance.n_qubits
        t, U = instance.coupling
        n_cells = [n_qubits // 2]
        h = qml.spin.fermi_hubbard(
            "chain", n_cells, hopping=float(t), coulomb=float(U), mapping="jordan_wigner"
        )
        return h, n_qubits

    if instance.family == "ti":
        # VQEzy fixes the TI lattice at cell_len=4, cell_wid=2 -> 8 qubits.
        cell_len, cell_wid = 4, 2
        n_qubits = cell_len * cell_wid
        j, hf = instance.coupling
        h = qml.spin.transverse_ising(
            "rectangle", [cell_len, cell_wid], coupling=float(j), h=float(hf)
        )
        return h, n_qubits

    raise ValueError(f"Unknown family: {instance.family}")
