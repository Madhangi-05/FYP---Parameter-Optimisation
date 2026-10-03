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


def build_hamiltonian_from_params(family: str, coupling, n_qubits: int):
    """Returns (hamiltonian, n_qubits). Takes just the 3 plain numbers/arrays
    actually needed (family, coupling, n_qubits) rather than a full
    VQEzyInstance, so this can be called WITHOUT ever touching the raw .h5
    file -- e.g. from a cached dataset (build_dataset.py / train_gpu.py)
    that only stored these, on a machine that never cloned external/VQEzy at
    all. build_hamiltonian() below is a thin wrapper for call sites that
    already have the full loaded instance.
    """
    if family == "xyz":
        J = coupling  # (J1, J2, J3)
        h = qml.spin.heisenberg("chain", [n_qubits], coupling=J)
        return h, n_qubits

    if family == "fh":
        t, U = coupling
        n_cells = [n_qubits // 2]
        h = qml.spin.fermi_hubbard(
            "chain", n_cells, hopping=float(t), coulomb=float(U), mapping="jordan_wigner"
        )
        return h, n_qubits

    if family == "ti":
        # VQEzy fixes the TI lattice at cell_len=4, cell_wid=2 -> 8 qubits.
        cell_len, cell_wid = 4, 2
        j, hf = coupling
        h = qml.spin.transverse_ising(
            "rectangle", [cell_len, cell_wid], coupling=float(j), h=float(hf)
        )
        return h, n_qubits

    raise ValueError(f"Unknown family: {family}")


def build_hamiltonian(instance: VQEzyInstance):
    """Returns (hamiltonian, n_qubits) for a VQEzy qmanybody instance."""
    return build_hamiltonian_from_params(instance.family, instance.coupling, instance.n_qubits)
