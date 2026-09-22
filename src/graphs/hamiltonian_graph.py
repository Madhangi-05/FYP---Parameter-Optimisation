"""Builds the Hamiltonian graph G_H (Section 5.1 / 7.2 of the project docs).

Design (qubit-count-scaling, unlike Qracle's 2^n x 2^n eigen-matrix graph --
see external/Qracle/qracle/datasets.py, which diagonalizes the full
Hamiltonian and is therefore exponential in n_qubits):

  Nodes   = qubits (0 .. n_qubits-1).
  Edges   = one per pair of qubits that co-occur in some Hamiltonian term
            (multi-body terms, e.g. the Y·Z·Y Jordan-Wigner strings in the
            Fermi-Hubbard Hamiltonian, contribute an edge for every pair in
            the term -- a clique expansion).
  Node feat  = [onsite_X, onsite_Y, onsite_Z coefficient sums, degree, i/n_qubits]
  Edge feat  = [|coeff| sum on this pair, X-count, Y-count, Z-count, max interaction order]
  Global     = [identity coefficient (constant energy offset), n_qubits, n_terms]

Pauli terms come from PennyLane's op_math tree (Sum of SProd(coeff, Prod(Pauli...))
or SProd(coeff, single Pauli) or SProd(coeff, Identity)) -- see the structure
printed by hamiltonians.build_hamiltonian() for the VQEzy qmanybody families.
"""
from __future__ import annotations

from collections import defaultdict
from itertools import combinations

import torch
from torch_geometric.data import Data

_PAULI_INDEX = {"PauliX": 0, "PauliY": 1, "PauliZ": 2}


def _iter_terms(hamiltonian):
    """Yields (coeff: float, wires: list[int], pauli_names: list[str]) per term.
    Identity terms yield wires=[] (handled separately as a global offset)."""
    operands = hamiltonian.operands if hasattr(hamiltonian, "operands") else [hamiltonian]
    for term in operands:
        coeff = float(term.scalar) if hasattr(term, "scalar") else 1.0
        base = term.base if hasattr(term, "base") else term
        sub_ops = base.operands if hasattr(base, "operands") else [base]
        wires, names = [], []
        for op in sub_ops:
            if op.name == "Identity":
                continue
            wires.append(int(list(op.wires)[0]))
            names.append(op.name)
        yield coeff, wires, names


def build_hamiltonian_graph(hamiltonian, n_qubits: int) -> Data:
    onsite = torch.zeros(n_qubits, 3)  # per-qubit sum of X/Y/Z coeffs from 1-body terms
    pair_weight = defaultdict(float)  # (i,j) -> summed |coeff|
    pair_pauli_count = defaultdict(lambda: [0, 0, 0])  # (i,j) -> [X,Y,Z] occurrence counts
    pair_order = defaultdict(int)  # (i,j) -> max arity of contributing terms
    identity_coeff = 0.0
    n_terms = 0

    for coeff, wires, names in _iter_terms(hamiltonian):
        n_terms += 1
        if len(wires) == 0:
            identity_coeff += coeff
            continue
        if len(wires) == 1:
            onsite[wires[0], _PAULI_INDEX[names[0]]] += coeff
            continue
        # Multi-body term -> clique expansion over all pairs in the term.
        for (a, na), (b, nb) in combinations(zip(wires, names), 2):
            key = (a, b) if a < b else (b, a)
            pair_weight[key] += abs(coeff)
            pair_pauli_count[key][_PAULI_INDEX[na]] += 1
            pair_pauli_count[key][_PAULI_INDEX[nb]] += 1
            pair_order[key] = max(pair_order[key], len(wires))

    degree = torch.zeros(n_qubits)
    for (a, b) in pair_weight:
        degree[a] += 1
        degree[b] += 1

    node_idx = torch.arange(n_qubits, dtype=torch.float32) / max(n_qubits - 1, 1)
    x = torch.cat([onsite, degree.unsqueeze(1), node_idx.unsqueeze(1)], dim=1)

    edge_index, edge_attr = [], []
    for (a, b), w in pair_weight.items():
        counts = pair_pauli_count[(a, b)]
        order = pair_order[(a, b)]
        feat = [w] + counts + [float(order)]
        # undirected -> both directions, matching PyG convention
        edge_index += [[a, b], [b, a]]
        edge_attr += [feat, feat]

    edge_index_t = (
        torch.tensor(edge_index, dtype=torch.long).T
        if edge_index
        else torch.zeros((2, 0), dtype=torch.long)
    )
    edge_attr_t = (
        torch.tensor(edge_attr, dtype=torch.float32)
        if edge_attr
        else torch.zeros((0, 5), dtype=torch.float32)
    )

    data = Data(x=x, edge_index=edge_index_t, edge_attr=edge_attr_t)
    data.global_features = torch.tensor(
        [[identity_coeff, float(n_qubits), float(n_terms)]], dtype=torch.float32
    )
    return data
