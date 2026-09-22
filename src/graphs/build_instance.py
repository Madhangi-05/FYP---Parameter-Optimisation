"""Ties the three graph builders together into one training instance
(G_H, G_A, G_D) for a VQEzy sample, matching the f_theta(G_H, G_A, G_D, N)
formulation in the project docs.
"""
from __future__ import annotations

from dataclasses import dataclass

from torch_geometric.data import Data

from src.data.vqezy_loader import VQEzyInstance
from src.graphs.ansatz_graph import build_ansatz_graph
from src.graphs.hamiltonian_graph import build_hamiltonian_graph
from src.graphs.hardware_graph import build_hardware_graph
from src.quantum.hamiltonians import build_hamiltonian


@dataclass
class GraphInstance:
    hamiltonian_graph: Data
    ansatz_graph: Data
    hardware_graph: Data
    n_qubits: int
    n_layers: int
    source: VQEzyInstance


def build_graph_instance(
    instance: VQEzyInstance,
    n_layers: int = 2,
    topology: str = "linear",
    noise_level: str | float = "medium",
    seed: int | None = None,
) -> GraphInstance:
    hamiltonian, n_qubits = build_hamiltonian(instance)
    g_h = build_hamiltonian_graph(hamiltonian, n_qubits)
    g_a = build_ansatz_graph(n_qubits, n_layers)
    g_d = build_hardware_graph(n_qubits, topology=topology, noise_level=noise_level, seed=seed)
    return GraphInstance(
        hamiltonian_graph=g_h,
        ansatz_graph=g_a,
        hardware_graph=g_d,
        n_qubits=n_qubits,
        n_layers=n_layers,
        source=instance,
    )
