"""Builds the ansatz/circuit graph G_A for the CZRXRY ansatz (src/quantum/ansatz.py).

  Nodes = (qubit, layer) gate slots -- each node owns one RX and one RY
          rotation, i.e. two entries of the flattened parameter vector
          `params[layer, qubit, 0:2]` used by czrxry_ansatz().
  Edges = "entangling": the CZ ring within a layer (i -> i+1, wrap n-1 -> 0)
          "temporal":   same qubit across consecutive layers (parameter/
                        structural dependency a layer-wise decoder can use)

Node feat = [qubit_idx/n_qubits, layer_idx/n_layers, rx_param_index, ry_param_index]
Edge feat = one-hot [is_entangling, is_temporal]
"""
from __future__ import annotations

import torch
from torch_geometric.data import Data


def build_ansatz_graph(n_qubits: int, n_layers: int) -> Data:
    nodes = [(q, l) for l in range(n_layers) for q in range(n_qubits)]
    node_idx = {n: i for i, n in enumerate(nodes)}

    x = torch.zeros(len(nodes), 4)
    for (q, l), i in node_idx.items():
        rx_index = l * n_qubits * 2 + q * 2 + 0
        ry_index = l * n_qubits * 2 + q * 2 + 1
        x[i] = torch.tensor(
            [q / max(n_qubits - 1, 1), l / max(n_layers - 1, 1), rx_index, ry_index]
        )

    edge_index, edge_attr = [], []

    def add_edge(u, v, is_entangling):
        edge_index.append([node_idx[u], node_idx[v]])
        edge_attr.append([1.0, 0.0] if is_entangling else [0.0, 1.0])

    for l in range(n_layers):
        # entangling CZ ring within the layer (undirected -> both directions)
        for q in range(n_qubits - 1):
            add_edge((q, l), (q + 1, l), is_entangling=True)
            add_edge((q + 1, l), (q, l), is_entangling=True)
        if n_qubits > 2:
            add_edge((n_qubits - 1, l), (0, l), is_entangling=True)
            add_edge((0, l), (n_qubits - 1, l), is_entangling=True)
        # temporal edge: same qubit, consecutive layers
        if l > 0:
            for q in range(n_qubits):
                add_edge((q, l - 1), (q, l), is_entangling=False)
                add_edge((q, l), (q, l - 1), is_entangling=False)

    edge_index_t = (
        torch.tensor(edge_index, dtype=torch.long).T
        if edge_index
        else torch.zeros((2, 0), dtype=torch.long)
    )
    edge_attr_t = (
        torch.tensor(edge_attr, dtype=torch.float32)
        if edge_attr
        else torch.zeros((0, 2), dtype=torch.float32)
    )

    data = Data(x=x, edge_index=edge_index_t, edge_attr=edge_attr_t)
    data.n_params = n_layers * n_qubits * 2
    return data
