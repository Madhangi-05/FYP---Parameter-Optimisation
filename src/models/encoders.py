"""GNN encoders for G_H, G_A, G_D (Section 5.1: "Hamiltonian Encoder",
"Ansatz Encoder", "Hardware/Noise Encoder").

Same architecture class for all three -- they differ only in input/edge
feature dimensions, which are fixed by the graph builders:
  Hamiltonian graph : node_dim=5, edge_dim=5  (src/graphs/hamiltonian_graph.py)
  Ansatz graph      : node_dim=4, edge_dim=2  (src/graphs/ansatz_graph.py)
  Hardware graph    : node_dim=3, edge_dim=1  (src/graphs/hardware_graph.py)

GATConv's edge_dim lets edge features (coupling weight, Pauli type, two-qubit
gate error, ...) directly influence attention -- this is how noise/coupling
strength actually reaches the node embeddings, not just node identity.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.nn import GATConv


class GraphEncoder(nn.Module):
    def __init__(self, in_dim: int, edge_dim: int, hidden: int = 64, layers: int = 3, heads: int = 2):
        super().__init__()
        assert hidden % heads == 0
        self.input_proj = nn.Linear(in_dim, hidden)
        self.convs = nn.ModuleList(
            [
                GATConv(hidden, hidden // heads, heads=heads, edge_dim=edge_dim, concat=True)
                for _ in range(layers)
            ]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(layers)])

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        for conv, norm in zip(self.convs, self.norms):
            h = norm(h + conv(h, edge_index, edge_attr).relu())  # residual + norm
        return h  # (n_nodes, hidden)
