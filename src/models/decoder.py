"""Layer-wise parameter decoder (Section 5.1): maps each fused ansatz-node
embedding to that node's 2 parameters (RX, RY angle). Node ordering is fixed
by ansatz_graph.build_ansatz_graph as [(q,l) for l in range(n_layers) for q
in range(n_qubits)], so a straight reshape recovers czrxry_ansatz's expected
params shape (n_layers, n_qubits, 2) -- no separate index bookkeeping needed.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ParameterDecoder(nn.Module):
    def __init__(self, dim: int = 64, hidden: int = 64):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(dim * 2, hidden), nn.ReLU(), nn.Linear(hidden, 2)
        )

    def forward(self, fused: torch.Tensor, h_ansatz: torch.Tensor, n_layers: int, n_qubits: int) -> torch.Tensor:
        combined = torch.cat([fused, h_ansatz], dim=-1)  # (n_layers*n_qubits, 2*dim)
        theta_flat = self.head(combined)  # (n_layers*n_qubits, 2)
        return theta_flat.view(n_layers, n_qubits, 2)
