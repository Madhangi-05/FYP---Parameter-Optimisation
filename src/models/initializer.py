"""f_theta(G_H, G_A, G_D) -> theta_0 (Eq. 3.1 / Section 5.1), minus the
geometry head (that's Phase 4). Ties GraphEncoder x3 + CrossAttentionFusion
+ ParameterDecoder together.

Fixed input/edge dims come from the graph builders (see encoders.py
docstring) -- not configurable per-instance, since the graph schemas are
fixed regardless of n_qubits.
"""
from __future__ import annotations

import torch.nn as nn

from src.models.decoder import ParameterDecoder
from src.models.encoders import GraphEncoder
from src.models.fusion import CrossAttentionFusion

HAM_NODE_DIM, HAM_EDGE_DIM = 5, 5
ANSATZ_NODE_DIM, ANSATZ_EDGE_DIM = 4, 2
HW_NODE_DIM, HW_EDGE_DIM = 3, 1


class VQEInitializer(nn.Module):
    def __init__(self, hidden: int = 64, gnn_layers: int = 3, attn_heads: int = 4):
        super().__init__()
        self.ham_encoder = GraphEncoder(HAM_NODE_DIM, HAM_EDGE_DIM, hidden, gnn_layers)
        self.ansatz_encoder = GraphEncoder(ANSATZ_NODE_DIM, ANSATZ_EDGE_DIM, hidden, gnn_layers)
        self.hw_encoder = GraphEncoder(HW_NODE_DIM, HW_EDGE_DIM, hidden, gnn_layers)
        self.fusion = CrossAttentionFusion(hidden, attn_heads)
        self.decoder = ParameterDecoder(hidden)

    def forward(self, gi):
        """gi: a src.graphs.build_instance.GraphInstance"""
        gh, ga, gd = gi.hamiltonian_graph, gi.ansatz_graph, gi.hardware_graph

        h_ham = self.ham_encoder(gh.x, gh.edge_index, gh.edge_attr)
        h_ansatz = self.ansatz_encoder(ga.x, ga.edge_index, ga.edge_attr)
        h_hw = self.hw_encoder(gd.x, gd.edge_index, gd.edge_attr)

        fused = self.fusion(h_ansatz, h_ham, h_hw)
        theta0 = self.decoder(fused, h_ansatz, gi.n_layers, gi.n_qubits)
        return theta0
