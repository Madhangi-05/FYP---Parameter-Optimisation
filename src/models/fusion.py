"""Cross-attention fusion (Section 5.1): ansatz nodes attend to Hamiltonian
structure first, then to hardware/noise structure -- noise is injected into
the fusion itself, not appended as an extra feature at the end.

Query = ansatz-node embeddings (one per (qubit, layer) gate slot).
Key/value stage 1 = Hamiltonian qubit-node embeddings.
Key/value stage 2 = hardware/noise qubit-node embeddings.
Sequence lengths differ (n_layers*n_qubits query vs n_qubits key/value) --
nn.MultiheadAttention handles that natively, no manual broadcasting needed.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class CrossAttentionFusion(nn.Module):
    def __init__(self, dim: int = 64, heads: int = 4):
        super().__init__()
        self.attn_ansatz_to_ham = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.attn_fused_to_hw = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, h_ansatz: torch.Tensor, h_ham: torch.Tensor, h_hw: torch.Tensor | None) -> torch.Tensor:
        q = h_ansatz.unsqueeze(0)  # (1, n_ansatz_nodes, dim)
        kv_h = h_ham.unsqueeze(0)  # (1, n_qubits, dim)

        fused_ha, _ = self.attn_ansatz_to_ham(q, kv_h, kv_h)
        fused_ha = self.norm1(q + fused_ha)
        if h_hw is None:  # no hardware/noise graph (ablation: zero noise awareness)
            return fused_ha.squeeze(0)
        kv_d = h_hw.unsqueeze(0)  # (1, n_qubits, dim)

        fused_had, _ = self.attn_fused_to_hw(fused_ha, kv_d, kv_d)
        fused = self.norm2(fused_ha + fused_had)

        return fused.squeeze(0)  # (n_ansatz_nodes, dim)
