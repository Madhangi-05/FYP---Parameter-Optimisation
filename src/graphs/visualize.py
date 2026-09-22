"""Renders G_H, G_A, G_D for one VQEzy instance side by side, so the graphs
built by hamiltonian_graph.py / ansatz_graph.py / hardware_graph.py can
actually be looked at, not just shape-checked."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import torch

from src.data.vqezy_loader import load_instance
from src.graphs.build_instance import build_graph_instance

REPO_ROOT = Path(__file__).resolve().parents[2]


def _to_nx(data, directed_pairs_as_single=True):
    g = nx.Graph() if directed_pairs_as_single else nx.DiGraph()
    g.add_nodes_from(range(data.x.shape[0]))
    ei = data.edge_index.numpy()
    ea = data.edge_attr.numpy() if data.edge_attr is not None else None
    for k in range(ei.shape[1]):
        u, v = int(ei[0, k]), int(ei[1, k])
        if u == v:
            continue
        attr = {"feat": ea[k]} if ea is not None else {}
        g.add_edge(u, v, **attr)
    return g


def plot_instance(gi, out_path: Path, title_suffix: str = ""):
    fig, axes = plt.subplots(1, 3, figsize=(19, 6))

    # --- Hamiltonian graph: nodes = qubits, edge width/label = coupling weight ---
    gh = gi.hamiltonian_graph
    G = _to_nx(gh)
    pos = nx.circular_layout(G)
    ax = axes[0]
    weights = [G[u][v]["feat"][0] for u, v in G.edges()]
    max_w = max(weights) if weights else 1.0
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color="#8ecae6", node_size=700)
    nx.draw_networkx_labels(G, pos, ax=ax, labels={i: f"q{i}" for i in G.nodes()})
    nx.draw_networkx_edges(
        G, pos, ax=ax, width=[1 + 4 * w / max_w for w in weights], edge_color="#023047"
    )
    edge_labels = {(u, v): f"{G[u][v]['feat'][0]:.2f}" for u, v in G.edges()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels, ax=ax, font_size=7)
    ax.set_title(f"Hamiltonian graph $G_H$\n({gi.source.family}, {gi.n_qubits}q)\nedge label = |coeff| sum")
    ax.axis("off")

    # --- Ansatz graph: nodes = (qubit, layer), colour by edge type ---
    # Concentric circles (one ring per layer, same angle per qubit) so the CZ
    # ring's wraparound edge (q_{n-1} <-> q_0) is visible as a ring instead of
    # overlapping the chain, which happens with a straight-column layout.
    import math

    ga = gi.ansatz_graph
    n_qubits, n_layers = gi.n_qubits, gi.n_layers
    node_pos = {}
    labels = {}
    idx = 0
    for l in range(n_layers):
        radius = 1.0 + 0.9 * l
        for q in range(n_qubits):
            angle = 2 * math.pi * q / n_qubits + math.pi / 2
            node_pos[idx] = (radius * math.cos(angle), radius * math.sin(angle))
            labels[idx] = f"q{q}L{l}"
            idx += 1
    G2 = nx.Graph()
    G2.add_nodes_from(range(ga.x.shape[0]))
    ei = ga.edge_index.numpy()
    ea = ga.edge_attr.numpy()
    ent_edges, temp_edges = [], []
    for k in range(ei.shape[1]):
        u, v = int(ei[0, k]), int(ei[1, k])
        if u > v:
            continue  # undirected, drawn once
        (ent_edges if ea[k, 0] == 1.0 else temp_edges).append((u, v))
    ax = axes[1]
    nx.draw_networkx_nodes(G2, node_pos, ax=ax, node_color="#ffb703", node_size=500)
    nx.draw_networkx_labels(G2, node_pos, ax=ax, labels=labels, font_size=7)
    nx.draw_networkx_edges(G2, node_pos, ax=ax, edgelist=ent_edges, edge_color="#d62828", width=2, label="entangling (CZ)")
    nx.draw_networkx_edges(G2, node_pos, ax=ax, edgelist=temp_edges, edge_color="#219ebc", width=2, style="dashed", label="temporal")
    ax.legend(loc="lower center", fontsize=8)
    ax.set_title(f"Ansatz graph $G_A$\n(CZRXRY, {n_layers} layers x {n_qubits} qubits)\nnode = 1 RX+RY pair")
    ax.axis("off")

    # --- Hardware/noise graph: nodes = physical qubits, size/color = noise ---
    gd = gi.hardware_graph
    G3 = _to_nx(gd)
    pos3 = nx.circular_layout(G3) if gd.topology != "grid" else nx.spring_layout(G3, seed=0)
    ax = axes[2]
    readout = gd.x[:, 0].numpy()
    node_colors = readout
    sc = nx.draw_networkx_nodes(
        G3, pos3, ax=ax, node_color=node_colors, cmap="Reds", node_size=700, vmin=0, vmax=readout.max() * 1.3
    )
    nx.draw_networkx_labels(G3, pos3, ax=ax, labels={i: f"q{i}" for i in G3.nodes()})
    nx.draw_networkx_edges(G3, pos3, ax=ax, edge_color="#555555")
    edge_labels3 = {(u, v): f"{G3[u][v]['feat'][0]:.3f}" for u, v in G3.edges()}
    nx.draw_networkx_edge_labels(G3, pos3, edge_labels3, ax=ax, font_size=7)
    plt.colorbar(sc, ax=ax, label="readout error", fraction=0.046, pad=0.04)
    ax.set_title(
        f"Hardware/noise graph $G_D$\ntopology={gd.topology}, noise={gd.noise_level}\n"
        f"node color = readout err, edge label = 2Q gate err"
    )
    ax.axis("off")

    fig.suptitle(f"Instance: {gi.source.family} / {gi.source.sample_key}{title_suffix}", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    print(f"Saved -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", default=str(REPO_ROOT / "external/VQEzy/qmanybody/xyz_4_qubit.h5"))
    ap.add_argument("--sample", default="sample_0")
    ap.add_argument("--n-layers", type=int, default=2)
    ap.add_argument("--topology", default="ring")
    ap.add_argument("--noise-level", default="medium")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    instance = load_instance(args.h5, args.sample)
    gi = build_graph_instance(
        instance, n_layers=args.n_layers, topology=args.topology, noise_level=args.noise_level, seed=0
    )
    out_path = Path(args.out) if args.out else REPO_ROOT / "results" / f"graphs_{instance.family}_{args.sample}.png"
    out_path.parent.mkdir(exist_ok=True)
    plot_instance(gi, out_path)


if __name__ == "__main__":
    main()
