"""Phase 2 training: quantum-in-the-loop, energy-only objective (no
geometry head yet -- that's Phase 4, no pre-optimized labels -- see
initializer.py docstring for why).

  loss = E(theta_0) = <psi(theta_0)| H |psi(theta_0)>

evaluated directly on a differentiable VQE circuit (vqe_torch.py), backprop
straight into the GNN. Decision gate (from work_implementation.docx Phase 2):
does the GNN-predicted theta_0 beat random initialization's E(theta_0) on
HELD-OUT instances?

Only 10 instances total here (7 train / 3 held-out) -- this is a smoke test
that the architecture can learn something at all, not a real experiment.
Scaling to hundreds+ of VQEzy instances is the natural next step once this
is confirmed to work.
"""
from __future__ import annotations

import argparse

import torch
from pennylane import numpy as pnp

from src.data.vqezy_loader import load_instance
from src.graphs.build_instance import build_graph_instance
from src.models.initializer import VQEInitializer
from src.quantum.hamiltonians import build_hamiltonian, true_ground_energy
from src.quantum.run_baseline_batch import REPO_ROOT
from src.quantum.vqe_torch import make_torch_vqe

# All 3 families represented in TRAINING this time (the previous split
# accidentally excluded `ti` entirely, so its held-out failures were really
# "never seen this family at all" rather than a fair generalization test).
# Held-out now separates two different kinds of generalization:
#   - unseen SAMPLE within a trained family+size (xyz s2, fh_6q s1, ti s1)
#   - unseen QUBIT COUNT entirely (xyz_12_qubit -- never trained above 8q)
TRAIN_INSTANCES = [
    ("external/VQEzy/qmanybody/xyz_4_qubit.h5", "sample_0"),
    ("external/VQEzy/qmanybody/xyz_4_qubit.h5", "sample_1"),
    ("external/VQEzy/qmanybody/fh_4_qubit.h5", "sample_0"),
    ("external/VQEzy/qmanybody/fh_4_qubit.h5", "sample_1"),
    ("external/VQEzy/qmanybody/fh_6_qubit.h5", "sample_0"),
    ("external/VQEzy/qmanybody/fh_8_qubit.h5", "sample_0"),
    ("external/VQEzy/qmanybody/ti_8_qubit.h5", "sample_0"),
    ("external/VQEzy/qmanybody/ti_8_qubit.h5", "sample_2"),
]
HELD_OUT_INSTANCES = [
    ("external/VQEzy/qmanybody/xyz_4_qubit.h5", "sample_2"),
    ("external/VQEzy/qmanybody/xyz_12_qubit.h5", "sample_0"),
    ("external/VQEzy/qmanybody/fh_6_qubit.h5", "sample_1"),
    ("external/VQEzy/qmanybody/ti_8_qubit.h5", "sample_1"),
]


def _prepare(h5_rel, sample, n_layers, topology, noise_level, seed):
    instance = load_instance(REPO_ROOT / h5_rel, sample)
    gi = build_graph_instance(instance, n_layers=n_layers, topology=topology, noise_level=noise_level, seed=seed)
    hamiltonian, n_qubits = build_hamiltonian(instance)
    circuit = make_torch_vqe(hamiltonian, n_qubits, n_layers)
    ground = true_ground_energy(hamiltonian, n_qubits)
    return instance, gi, circuit, ground


def random_baseline_energy(circuit, n_layers, n_qubits, n_trials=5, seed=0):
    rng = pnp.random.default_rng(seed)
    energies = []
    for _ in range(n_trials):
        theta = torch.tensor(
            rng.uniform(0, 2 * pnp.pi, size=(n_layers, n_qubits, 2)), dtype=torch.float32
        )
        energies.append(circuit(theta).item())
    return sum(energies) / len(energies)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-layers", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--gnn-layers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    model = VQEInitializer(hidden=args.hidden, gnn_layers=args.gnn_layers)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_data = [
        _prepare(h5, s, args.n_layers, "ring", "medium", seed=i)
        for i, (h5, s) in enumerate(TRAIN_INSTANCES)
    ]

    print(f"Training on {len(train_data)} instances, {args.epochs} epochs...")
    for epoch in range(args.epochs):
        epoch_losses = []
        for instance, gi, circuit, ground in train_data:
            opt.zero_grad()
            theta0 = model(gi)
            energy = circuit(theta0)
            energy.backward()
            opt.step()
            epoch_losses.append(energy.item())
        if epoch % 5 == 0 or epoch == args.epochs - 1:
            mean_loss = sum(epoch_losses) / len(epoch_losses)
            print(f"  epoch {epoch:3d}: mean train E(theta0) = {mean_loss:8.4f}")

    train_families_sizes = {(load_instance(REPO_ROOT / h5, s).family, load_instance(REPO_ROOT / h5, s).n_qubits) for h5, s in TRAIN_INSTANCES}

    print(f"\n=== Held-out evaluation ({len(HELD_OUT_INSTANCES)} instances, model never trained on these) ===")
    model.eval()
    wins = 0
    for h5, s in HELD_OUT_INSTANCES:
        instance, gi, circuit, ground = _prepare(h5, s, args.n_layers, "ring", "medium", seed=99)
        kind = "unseen_qubit_count" if (instance.family, instance.n_qubits) not in train_families_sizes else "unseen_sample"
        with torch.no_grad():
            theta0 = model(gi)
        gnn_energy = circuit(theta0).item()
        rand_energy = random_baseline_energy(circuit, args.n_layers, gi.n_qubits, n_trials=5, seed=42)
        better = gnn_energy < rand_energy
        wins += better
        print(
            f"  {instance.family:3s}/{s} ({instance.n_qubits}q, {kind:>18s}): "
            f"GNN_init_E={gnn_energy:8.4f}  "
            f"random_init_E(avg of 5)={rand_energy:8.4f}  "
            f"true_ground={ground if ground is not None else float('nan'):8.4f}  "
            f"GNN_better={better}"
        )

    print(f"\nGNN beat random init on {wins}/{len(HELD_OUT_INSTANCES)} held-out instances.")

    ckpt_dir = REPO_ROOT / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    torch.save(model.state_dict(), ckpt_dir / "phase2_initializer.pt")
    print(f"Saved -> {ckpt_dir / 'phase2_initializer.pt'}")


if __name__ == "__main__":
    main()
