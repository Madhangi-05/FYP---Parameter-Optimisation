"""Phase 2 training: quantum-in-the-loop, energy-only objective (no
geometry head yet -- that's Phase 4, no pre-optimized labels -- see
initializer.py docstring for why).

  loss = E(theta_0) = <psi(theta_0)| H |psi(theta_0)>

evaluated directly on a differentiable VQE circuit (vqe_torch.py), backprop
straight into the GNN. Decision gate (from work_implementation.docx Phase 2):
does the GNN-predicted theta_0 beat random initialization's E(theta_0) on
HELD-OUT instances?

Split is built programmatically from real sample counts per VQEzy file
(--train-per-file, --heldout-per-file), not a hardcoded instance list, so
this scales from a quick smoke test up to hundreds of instances with the
same code. xyz_12_qubit is always reserved entirely for held-out, as the
one genuine "unseen qubit count" test (all training stays <=8 qubits, so
circuit simulation cost stays cheap on CPU).
"""
from __future__ import annotations

import argparse
import random
import time

import torch
from pennylane import numpy as pnp

from src.data.vqezy_loader import iter_sample_keys, load_instance
from src.graphs.build_instance import build_graph_instance
from src.models.initializer import VQEInitializer
from src.quantum.hamiltonians import build_hamiltonian, true_ground_energy
from src.quantum.run_baseline_batch import REPO_ROOT
from src.quantum.vqe_torch import make_torch_vqe

TRAIN_FILES = [
    "external/VQEzy/qmanybody/xyz_4_qubit.h5",
    "external/VQEzy/qmanybody/fh_4_qubit.h5",
    "external/VQEzy/qmanybody/fh_6_qubit.h5",
    "external/VQEzy/qmanybody/fh_8_qubit.h5",
    "external/VQEzy/qmanybody/ti_8_qubit.h5",
]
UNSEEN_SIZE_FILE = "external/VQEzy/qmanybody/xyz_12_qubit.h5"  # reserved, never in training


def build_split(train_per_file: int, heldout_per_file: int, heldout_unseen_size: int, seed: int = 0):
    rng = random.Random(seed)
    train, held_out = [], []
    for h5_rel in TRAIN_FILES:
        keys = iter_sample_keys(REPO_ROOT / h5_rel)
        rng.shuffle(keys)
        train += [(h5_rel, k) for k in keys[:train_per_file]]
        held_out += [(h5_rel, k) for k in keys[train_per_file : train_per_file + heldout_per_file]]

    unseen_keys = iter_sample_keys(REPO_ROOT / UNSEEN_SIZE_FILE)
    rng.shuffle(unseen_keys)
    held_out += [(UNSEEN_SIZE_FILE, k) for k in unseen_keys[:heldout_unseen_size]]
    return train, held_out


def _prepare(h5_rel, sample, n_layers, topology, noise_level, seed, diff_method="best"):
    instance = load_instance(REPO_ROOT / h5_rel, sample)
    gi = build_graph_instance(instance, n_layers=n_layers, topology=topology, noise_level=noise_level, seed=seed)
    hamiltonian, n_qubits = build_hamiltonian(instance)
    circuit = make_torch_vqe(hamiltonian, n_qubits, n_layers, diff_method=diff_method)
    ground = true_ground_energy(hamiltonian, n_qubits)
    return instance, gi, circuit, ground


def unrolled_final_energy(model, gi, circuit, k_steps: int, inner_lr: float):
    """loss = E(theta_K), theta_K = K steps of GD from theta_0 = model(gi).
    Trains the GNN for 'leads somewhere good after K optimizer steps', not
    'starts low' -- the direct-E(theta0) objective was empirically shown to
    produce starting points with low energy but weak/flat local gradients
    (see compare_init_strategies.py results), which a subsequent optimizer
    then can't make progress from. Requires diff_method="parameter-shift"
    on `circuit` (create_graph=True needs a second-order-differentiable
    diff_method; "best"/adjoint does not reliably support that)."""
    theta = model(gi)
    for _ in range(k_steps):
        energy = circuit(theta)
        grad = torch.autograd.grad(energy, theta, create_graph=True)[0]
        theta = theta - inner_lr * grad
    return circuit(theta)


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
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--gnn-layers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train-per-file", type=int, default=16)
    ap.add_argument("--heldout-per-file", type=int, default=4)
    ap.add_argument("--heldout-unseen-size", type=int, default=5)
    ap.add_argument("--ckpt-name", default="phase2_initializer.pt")
    ap.add_argument(
        "--unroll-steps", type=int, default=0,
        help="If >0, train against E(theta_K) after K unrolled GD steps instead of E(theta_0). Slower but avoids optimizing straight into flat/plateau regions."
    )
    ap.add_argument("--unroll-lr", type=float, default=0.1, help="inner-loop GD stepsize used only for the unroll")
    args = ap.parse_args()

    train_instances, held_out_instances = build_split(
        args.train_per_file, args.heldout_per_file, args.heldout_unseen_size, seed=args.seed
    )

    torch.manual_seed(args.seed)
    model = VQEInitializer(hidden=args.hidden, gnn_layers=args.gnn_layers)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    diff_method = "parameter-shift" if args.unroll_steps > 0 else "best"
    t_prep0 = time.perf_counter()
    train_data = [
        _prepare(h5, s, args.n_layers, "ring", "medium", seed=i, diff_method=diff_method)
        for i, (h5, s) in enumerate(train_instances)
    ]
    print(f"Prepared {len(train_data)} training instances in {time.perf_counter()-t_prep0:.1f}s")

    loss_label = f"E(theta_{args.unroll_steps})" if args.unroll_steps > 0 else "E(theta_0)"
    print(f"Training on {len(train_data)} instances, {args.epochs} epochs, loss={loss_label}...")
    t_train0 = time.perf_counter()
    for epoch in range(args.epochs):
        epoch_losses = []
        t_epoch0 = time.perf_counter()
        for instance, gi, circuit, ground in train_data:
            opt.zero_grad()
            if args.unroll_steps > 0:
                energy = unrolled_final_energy(model, gi, circuit, args.unroll_steps, args.unroll_lr)
            else:
                energy = circuit(model(gi))
            energy.backward()
            opt.step()
            epoch_losses.append(energy.item())
        mean_loss = sum(epoch_losses) / len(epoch_losses)
        print(
            f"  epoch {epoch:3d}: mean train {loss_label} = {mean_loss:8.4f}  "
            f"({time.perf_counter()-t_epoch0:.1f}s)"
        )
    print(f"Total training time: {time.perf_counter()-t_train0:.1f}s")

    train_families_sizes = {
        (load_instance(REPO_ROOT / h5, s).family, load_instance(REPO_ROOT / h5, s).n_qubits)
        for h5, s in train_instances
    }

    print(f"\n=== Held-out evaluation ({len(held_out_instances)} instances, model never trained on these) ===")
    model.eval()
    wins = 0
    for h5, s in held_out_instances:
        instance, gi, circuit, ground = _prepare(h5, s, args.n_layers, "ring", "medium", seed=99)
        kind = (
            "unseen_qubit_count"
            if (instance.family, instance.n_qubits) not in train_families_sizes
            else "unseen_sample"
        )
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

    print(f"\nGNN beat random init on {wins}/{len(held_out_instances)} held-out instances.")

    ckpt_dir = REPO_ROOT / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    torch.save(model.state_dict(), ckpt_dir / args.ckpt_name)
    print(f"Saved -> {ckpt_dir / args.ckpt_name}")


if __name__ == "__main__":
    main()
