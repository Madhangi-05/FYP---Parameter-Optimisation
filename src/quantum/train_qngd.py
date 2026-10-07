"""Trains the GNN initializer against the energy after K QNGD steps --
the optimizer it is evaluated with -- using the validated torch simulator
(torch_sim.py), batched on the GPU.

    loss = mean over batch of  E(theta_K) / ||H||_1
    theta_{k+1} = theta_k - eta * pinv(F_blockdiag(theta_k)) grad E(theta_k)

exactly qml.QNGOptimizer's update (eta=0.02 as in evaluation), differentiated
end-to-end (through F and its pinv too) back into the GNN.

Two variants (the noise-awareness study):
  --variant clean : GNN without the hardware/noise graph, noiseless circuits,
                    data_cache/ (zero noise awareness).
  --variant noisy : GNN with the hardware/noise graph, noisy circuits whose
                    noise is derived from that same graph, data_cache_noisy/
                    (varied presets + topologies).

Differences from train_gpu.py, and why:
  - minibatches of same-size instances (the simulator batches over them)
    instead of one instance per optimizer step;
  - energy normalized by the Hamiltonian's coefficient 1-norm, so ti
    (energies ~ -30) doesn't drown out fh (~ -4) in the batch mean;
  - the checkpoint kept is the epoch with the best validation loss.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch

from src.data.build_dataset import DATASET_DIR
from src.data.build_noisy_dataset import NOISY_DATASET_DIR
from src.models.initializer import VQEInitializer
from src.quantum.hamiltonians import build_hamiltonian_from_params
from src.quantum.noise_model import noise_from_hardware_graph
from src.quantum.run_baseline_batch import REPO_ROOT
from src.quantum.torch_sim import CZRXRYSim, hamiltonian_l1, hamiltonian_matrix


def prepare(data, device, noisy, cdtype):
    """Per-instance simulator inputs, computed once."""
    for ci in data:
        H, n = build_hamiltonian_from_params(ci.family, ci.coupling, ci.n_qubits)
        ci.H_mat = torch.tensor(hamiltonian_matrix(H, n), dtype=cdtype, device=device)
        ci.h_l1 = hamiltonian_l1(H)
        ci.noise = noise_from_hardware_graph(ci.hardware_graph, n) if noisy else None
        for g in ("hamiltonian_graph", "ansatz_graph", "hardware_graph"):
            setattr(ci, g, getattr(ci, g).to(device))
    return data


def batches(data, batch_size_for, shuffle, rng):
    by_n: dict[int, list] = {}
    for ci in data:
        by_n.setdefault(ci.n_qubits, []).append(ci)
    out = []
    for n, items in by_n.items():
        items = items[:]
        if shuffle:
            rng.shuffle(items)
        bs = batch_size_for(n)
        out += [items[i : i + bs] for i in range(0, len(items), bs)]
    if shuffle:
        rng.shuffle(out)
    return out


def batch_loss(model, batch, args, create_graph):
    n, L = batch[0].n_qubits, batch[0].n_layers
    H = torch.stack([ci.H_mat for ci in batch])
    noise = [ci.noise for ci in batch] if batch[0].noise is not None else None
    sim = CZRXRYSim(n, L, H, noise=noise)
    theta0 = torch.stack([model(ci) for ci in batch])
    energies, _ = sim.qngd_unroll(theta0, args.unroll_steps, args.stepsize, create_graph=create_graph)
    if args.no_normalize:
        return energies, energies
    scale = torch.tensor([ci.h_l1 for ci in batch], device=H.device, dtype=energies.dtype)
    return (energies / scale[:, None]), energies


def objective(norm_e, loss_kind):
    """final: E(theta_K). traj_mean: mean of E(theta_1..theta_K) -- penalizes
    starts that drop fast for a few steps and then stall in a shallow basin
    (short-horizon bias, seen with K=3 'final'). E(theta_0) is excluded on
    purpose: rewarding a low START energy is the original plateau trap."""
    return norm_e[:, -1] if loss_kind == "final" else norm_e[:, 1:].mean(dim=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["clean", "noisy"], required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--gnn-layers", type=int, default=3)
    ap.add_argument("--unroll-steps", type=int, default=3)
    ap.add_argument("--stepsize", type=float, default=0.02)
    ap.add_argument("--batch-elems", type=int, default=2**20, help="batch size = max(1, this // 4^n_qubits)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--loss", choices=["final", "traj_mean"], default="final")
    ap.add_argument("--no-normalize", action="store_true", help="raw energies instead of E/||H||_1")
    ap.add_argument("--ckpt-name", default=None)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    noisy = args.variant == "noisy"
    data_dir = Path(NOISY_DATASET_DIR if noisy else DATASET_DIR)
    ckpt_name = args.ckpt_name or f"qngd_{args.variant}.pt"
    cdtype = torch.complex64  # validated: <3e-6 drift vs complex128 over the unroll
    print(f"variant={args.variant}  data={data_dir}  device={device}  K={args.unroll_steps}  eta={args.stepsize}  loss={args.loss}  normalize={not args.no_normalize}")

    t0 = time.perf_counter()
    train = prepare(torch.load(data_dir / "train.pt", weights_only=False), device, noisy, cdtype)
    val = prepare(torch.load(data_dir / "val.pt", weights_only=False), device, noisy, cdtype)
    print(f"Prepared train={len(train)} val={len(val)} in {time.perf_counter()-t0:.1f}s")

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    model = VQEInitializer(hidden=args.hidden, gnn_layers=args.gnn_layers, use_hardware=noisy).to(device)
    print(f"Model params: {sum(p.numel() for p in model.parameters())}  use_hardware={noisy}")
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    bs_for = lambda n: max(1, args.batch_elems // 4**n)

    ckpt_dir = REPO_ROOT / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    history, best = [], float("inf")
    for epoch in range(args.epochs):
        model.train()
        t_ep, losses = time.perf_counter(), []
        for batch in batches(train, bs_for, True, rng):
            opt.zero_grad()
            norm_e, _ = batch_loss(model, batch, args, create_graph=True)
            obj = objective(norm_e, args.loss)
            obj.mean().backward()
            opt.step()
            losses += obj.detach().cpu().tolist()

        model.eval()
        val_e0, val_ek = [], []
        for batch in batches(val, bs_for, False, rng):
            with torch.enable_grad():
                norm_e, _ = batch_loss(model, batch, args, create_graph=False)
            val_e0 += norm_e[:, 0].detach().cpu().tolist()
            val_ek += objective(norm_e, args.loss).detach().cpu().tolist()
        tr, v0, vk = (sum(x) / len(x) for x in (losses, val_e0, val_ek))
        improved = vk < best
        if improved:
            best = vk
            torch.save(model.state_dict(), ckpt_dir / ckpt_name)
        dt = time.perf_counter() - t_ep
        history.append({"epoch": epoch, "train_EK": tr, "val_E0": v0, "val_EK": vk, "seconds": dt})
        print(f"  epoch {epoch:3d}: train obj {tr:8.4f}   val E_0 {v0:8.4f}  obj {vk:8.4f}  ({dt:.0f}s){'  *saved' if improved else ''}", flush=True)

    out = REPO_ROOT / "results" / f"train_history_{Path(ckpt_name).stem}.json"
    with open(out, "w") as f:
        json.dump({"args": vars(args), "history": history, "best_val_EK": best}, f, indent=2)
    print(f"\nBest val E_K/|H| = {best:.4f} -> {ckpt_dir / ckpt_name}")
    print(f"History -> {out}")


if __name__ == "__main__":
    main()
