"""Checks torch_sim.py against PennyLane before it is trusted for training/eval.

  1. noiseless energy           vs lightning.qubit
  2. noiseless QNGD trajectory  vs qml.QNGOptimizer on lightning.qubit
  3. noisy energy               vs default.mixed with noise_model.noisy_czrxry_ansatz
  4. noisy QNGD trajectory      vs qml.QNGOptimizer on default.mixed
  5. unrolled meta-gradient d E(theta_K)/d theta_0 vs central finite differences
  6. complex64-on-GPU vs complex128-on-CPU drift over the unrolled steps

Noisy cases cover all three topologies and noise presets (routing matters
for linear/grid), so every branch of the noise model is exercised.
"""
from __future__ import annotations

import numpy as np
import pennylane as qml
import torch
from pennylane import numpy as pnp

from src.data.build_dataset import DATASET_DIR
from src.graphs.hardware_graph import build_hardware_graph
from src.quantum.ansatz import czrxry_ansatz
from src.quantum.hamiltonians import build_hamiltonian_from_params
from src.quantum.noise_model import noise_from_hardware_graph, noisy_czrxry_ansatz
from src.quantum.torch_sim import CZRXRYSim, hamiltonian_matrix

STEPS, ETA = 5, 0.02


def pl_qngd(circuit, theta0, steps):
    opt = qml.QNGOptimizer(stepsize=ETA)
    th = pnp.array(theta0, requires_grad=True)
    traj, energies = [np.array(th)], []
    for _ in range(steps):
        th, e = opt.step_and_cost(circuit, th)
        traj.append(np.array(th))
        energies.append(float(e))
    return np.array(traj), np.array(energies)


def torch_qngd(sim, theta0, steps):
    th = torch.tensor(theta0[None], dtype=torch.float64)
    traj, energies = [theta0], []
    for _ in range(steps):
        e, th = (lambda r: (r[0][0, 0].item(), r[1]))(sim.qngd_unroll(th, 1, ETA, create_graph=False))
        traj.append(th[0].numpy())
        energies.append(e)
    return np.array(traj), np.array(energies)


def main():
    data = torch.load(DATASET_DIR / "test.pt", weights_only=False)
    picked = {}
    for ci in data:
        if ci.n_qubits <= 8:
            picked.setdefault((ci.family, ci.n_qubits), ci)
    rng = np.random.default_rng(7)
    worst = {k: 0.0 for k in ["E_clean", "traj_clean", "E_noisy", "traj_noisy", "metagrad_rel", "c64_drift"]}
    configs = [("low", "linear"), ("medium", "ring"), ("high", "grid"), ("high", "linear")]

    for idx, ((fam, n), ci) in enumerate(picked.items()):
        H, _ = build_hamiltonian_from_params(fam, ci.coupling, n)
        Hm = torch.tensor(hamiltonian_matrix(H, n)[None], dtype=torch.complex128)
        theta0 = rng.uniform(0, 2 * np.pi, (2, n, 2))

        # --- noiseless ---
        dev = qml.device("lightning.qubit", wires=n)

        @qml.qnode(dev, diff_method="parameter-shift")
        def clean(t):
            czrxry_ansatz(t, n, 2)
            return qml.expval(H)

        sim = CZRXRYSim(n, 2, Hm)
        e_t = sim.energy(torch.tensor(theta0[None]))[0].item()
        worst["E_clean"] = max(worst["E_clean"], abs(e_t - float(clean(pnp.array(theta0)))))
        tr_pl, _ = pl_qngd(clean, theta0, STEPS)
        tr_t, _ = torch_qngd(sim, theta0, STEPS)
        worst["traj_clean"] = max(worst["traj_clean"], np.abs(tr_pl - tr_t).max())

        # --- noisy (default.mixed is slow: 4-6 qubits get the full trajectory check) ---
        level, topo = configs[idx % len(configs)]
        hw = build_hardware_graph(n, topology=topo, noise_level=level, seed=idx)
        nz = noise_from_hardware_graph(hw, n)
        dev_m = qml.device("default.mixed", wires=n)

        @qml.qnode(dev_m, diff_method="parameter-shift")
        def noisy(t):
            noisy_czrxry_ansatz(t, n, 2, nz)
            return qml.expval(H)

        sim_n = CZRXRYSim(n, 2, Hm, noise=[nz])
        e_t = sim_n.energy(torch.tensor(theta0[None]))[0].item()
        worst["E_noisy"] = max(worst["E_noisy"], abs(e_t - float(noisy(pnp.array(theta0)))))
        if n <= 6:
            tr_pl, _ = pl_qngd(noisy, theta0, 2)
            tr_t, _ = torch_qngd(sim_n, theta0, 2)
            worst["traj_noisy"] = max(worst["traj_noisy"], np.abs(tr_pl - tr_t).max())

        # --- meta-gradient vs finite differences (noisy sim, K=3) ---
        t0 = torch.tensor(theta0[None], requires_grad=True)
        loss = sim_n.qngd_unroll(t0, 3, ETA)[0][0, -1]
        (g,) = torch.autograd.grad(loss, t0)
        v = torch.tensor(rng.standard_normal(theta0.shape)[None])
        eps = 1e-5
        f = lambda t: sim_n.qngd_unroll(t, 3, ETA, create_graph=False)[0][0, -1].item()
        fd = (f(t0.detach() + eps * v) - f(t0.detach() - eps * v)) / (2 * eps)
        an = (g * v).sum().item()
        worst["metagrad_rel"] = max(worst["metagrad_rel"], abs(an - fd) / max(1e-8, abs(fd)))

        # --- complex64 on GPU vs complex128 on CPU ---
        if torch.cuda.is_available():
            sim32 = CZRXRYSim(n, 2, Hm.to("cuda", torch.complex64), noise=[nz])
            e32 = sim32.qngd_unroll(torch.tensor(theta0[None], dtype=torch.float32, device="cuda"), 3, ETA, create_graph=False)[0]
            e64 = sim_n.qngd_unroll(torch.tensor(theta0[None]), 3, ETA, create_graph=False)[0]
            worst["c64_drift"] = max(worst["c64_drift"], (e32.cpu().double() - e64).abs().max().item())

        print(f"{fam}_{n} ({level}/{topo}) checked", flush=True)

    print("\nWorst-case discrepancies:")
    for k, v in worst.items():
        print(f"  {k:14s} {v:.3e}")


if __name__ == "__main__":
    main()
