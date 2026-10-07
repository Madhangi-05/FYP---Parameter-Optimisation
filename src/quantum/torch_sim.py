"""Exact, batched, differentiable density-matrix simulator for czrxry_ansatz
(optionally with the noise model in noise_model.py), plus the QNGD update
exactly as qml.QNGOptimizer computes it.

Why a custom simulator: training the GNN against the energy after K QNGD
steps needs d(theta_K)/d(theta_0) THROUGH the metric tensor and its pinv.
PennyLane can't give that cheaply (and default.mixed is far too slow for
1000s of noisy instances), whereas a plain-torch simulator batches many
same-size instances on the GPU and autograd handles everything. It is only
trusted because validate_torch_sim.py checks it against PennyLane: energies,
QNGD steps (lightning.qubit noiseless, default.mixed noisy) and the
unrolled meta-gradient vs finite differences.

Conventions match PennyLane: wire 0 is the most significant bit;
RX(t) = exp(-i t X/2), RY(t) = exp(-i t Y/2).

QNGD (qml.QNGOptimizer, approx="block-diag", lam=0):
    theta <- theta - eta * pinv(F) @ grad E
F is block-diagonal over PennyLane's parametrized layers (see run() for how
those fall on this ansatz); each block is the covariance of the gate
generators (X_q/2 or Y_q/2) in the state just before that block.
"""
from __future__ import annotations

import numpy as np
import pennylane as qml
import torch

from src.quantum.noise_model import NoiseParams, ansatz_cz_pairs


def hamiltonian_matrix(hamiltonian, n_qubits: int) -> np.ndarray:
    return qml.matrix(hamiltonian, wire_order=range(n_qubits))


def hamiltonian_l1(hamiltonian) -> float:
    coeffs, _ = hamiltonian.terms()
    return float(np.sum(np.abs(np.asarray(coeffs))))


class CZRXRYSim:
    """Simulates a batch of B same-size instances. Noise is per instance
    (pass a list of B NoiseParams, or None for noiseless)."""

    def __init__(self, n_qubits: int, n_layers: int, H: torch.Tensor, noise: list | None = None):
        self.n, self.L = n_qubits, n_layers
        self.H = H  # (B, 2^n, 2^n) complex
        self.B = H.shape[0]
        self.cdtype = H.dtype
        self.rdtype = torch.float64 if self.cdtype == torch.complex128 else torch.float32
        dev = H.device
        self.device = dev
        dim = 2**n_qubits
        bits = ((torch.arange(dim, device=dev)[:, None] >> torch.arange(n_qubits - 1, -1, -1, device=dev)) & 1)
        self.z = (1 - 2 * bits).to(self.rdtype)  # (dim, n): Z_q eigenvalue of basis state s
        self.cz_pairs = ansatz_cz_pairs(n_qubits)
        self.cz_diag = []
        for a, b in self.cz_pairs:
            d = 1 - 2 * (bits[:, a] * bits[:, b])
            self.cz_diag.append((d[:, None] * d[None, :]).to(self.rdtype))
        if noise is not None:
            t = lambda arr: torch.tensor(np.stack(arr), dtype=self.rdtype, device=dev)
            self.cz_p = t([nz.cz_p for nz in noise])  # (B, n_cz)
            self.sq_p = t([nz.sq_p for nz in noise])  # (B, n)
            self.gamma = t([nz.gamma for nz in noise])
            self.readout_p = t([nz.readout_p for nz in noise])
        self.noisy = noise is not None
        h = torch.tensor([[1, 1], [1, -1]], dtype=self.cdtype, device=dev) / np.sqrt(2)
        sdg = torch.tensor([[1, 0], [0, -1j]], dtype=self.cdtype, device=dev)
        # V with V^dag Z V = generator Pauli: H for X, H S^dag for Y.
        self.basis_rot = {0: h, 1: h @ sdg}

    # ---- primitive ops on rho (B, dim, dim) ----
    def _view(self, rho, q):
        a, c = 2**q, 2 ** (self.n - q - 1)
        return rho.reshape(self.B, a, 2, c, a, 2, c)

    def _apply_1q(self, rho, U, q):
        """U: (B,2,2) or (2,2)."""
        r = self._view(rho, q)
        if U.dim() == 2:
            U = U.expand(self.B, 2, 2)
        r = torch.einsum("bxi,baicdje->baxcdje", U, r)
        r = torch.einsum("baxcdje,byj->baxcdye", r, U.conj())
        return r.reshape(rho.shape)

    def _depolarize(self, rho, p, q):
        """(1-p) rho + p/3 (X rho X + Y rho Y + Z rho Z)
        = (1 - 4p/3) rho + (2p/3) I_q (x) Tr_q(rho).   p: (B,)"""
        r = self._view(rho, q)
        tr = r[:, :, 0, :, :, 0, :] + r[:, :, 1, :, :, 1, :]  # (B,a,c,a,c)
        eye = torch.eye(2, dtype=self.cdtype, device=self.device)
        mixed = torch.einsum("bacde,ij->baicdje", tr, eye)
        p = p.to(self.cdtype).view(-1, 1, 1, 1, 1, 1, 1)
        return ((1 - 4 * p / 3) * r + (2 * p / 3) * mixed).reshape(rho.shape)

    def _amp_damp(self, rho, g, q):
        """Kraus K0 = diag(1, sqrt(1-g)), K1 = [[0, sqrt g],[0, 0]].   g: (B,)"""
        r = self._view(rho, q)
        g = g.to(self.cdtype).view(-1, 1, 1, 1, 1)
        s = torch.sqrt(1 - g)
        r00, r01 = r[:, :, 0, :, :, 0, :], r[:, :, 0, :, :, 1, :]
        r10, r11 = r[:, :, 1, :, :, 0, :], r[:, :, 1, :, :, 1, :]
        n00, n01, n10, n11 = r00 + g * r11, s * r01, s * r10, (1 - g) * r11
        row0 = torch.stack([n00, n01], dim=4)  # (B,a,c,a,2,c) over j
        row1 = torch.stack([n10, n11], dim=4)
        return torch.stack([row0, row1], dim=2).reshape(rho.shape)

    def _rot(self, theta, axis):
        """theta: (B,) real -> (B,2,2) RX (axis 0) / RY (axis 1)."""
        c = torch.cos(theta / 2).to(self.cdtype)
        s = torch.sin(theta / 2).to(self.cdtype)
        if axis == 0:
            return torch.stack([torch.stack([c, -1j * s], -1), torch.stack([-1j * s, c], -1)], -2)
        return torch.stack([torch.stack([c, -s], -1), torch.stack([s, c], -1)], -2)

    # ---- circuit ----
    def run(self, theta):
        """theta: (B, L, n, 2). Returns (energy (B,), list of (rho, block)) for
        the block-diag metric tensor, block = [(layer, qubit, axis), ...].

        Gates are applied in czrxry_layer's queue order (RX(q), RY(q), q=0..n-1)
        because PennyLane's block-diag layering depends on it: it walks the
        trainable gates in queue order and starts a new block whenever a gate
        depends on one already in the block. Per ansatz layer that gives
            [RX0], [RY0, RX1], [RY1, RX2], ..., [RY(n-2), RX(n-1)], [RY(n-1)]
        (verified against qml.metric_tensor). Each block's covariance is taken
        in the state right before the block's first gate."""
        dim = 2**self.n
        rho = torch.zeros(self.B, dim, dim, dtype=self.cdtype, device=self.device)
        rho[:, 0, 0] = 1
        pre = []
        for l in range(self.L):
            for k, (a, b) in enumerate(self.cz_pairs):
                rho = rho * self.cz_diag[k]
                if self.noisy:
                    rho = self._depolarize(rho, self.cz_p[:, k], a)
                    rho = self._depolarize(rho, self.cz_p[:, k], b)
            pre.append((rho, [(l, 0, 0)]))
            for q in range(self.n):
                rho = self._apply_1q(rho, self._rot(theta[:, l, q, 0], 0), q)
                if self.noisy:
                    rho = self._depolarize(rho, self.sq_p[:, q], q)
                block = [(l, q, 1)] + ([(l, q + 1, 0)] if q + 1 < self.n else [])
                pre.append((rho, block))
                rho = self._apply_1q(rho, self._rot(theta[:, l, q, 1], 1), q)
                if self.noisy:
                    rho = self._depolarize(rho, self.sq_p[:, q], q)
            if self.noisy:
                for q in range(self.n):
                    rho = self._amp_damp(rho, self.gamma[:, q], q)
        if self.noisy:
            for q in range(self.n):
                rho = self._depolarize(rho, self.readout_p[:, q], q)
        energy = torch.einsum("bij,bji->b", rho, self.H).real
        return energy, pre

    def energy(self, theta):
        return self.run(theta)[0]

    def metric_tensor(self, pre):
        """Block-diag metric (B, P, P), P flattened like theta.reshape(B, -1)
        i.e. index = l*2n + 2q + axis. Block entries: covariance of the
        generators P_q/2 (P = X for RX, Y for RY), measured by rotating each
        block qubit into its generator's eigenbasis and reading probabilities."""
        P = self.L * self.n * 2
        F = torch.zeros(self.B, P, P, dtype=self.rdtype, device=self.device)
        for rho, block in pre:
            r = rho
            for _, q, axis in block:
                r = self._apply_1q(r, self.basis_rot[axis], q)
            probs = torch.diagonal(r, dim1=-2, dim2=-1).real  # (B, dim)
            qs = [q for _, q, _ in block]
            z = self.z[:, qs]  # (dim, k)
            m = probs @ z
            zz = torch.einsum("bs,si,sj->bij", probs, z, z)
            cov = (zz - m[:, :, None] * m[:, None, :]) / 4
            idx = torch.tensor([l * 2 * self.n + 2 * q + axis for l, q, axis in block], device=self.device)
            F[:, idx[:, None], idx[None, :]] = cov
        return F

    def qngd_unroll(self, theta0, n_steps: int, stepsize: float, create_graph: bool = True):
        """Runs n_steps of QNGOptimizer's update from theta0 (B,L,n,2).
        Returns energies (B, n_steps+1): E(theta_0..theta_K) -- last one is
        the training loss for the unrolled objective."""
        theta = theta0
        energies = []
        for _ in range(n_steps):
            if not theta.requires_grad:
                theta = theta.requires_grad_(True)
            e, pre = self.run(theta)
            energies.append(e)
            g = torch.autograd.grad(e.sum(), theta, create_graph=create_graph)[0]
            F = self.metric_tensor(pre)
            if not create_graph:
                F = F.detach()
            upd = torch.linalg.pinv(F) @ g.reshape(self.B, -1, 1)
            theta = theta - stepsize * upd.reshape(theta.shape)
            if not create_graph:
                theta = theta.detach()
        energies.append(self.energy(theta))
        return torch.stack(energies, dim=1), theta
