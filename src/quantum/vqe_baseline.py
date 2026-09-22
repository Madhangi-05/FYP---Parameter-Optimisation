"""Phase 1: plain VQE+QNGD baseline (no learning yet).

Loads a real VQEzy instance (Hamiltonian + reference optimum), runs our own
VQE with (a) plain gradient descent and (b) QNGD, and reports circuit-eval
counts to reach the reference ground energy. This is the B0 reference point
that every later GNN-initialized / geometry-guided variant is measured
against.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pennylane as qml
from pennylane import numpy as pnp

from src.data.vqezy_loader import iter_sample_keys, load_instance
from src.quantum.ansatz import czrxry_ansatz, n_params
from src.quantum.hamiltonians import EXACT_DIAG_MAX_QUBITS, build_hamiltonian, true_ground_energy

REPO_ROOT = Path(__file__).resolve().parents[2]


def _make_qnode(hamiltonian, n_qubits, n_layers, dev):
    @qml.qnode(dev, diff_method="parameter-shift")
    def circuit(params):
        czrxry_ansatz(params, n_qubits, n_layers)
        return qml.expval(hamiltonian)

    return circuit


def run_optimizer(circuit, dev, init_params, optimizer, n_steps, target_energy, tol):
    """Circuit-eval count comes from qml.Tracker (device-level executions), not a
    Python-side call counter: parameter-shift gradients and QNGD's metric-tensor
    estimation both trigger many extra device executions per step that never
    re-invoke the decorated `circuit` function itself, so a counter inside it
    would undercount by ~30x."""
    params = init_params.copy()
    history = []
    steps_to_threshold = None
    t0 = time.perf_counter()
    with qml.Tracker(dev) as tracker:
        for step in range(n_steps):
            params, energy = optimizer.step_and_cost(circuit, params)
            history.append(float(energy))
            if steps_to_threshold is None and energy <= target_energy + tol:
                steps_to_threshold = step + 1
    elapsed = time.perf_counter() - t0
    return {
        "final_energy": history[-1],
        "history": history,
        "circuit_evals": tracker.totals.get("executions", 0),
        "circuit_simulations": tracker.totals.get("simulations", 0),
        "steps_to_threshold": steps_to_threshold,
        "wall_clock_s": elapsed,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", default=str(REPO_ROOT / "external/VQEzy/qmanybody/xyz_4_qubit.h5"))
    ap.add_argument("--sample", default="sample_0")
    ap.add_argument("--n-layers", type=int, default=2)
    ap.add_argument("--n-steps", type=int, default=60)
    ap.add_argument("--gd-stepsize", type=float, default=0.1)
    ap.add_argument("--qngd-stepsize", type=float, default=0.02)
    ap.add_argument("--tol", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    instance = load_instance(args.h5, args.sample)
    hamiltonian, n_qubits = build_hamiltonian(instance)
    vqezy_reference = float(instance.reference_loss_history[-1])
    exact_ground = true_ground_energy(hamiltonian, n_qubits)
    target_energy = exact_ground if exact_ground is not None else vqezy_reference
    p = n_params(n_qubits, args.n_layers)

    print(f"[{instance.family}] {args.sample}: n_qubits={n_qubits}, n_params={p}")
    print(f"VQEzy reference final energy (their optimizer, {len(instance.reference_loss_history)} steps): {vqezy_reference:.6f}")
    if exact_ground is not None:
        print(f"Exact ground energy (diagonalization) -- used as target: {exact_ground:.6f}")
    else:
        print(
            f"n_qubits > {EXACT_DIAG_MAX_QUBITS} -- exact diag skipped, "
            "using VQEzy reference as target (may be under-converged)"
        )

    rng = pnp.random.default_rng(args.seed)
    init = pnp.array(
        rng.uniform(0, 2 * pnp.pi, size=(args.n_layers, n_qubits, 2)), requires_grad=True
    )

    dev = qml.device("lightning.qubit", wires=n_qubits)

    # QNGD needs a much smaller nominal stepsize than plain GD: the inverse-Fisher
    # preconditioning rescales the update per-direction, so the same stepsize that
    # is stable for GD tends to overshoot/diverge under QNGD (verified empirically
    # on this instance -- stepsize=0.1 diverges, 0.02 converges smoothly).
    results = {}
    for name, opt in [
        ("gradient_descent", qml.GradientDescentOptimizer(stepsize=args.gd_stepsize)),
        ("qngd", qml.QNGOptimizer(stepsize=args.qngd_stepsize)),
    ]:
        circuit = _make_qnode(hamiltonian, n_qubits, args.n_layers, dev)
        res = run_optimizer(circuit, dev, init, opt, args.n_steps, target_energy, args.tol)
        results[name] = res
        print(
            f"  {name:>17s}: final_energy={res['final_energy']:.6f}  "
            f"circuit_evals={res['circuit_evals']:5d}  "
            f"steps_to_within_{args.tol}={res['steps_to_threshold']}  "
            f"time={res['wall_clock_s']:.2f}s"
        )

    out_dir = REPO_ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"phase1_baseline_{instance.family}_{args.sample}.json"
    with open(out_path, "w") as f:
        json.dump(
            {
                "instance": {
                    "family": instance.family,
                    "sample": args.sample,
                    "n_qubits": n_qubits,
                    "n_layers": args.n_layers,
                    "vqezy_reference_energy": vqezy_reference,
                    "exact_ground_energy": exact_ground,
                    "target_energy_used": target_energy,
                },
                "results": results,
            },
            f,
            indent=2,
        )
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
