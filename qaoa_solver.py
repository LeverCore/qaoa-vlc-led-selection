from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

try:
    from scipy.optimize import minimize
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False

from config import CONFIG, VLCConfig, get_led_positions, get_receiver_grid
from vlc_model import compute_channel_gain_matrix
from metrics import evaluate_selection_with_metrics, format_metrics_for_print
from qubo_objective import (
    build_qubo_from_vlc_channel,
    build_cost_energy_for_all_basis_states,
    compute_qubo_energy,
    compute_qubo_score,
    index_to_selection,
    selection_to_bitstring,
)

try:
    from plot_utils import save_result_plots
    PLOT_AVAILABLE = True
except ImportError:
    PLOT_AVAILABLE = False


_EPS = 1.0e-30


# ============================================================
# 1. Qiskit QAOA circuit construction
# ============================================================

def split_qaoa_params(
    params: np.ndarray,
    p: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Split QAOA parameters.

        params = [gamma_1, ..., gamma_p, beta_1, ..., beta_p]
    """
    params = np.asarray(params, dtype=float).ravel()

    if params.size != 2 * p:
        raise ValueError(f"Expected {2 * p} QAOA parameters, got {params.size}.")

    gammas = params[:p]
    betas = params[p:]

    return gammas, betas


def append_qubo_cost_layer(
    circuit: QuantumCircuit,
    Q: np.ndarray,
    gamma: float,
    coeff_tol: float = 1.0e-12,
) -> None:
    """
    Append one QUBO cost layer to a Qiskit circuit.

    The QUBO energy is:

        E(x) = x^T Q x
             = sum_i Q_ii x_i + sum_{i<j} (Q_ij + Q_ji) x_i x_j

    Using x_i = (1 - Z_i) / 2, the cost unitary

        exp[-i gamma E(x)]

    can be implemented up to a global phase by RZ and RZZ rotations.

    For a linear term a_i x_i:
        RZ_i(-gamma * a_i)

    For a quadratic term b_ij x_i x_j:
        RZ_i(-gamma * b_ij / 2)
        RZ_j(-gamma * b_ij / 2)
        RZZ_ij(gamma * b_ij / 2)
    """
    Q = np.asarray(Q, dtype=float)

    if Q.ndim != 2 or Q.shape[0] != Q.shape[1]:
        raise ValueError("Q must be a square QUBO matrix.")

    num_qubits = Q.shape[0]

    # Diagonal linear terms.
    for i in range(num_qubits):
        a_i = float(Q[i, i])
        if abs(a_i) > coeff_tol:
            circuit.rz(-gamma * a_i, i)

    # Off-diagonal quadratic terms.
    for i in range(num_qubits):
        for j in range(i + 1, num_qubits):
            b_ij = float(Q[i, j] + Q[j, i])

            if abs(b_ij) <= coeff_tol:
                continue

            circuit.rz(-gamma * b_ij / 2.0, i)
            circuit.rz(-gamma * b_ij / 2.0, j)
            circuit.rzz(gamma * b_ij / 2.0, i, j)


def append_x_mixer_layer(
    circuit: QuantumCircuit,
    beta: float,
) -> None:
    """
    Append one standard X-mixer layer:

        U_M(beta) = exp[-i beta sum_i X_i]

    Since RX(theta) = exp[-i theta X / 2], the required gate is:

        RX(2 beta)
    """
    for qubit in range(circuit.num_qubits):
        circuit.rx(2.0 * beta, qubit)


def build_qaoa_circuit(
    params: np.ndarray,
    Q: np.ndarray,
    p: int = 1,
    measure: bool = False,
) -> QuantumCircuit:
    """
    Build a Qiskit QAOA circuit for the unconstrained LED-selection problem.

    The initial state is |+>^N, so all LED on/off patterns are explored.
    The mixer is the standard X mixer because the number of selected LEDs
    is not fixed.
    """
    if not QISKIT_AVAILABLE:
        raise ImportError(
            "Qiskit is not installed. Please install qiskit before running this file."
        )

    Q = np.asarray(Q, dtype=float)

    if Q.ndim != 2 or Q.shape[0] != Q.shape[1]:
        raise ValueError("Q must be a square QUBO matrix.")

    num_qubits = Q.shape[0]
    gammas, betas = split_qaoa_params(params, p)

    if measure:
        circuit = QuantumCircuit(num_qubits, num_qubits)
    else:
        circuit = QuantumCircuit(num_qubits)

    # Initial state |+>^N.
    for qubit in range(num_qubits):
        circuit.h(qubit)

    # Alternating QAOA layers.
    for layer in range(p):
        append_qubo_cost_layer(
            circuit=circuit,
            Q=Q,
            gamma=float(gammas[layer]),
        )

        append_x_mixer_layer(
            circuit=circuit,
            beta=float(betas[layer]),
        )

    if measure:
        circuit.measure(range(num_qubits), range(num_qubits))

    return circuit


def qaoa_statevector(
    params: np.ndarray,
    Q: np.ndarray,
    p: int = 1,
) -> Statevector:
    """
    Build the final QAOA statevector using Qiskit.
    """
    circuit = build_qaoa_circuit(
        params=params,
        Q=Q,
        p=p,
        measure=False,
    )

    return Statevector.from_instruction(circuit)


def qaoa_expectation(
    params: np.ndarray,
    Q: np.ndarray,
    cost_energies: np.ndarray,
    p: int = 1,
) -> float:
    """
    Compute the expected QUBO energy of the QAOA state.
    """
    state = qaoa_statevector(
        params=params,
        Q=Q,
        p=p,
    )

    probs = np.asarray(state.probabilities(), dtype=float)
    probs = probs / max(float(np.sum(probs)), _EPS)

    return float(np.dot(probs, cost_energies))


# ============================================================
# 2. Optimize QAOA angles
# ============================================================

def random_initial_params(
    p: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generate random initial QAOA parameters.
    """
    gammas = rng.uniform(0.0, 2.0 * np.pi, size=p)
    betas = rng.uniform(0.0, np.pi, size=p)

    return np.concatenate([gammas, betas])


def optimize_qaoa_angles(
    Q: np.ndarray,
    cost_energies: np.ndarray,
    p: int = 1,
    maxiter: int = 60,
    n_starts: int = 5,
    seed: int = 123,
) -> dict:
    """
    Optimize QAOA angles using a classical optimizer.
    """
    rng = np.random.default_rng(seed)

    best_params = None
    best_value = np.inf
    best_info = {}

    def objective(params: np.ndarray) -> float:
        return qaoa_expectation(
            params=params,
            Q=Q,
            cost_energies=cost_energies,
            p=p,
        )

    if SCIPY_AVAILABLE:
        for start_id in range(n_starts):
            x0 = random_initial_params(p, rng)

            res = minimize(
                objective,
                x0,
                method="COBYLA",
                options={
                    "maxiter": maxiter,
                    "rhobeg": 0.5,
                    "tol": 1.0e-6,
                    "disp": False,
                },
            )

            if float(res.fun) < best_value:
                best_value = float(res.fun)
                best_params = np.asarray(res.x, dtype=float)
                best_info = {
                    "optimizer": "COBYLA",
                    "success": bool(res.success),
                    "message": str(res.message),
                    "nfev": int(res.nfev) if hasattr(res, "nfev") else None,
                    "start_id": int(start_id),
                }

    else:
        # Fallback if SciPy is unavailable.
        num_trials = maxiter * n_starts

        for trial in range(num_trials):
            params = random_initial_params(p, rng)
            value = objective(params)

            if value < best_value:
                best_value = float(value)
                best_params = params

        best_info = {
            "optimizer": "random_search",
            "success": True,
            "message": "SciPy is not available; random search was used.",
            "nfev": int(num_trials),
            "start_id": None,
        }

    return {
        "best_params": best_params,
        "best_expectation": float(best_value),
        "optimizer_info": best_info,
    }


# ============================================================
# 3. Sampling and VLC evaluation
# ============================================================

def sample_qiskit_statevector(
    state: Statevector,
    shots: int = 4096,
    seed: int = 123,
) -> np.ndarray:
    """
    Sample computational-basis states from a Qiskit Statevector.

    Returns
    -------
    counts : ndarray, shape (2^N,)
        counts[index] is the number of times basis state |index> appears.
    """
    rng = np.random.default_rng(seed)

    probs = np.asarray(state.probabilities(), dtype=float)
    probs = probs / max(float(np.sum(probs)), _EPS)

    counts = rng.multinomial(shots, probs)

    return counts


def evaluate_sampled_qaoa_solutions(
    counts: np.ndarray,
    Q: np.ndarray,
    H: np.ndarray,
    config: VLCConfig = CONFIG,
) -> list[dict]:
    """
    Evaluate sampled QAOA bitstrings.

    For each sampled bitstring, this function computes:
        1. communication-performance metrics
        2. QUBO energy
        3. QUBO score

    The sampled results are ranked by qubo_score in descending order.
    """
    results = []

    num_qubits = config.num_led_total
    sampled_indices = np.where(counts > 0)[0]
    total_counts = int(np.sum(counts))

    for idx in sampled_indices:
        selection = index_to_selection(
            index=int(idx),
            num_bits=num_qubits,
        )

        result = evaluate_selection_with_metrics(
            selection=selection,
            H=H,
            config=config,
        )

        qubo_energy = compute_qubo_energy(selection, Q)
        qubo_score = -qubo_energy

        result["qaoa_index"] = int(idx)
        result["qaoa_bitstring"] = selection_to_bitstring(selection)
        result["qaoa_count"] = int(counts[idx])
        result["qaoa_probability"] = float(counts[idx] / max(total_counts, 1))

        result["qaoa_qubo_energy"] = float(qubo_energy)
        result["qubo_energy"] = float(qubo_energy)
        result["qubo_score"] = float(qubo_score)

        results.append(result)

    results.sort(
        key=lambda r: (
            r["qubo_score"],
            r["coverage_rate"],
            r["power_uniformity"],
            -r["power_consumption_ratio"],
            r["min_snr_db"],
            r["mean_snr_db"],
        ),
        reverse=True,
    )

    return results


def save_qaoa_sample_results_to_csv(
    results: list[dict],
    output_path: str | Path,
) -> None:
    """
    Save sampled QAOA solutions and their VLC performance metrics.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []

    for rank, r in enumerate(results, start=1):
        rows.append({
            "rank_by_qubo_score": rank,

            "selected_leds": "-".join(map(str, r["selected_leds"])),
            "bitstring": "b_" + r["qaoa_bitstring"],

            "qubo_score": r["qubo_score"],
            "qubo_energy": r["qubo_energy"],
            "qaoa_qubo_energy": r["qaoa_qubo_energy"],

            "num_selected_leds": r["num_selected_leds"],
            "power_consumption_ratio": r["power_consumption_ratio"],

            "qaoa_count": r["qaoa_count"],
            "qaoa_probability": r["qaoa_probability"],

            "coverage_rate": r["coverage_rate"],
            "power_uniformity": r["power_uniformity"],
            "snr_uniformity": r["snr_uniformity"],

            "snr_threshold_db": r["snr_threshold_db"],
            "mean_snr_db": r["mean_snr_db"],
            "min_snr_db": r["min_snr_db"],
            "max_snr_db": r["max_snr_db"],

            "mean_received_power": r["mean_received_power"],
            "min_received_power": r["min_received_power"],
            "max_received_power": r["max_received_power"],
        })

    if not rows:
        return

    with output_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# 4. Main QAOA solver
# ============================================================

def run_qaoa_solver(
    config: VLCConfig = CONFIG,
    p: int = 1,
    maxiter: int = 60,
    n_starts: int = 5,
    shots: int = 4096,
    seed: int = 123,
    tracking_weight: float | None = None,
    power_weight: float | None = None,
    target_power_margin: float | None = None,
    target_power: float | None = None,
    save_csv: bool = True,
    save_plots: bool = True,
    print_circuit: bool = False,
) -> dict:
    """
    Run Qiskit-based QAOA for VLC LED selection.
    """
    if not QISKIT_AVAILABLE:
        raise ImportError(
            "Qiskit is not installed. Please install it first, for example:\n"
            "    pip install qiskit\n"
        )

    print("=" * 100)
    print("Qiskit-based QAOA for indoor VLC LED selection")
    print("=" * 100)
    print(f"Room size: {config.room_length} m × {config.room_width} m × {config.room_height} m")
    print(f"Candidate LEDs: {config.num_led_x} × {config.num_led_y} = {config.num_led_total}")
    print("LED selection mode: variable LED number")
    print(f"QAOA qubits: {config.num_led_total}")
    print(f"QAOA layers p: {p}")
    print(f"Shots: {shots}")
    print("Mixer: standard X mixer")
    print("Constraint: no fixed Hamming-weight constraint")
    print("=" * 100)

    print("\nBuilding VLC channel gain matrix...")

    led_positions = get_led_positions(config)
    _, _, _, receiver_points = get_receiver_grid(config)

    H = compute_channel_gain_matrix(
        config=config,
        led_positions=led_positions,
        receiver_points=receiver_points,
    )

    print(f"Channel gain matrix shape: {H.shape}")

    print("\nBuilding target-power-tracking QUBO objective...")

    Q, qubo_info = build_qubo_from_vlc_channel(
        H=H,
        config=config,
        tracking_weight=tracking_weight,
        power_weight=power_weight,
        target_power_margin=target_power_margin,
        target_power=target_power,
    )

    print("QUBO information:")
    for key, value in qubo_info.items():
        print(f"  {key}: {value}")

    print("\nBuilding diagonal QUBO energies for expectation evaluation...")
    cost_energies = build_cost_energy_for_all_basis_states(
        Q=Q,
        num_qubits=config.num_led_total,
    )

    print("\nOptimizing QAOA parameters...")
    opt_result = optimize_qaoa_angles(
        Q=Q,
        cost_energies=cost_energies,
        p=p,
        maxiter=maxiter,
        n_starts=n_starts,
        seed=seed,
    )

    best_params = opt_result["best_params"]

    print(f"Best expected QUBO energy: {opt_result['best_expectation']:.6f}")
    print("Best QAOA parameters:", best_params)
    print("Optimizer info:", opt_result["optimizer_info"])

    print("\nBuilding final Qiskit QAOA circuit...")
    final_circuit = build_qaoa_circuit(
        params=best_params,
        Q=Q,
        p=p,
        measure=False,
    )

    if print_circuit:
        print(final_circuit)

    print("\nSimulating final QAOA statevector and sampling...")
    final_state = Statevector.from_instruction(final_circuit)

    counts = sample_qiskit_statevector(
        state=final_state,
        shots=shots,
        seed=seed,
    )

    sampled_results = evaluate_sampled_qaoa_solutions(
        counts=counts,
        Q=Q,
        H=H,
        config=config,
    )

    if not sampled_results:
        raise RuntimeError("No QAOA samples were obtained.")

    best_result = sampled_results[0]

    best_result["qaoa_p"] = int(p)
    best_result["qaoa_shots"] = int(shots)
    best_result["qaoa_best_params"] = best_params.tolist()
    best_result["qaoa_best_expected_qubo_energy"] = float(opt_result["best_expectation"])
    best_result["qaoa_optimizer_info"] = opt_result["optimizer_info"]
    best_result["qaoa_qubo_info"] = qubo_info

    print("\nBest QAOA sampled result ranked by QUBO score:")
    print("-" * 100)
    print(format_metrics_for_print(best_result))
    print(f"QAOA bitstring: {best_result['qaoa_bitstring']}")
    print(f"QAOA sampled count: {best_result['qaoa_count']}")
    print(f"QAOA sampled probability: {best_result['qaoa_probability']:.6f}")
    print(f"QAOA QUBO energy: {best_result['qaoa_qubo_energy']:.6f}")

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if save_csv:
        csv_path = output_dir / f"qaoa_qiskit_sampled_results_p{p}.csv"

        save_qaoa_sample_results_to_csv(
            results=sampled_results,
            output_path=csv_path,
        )

        print("\nSampled QAOA results saved to:")
        print(csv_path)

    if save_plots and PLOT_AVAILABLE:
        saved_paths = save_result_plots(
            result=best_result,
            config=config,
            output_dir=output_dir,
            prefix=f"best_qaoa_qiskit_p{p}",
        )

        print("\nFigures saved:")
        for name, path in saved_paths.items():
            print(f"{name}: {path}")

    print("\nQiskit QAOA experiment finished.")

    return best_result


if __name__ == "__main__":
    run_qaoa_solver(
        config=CONFIG,
        p=3,
        maxiter=220,
        n_starts=12,
        shots=8192,
        seed=123,
        tracking_weight=None,
        power_weight=None,
        target_power_margin=None,
        target_power=None,
        save_csv=True,
        save_plots=True,
        print_circuit=False,
    )
