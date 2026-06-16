from __future__ import annotations

import numpy as np

from config import CONFIG, VLCConfig, get_led_positions, get_receiver_grid
from vlc_model import (
    compute_channel_gain_matrix,
    compute_effective_led_power,
    compute_snr,
    selected_indices_to_vector,
)


_EPS = 1.0e-30


# ============================================================
# 1. Basic bitstring tools
# ============================================================

def index_to_selection(
    index: int,
    num_bits: int,
) -> np.ndarray:
    """
    Convert a computational-basis index to a binary LED selection vector.

    Bit i corresponds to LED i + 1.
    """
    return np.array(
        [(int(index) >> i) & 1 for i in range(num_bits)],
        dtype=int,
    )


def selection_to_index(selection: np.ndarray) -> int:
    """
    Convert a binary LED selection vector to a computational-basis index.
    """
    selection = np.asarray(selection, dtype=int).ravel()

    index = 0
    for i, bit in enumerate(selection):
        if bit == 1:
            index |= 1 << i

    return int(index)


def selection_to_bitstring(selection: np.ndarray) -> str:
    """
    Convert a selection vector to a readable bitstring in LED order.

    Example
    -------
    [1, 0, 1, 0] -> "1010"
    """
    selection = np.asarray(selection, dtype=int).ravel()
    return "".join(map(str, selection.tolist()))


def selected_leds_from_selection(selection: np.ndarray) -> list[int]:
    """
    Convert a binary selection vector to 1-based selected LED indices.
    """
    selection = np.asarray(selection, dtype=int).ravel()
    return (np.where(selection == 1)[0] + 1).tolist()


# ============================================================
# 2. SNR threshold -> target received power
# ============================================================

def _snr_linear_at_received_power(
    received_power: float,
    config: VLCConfig = CONFIG,
) -> float:
    """
    Compute linear SNR for a scalar received optical power using the
    current complete VLC noise model in vlc_model.py.
    """
    received_power_array = np.array([float(received_power)], dtype=float)

    snr_linear, _ = compute_snr(
        received_power=received_power_array,
        config=config,
    )

    return float(np.asarray(snr_linear, dtype=float).ravel()[0])


def compute_required_received_power_from_snr_threshold(
    config: VLCConfig = CONFIG,
    snr_threshold_db: float | None = None,
    max_iter: int = 100,
    tolerance: float = 1.0e-15,
) -> float:
    """
    Compute the received optical power required to reach the SNR threshold.

    This function does not use a simplified noise model. Instead, it calls
    compute_snr() from vlc_model.py, so the result is consistent with the
    current physical model, including background light, bandwidth limitation,
    shot noise, thermal noise, and nonlinear distortion noise.

    Parameters
    ----------
    config : VLCConfig
        Simulation configuration.

    snr_threshold_db : float, optional
        SNR threshold in dB. If None, config.snr_threshold_db is used.

    max_iter : int
        Maximum number of bisection iterations.

    tolerance : float
        Stop tolerance for received power.

    Returns
    -------
    required_power : float
        Received optical power corresponding to the SNR threshold.
    """
    if snr_threshold_db is None:
        snr_threshold_db = float(config.snr_threshold_db)

    target_snr_linear = 10.0 ** (float(snr_threshold_db) / 10.0)

    # Lower bound: no received power.
    low = 0.0

    # Upper bound: increase until the current SNR model reaches the threshold.
    high = 1.0e-12

    for _ in range(200):
        if _snr_linear_at_received_power(high, config) >= target_snr_linear:
            break
        high *= 2.0
    else:
        raise RuntimeError(
            "Failed to find an upper bound for the required received power. "
            "Please check the SNR threshold and physical parameters."
        )

    # Bisection.
    for _ in range(max_iter):
        mid = 0.5 * (low + high)
        snr_mid = _snr_linear_at_received_power(mid, config)

        if snr_mid >= target_snr_linear:
            high = mid
        else:
            low = mid

        if high - low <= tolerance * max(1.0, high):
            break

    return float(high)


# ============================================================
# 3. Build QUBO objective
# ============================================================

def build_qubo_from_vlc_channel(
    H: np.ndarray,
    config: VLCConfig = CONFIG,
    tracking_weight: float | None = None,
    power_weight: float | None = None,
    target_power: float | None = None,
    target_power_margin: float | None = None,
    **unused_kwargs,
) -> tuple[np.ndarray, dict]:
    """
    Build the formal QUBO objective from the VLC channel matrix.

    The minimized objective is:

        E_obj(x) =
            tracking_weight / M * sum_j (P_r,j(x) / P_tar - 1)^2
            + power_weight * sum_i x_i / N

    where:

        P_r,j(x) = sum_i x_i * P_eff * H_ij

    After removing the constant term, the QUBO energy is:

        E_QUBO(x) = x^T Q x

    Notes
    -----
    1. The constant term does not affect QAOA or enumeration ranking.
    2. P_tar is obtained from the SNR threshold through the current
       complete SNR model unless target_power is manually specified.
    3. unused_kwargs is kept only for compatibility with old calls.
       Old parameters such as mean_power_weight and variance_weight are
       ignored by this target-tracking objective.
    """
    H = np.asarray(H, dtype=float)

    if H.ndim != 2:
        raise ValueError("H must be a 2D channel matrix with shape (N_led, N_grid).")

    num_led, num_grid = H.shape

    if num_led != config.num_led_total:
        raise ValueError(
            f"H should have {config.num_led_total} rows, but got {num_led}."
        )

    if tracking_weight is None:
        tracking_weight = float(getattr(config, "tracking_weight", 1.0))

    if power_weight is None:
        power_weight = float(getattr(config, "power_weight", 0.2))

    if target_power_margin is None:
        target_power_margin = float(getattr(config, "target_power_margin", 7.0))

    if target_power_margin <= 0.0:
        raise ValueError("target_power_margin must be positive.")

    required_power = compute_required_received_power_from_snr_threshold(
        config=config,
        snr_threshold_db=float(config.snr_threshold_db),
    )

    if target_power is None:
        target_power = target_power_margin * required_power

    target_power = float(target_power)

    if target_power <= _EPS:
        raise ValueError("target_power must be positive.")

    # Effective LED power includes optional LED nonlinearity.
    effective_led_power = compute_effective_led_power(config)

    # A[i, j] is the received-power contribution of LED i at receiver point j.
    A = effective_led_power * H

    # ------------------------------------------------------------
    # Target-power tracking term
    # ------------------------------------------------------------
    # P_r(x) over all grid points:
    #     p(x) = A^T x
    #
    # 1/M * sum_j (p_j / P_tar - 1)^2
    # =
    # x^T [A A^T / (M P_tar^2)] x
    # - 2 * mean_i(A_i) / P_tar * x_i
    # + 1
    #
    # The constant +1 is omitted in Q because it does not affect ranking.
    quadratic_tracking = (A @ A.T) / (max(num_grid, 1) * target_power ** 2)
    linear_tracking = -2.0 * np.mean(A, axis=1) / target_power

    Q = tracking_weight * quadratic_tracking

    for i in range(num_led):
        # Linear tracking term.
        Q[i, i] += tracking_weight * linear_tracking[i]

        # Normalized LED power-consumption penalty.
        Q[i, i] += power_weight / num_led

    # Symmetrize for numerical safety.
    Q = 0.5 * (Q + Q.T)

    info = {
        "objective_type": "target_power_tracking",
        "num_led": int(num_led),
        "num_grid": int(num_grid),

        "tracking_weight": float(tracking_weight),
        "power_weight": float(power_weight),

        "snr_threshold_db": float(config.snr_threshold_db),
        "required_power": float(required_power),
        "target_power_margin": float(target_power_margin),
        "target_power": float(target_power),

        "effective_led_power": float(effective_led_power),
        "qubo_constant_omitted": float(tracking_weight),
    }

    return Q, info


# ============================================================
# 4. QUBO score and diagnostics
# ============================================================

def compute_qubo_energy(
    selection: np.ndarray,
    Q: np.ndarray,
) -> float:
    """
    Compute QUBO energy:

        E_QUBO(x) = x^T Q x

    Smaller energy means better objective value.
    The constant term of the target-tracking objective is omitted.
    """
    x = np.asarray(selection, dtype=float).ravel()
    Q = np.asarray(Q, dtype=float)

    if Q.ndim != 2 or Q.shape[0] != Q.shape[1]:
        raise ValueError("Q must be a square matrix.")

    if x.size != Q.shape[0]:
        raise ValueError(
            f"Selection length should be {Q.shape[0]}, but got {x.size}."
        )

    return float(x @ Q @ x)


def compute_qubo_score(
    selection: np.ndarray,
    Q: np.ndarray,
) -> float:
    """
    Compute QUBO score for ranking:

        qubo_score = -qubo_energy

    Larger score means a better LED-selection scheme.
    """
    return float(-compute_qubo_energy(selection, Q))


def compute_full_tracking_objective(
    selection: np.ndarray,
    H: np.ndarray,
    config: VLCConfig = CONFIG,
    tracking_weight: float | None = None,
    power_weight: float | None = None,
    target_power: float | None = None,
    target_power_margin: float | None = None,
) -> float:
    """
    Compute the full target-tracking objective including the constant term.

    This function is only for diagnostics and interpretation. QAOA and
    enumeration ranking use x^T Q x, which differs by a constant.
    """
    selection = np.asarray(selection, dtype=float).ravel()
    H = np.asarray(H, dtype=float)

    if tracking_weight is None:
        tracking_weight = float(getattr(config, "tracking_weight", 1.0))

    if power_weight is None:
        power_weight = float(getattr(config, "power_weight", 0.8))

    if target_power_margin is None:
        target_power_margin = float(getattr(config, "target_power_margin", 2.0))

    if target_power is None:
        required_power = compute_required_received_power_from_snr_threshold(config)
        target_power = target_power_margin * required_power

    effective_led_power = compute_effective_led_power(config)
    received_power = selection @ (effective_led_power * H)

    tracking_error = np.mean((received_power / max(float(target_power), _EPS) - 1.0) ** 2)
    normalized_power = np.sum(selection) / config.num_led_total

    objective = (
        tracking_weight * tracking_error
        + power_weight * normalized_power
    )

    return float(objective)


def add_qubo_metrics_to_result(
    result: dict,
    Q: np.ndarray,
) -> dict:
    """
    Add qubo_energy and qubo_score to an existing result dictionary.

    The result dictionary should contain either:
        - "selection"
        - or "selection_vector"
    """
    if "selection" in result:
        selection = result["selection"]
    elif "selection_vector" in result:
        selection = result["selection_vector"]
    else:
        raise KeyError("Result must contain 'selection' or 'selection_vector'.")

    qubo_energy = compute_qubo_energy(selection, Q)
    qubo_score = -qubo_energy

    result["qubo_energy"] = float(qubo_energy)
    result["qubo_score"] = float(qubo_score)

    return result


def build_cost_energy_for_all_basis_states(
    Q: np.ndarray,
    num_qubits: int,
) -> np.ndarray:
    """
    Build diagonal QUBO energy values for all computational basis states.

    This is used by QAOA statevector simulation to compute:

        <E_QUBO> = sum_x P(x) E_QUBO(x)
    """
    num_states = 2 ** num_qubits
    energies = np.zeros(num_states, dtype=float)

    for idx in range(num_states):
        selection = index_to_selection(idx, num_qubits)
        energies[idx] = compute_qubo_energy(selection, Q)

    return energies


# ============================================================
# 5. Simple test
# ============================================================

if __name__ == "__main__":
    config = CONFIG

    led_positions = get_led_positions(config)
    _, _, _, receiver_points = get_receiver_grid(config)

    H = compute_channel_gain_matrix(
        config=config,
        led_positions=led_positions,
        receiver_points=receiver_points,
    )

    Q, info = build_qubo_from_vlc_channel(
        H=H,
        config=config,
    )

    test_cases = [
        [1, 4, 13, 16],
        [1, 4, 6, 13, 16],
        list(range(1, config.num_led_total + 1)),
    ]

    print("Q shape:", Q.shape)
    print("QUBO info:", info)

    for leds in test_cases:
        selection = selected_indices_to_vector(
            leds,
            config=config,
            require_fixed_number=False,
        )

        energy = compute_qubo_energy(selection, Q)
        score = compute_qubo_score(selection, Q)
        full_obj = compute_full_tracking_objective(
            selection=selection,
            H=H,
            config=config,
            target_power=info["target_power"],
        )

        print("-" * 80)
        print("Selected LEDs:", leds)
        print(f"QUBO energy: {energy:.6f}")
        print(f"QUBO score: {score:.6f}")
        print(f"Full objective value: {full_obj:.6f}")
