import numpy as np

from config import VLCConfig, CONFIG
from vlc_model import evaluate_led_selection


_EPS = 1e-30


def compute_coverage_rate(
    snr_db: np.ndarray,
    config: VLCConfig = CONFIG,
) -> float:
    """
    Compute communication coverage rate.

    A receiver grid point is regarded as covered if:

        SNR_j,dB >= SNR_threshold

    Coverage rate:

        C = number of covered grid points / total number of grid points

    Parameters
    ----------
    snr_db : ndarray, shape (N_grid,)
        SNR values in dB on all receiver grid points.

    config : VLCConfig
        Simulation configuration.

    Returns
    -------
    coverage_rate : float
        Coverage rate in [0, 1].
    """
    snr_db = np.asarray(snr_db, dtype=float).ravel()

    covered = snr_db >= config.snr_threshold_db
    coverage_rate = np.mean(covered)

    return float(coverage_rate)


def compute_power_uniformity(received_power: np.ndarray) -> float:
    """
    Compute received-power uniformity.

    The received-power uniformity is defined as:

        U_P = min(P_r,j) / mean(P_r,j)

    A larger value means that the received optical power is more uniformly
    distributed over the working plane.

    Parameters
    ----------
    received_power : ndarray, shape (N_grid,)
        Received optical power on all receiver grid points.

    Returns
    -------
    uniformity : float
        Power uniformity.
    """
    received_power = np.asarray(received_power, dtype=float).ravel()

    mean_power = np.mean(received_power)
    min_power = np.min(received_power)

    if mean_power <= _EPS:
        return 0.0

    uniformity = min_power / mean_power

    return float(uniformity)


def compute_snr_uniformity(snr_linear: np.ndarray) -> float:
    """
    Compute SNR uniformity in linear scale.

    The SNR uniformity is defined as:

        U_SNR = min(SNR_j) / mean(SNR_j)

    Parameters
    ----------
    snr_linear : ndarray, shape (N_grid,)
        Linear SNR values on all receiver grid points.

    Returns
    -------
    uniformity : float
        SNR uniformity.
    """
    snr_linear = np.asarray(snr_linear, dtype=float).ravel()

    mean_snr = np.mean(snr_linear)
    min_snr = np.min(snr_linear)

    if mean_snr <= _EPS:
        return 0.0

    uniformity = min_snr / mean_snr

    return float(uniformity)


def compute_snr_statistics(
    snr_linear: np.ndarray,
    snr_db: np.ndarray,
) -> dict:
    """
    Compute basic SNR statistics.

    Notes
    -----
    Two average SNR values are provided:

    1. mean_snr_db_from_linear:
       First average the linear SNR, then convert it to dB.

           10 log10(mean(SNR_linear))

       This is more physically meaningful.

    2. mean_snr_db_direct:
       Directly average SNR values in dB.

           mean(SNR_dB)

       This is useful for observing the average level on a dB heatmap.

    Parameters
    ----------
    snr_linear : ndarray, shape (N_grid,)
        Linear SNR values.

    snr_db : ndarray, shape (N_grid,)
        SNR values in dB.

    Returns
    -------
    stats : dict
        SNR statistics.
    """
    snr_linear = np.asarray(snr_linear, dtype=float).ravel()
    snr_db = np.asarray(snr_db, dtype=float).ravel()

    mean_snr_linear = np.mean(snr_linear)
    min_snr_linear = np.min(snr_linear)
    max_snr_linear = np.max(snr_linear)

    mean_snr_db = 10.0 * np.log10(max(mean_snr_linear, _EPS))

    stats = {
        "mean_snr_linear": float(mean_snr_linear),
        "min_snr_linear": float(min_snr_linear),
        "max_snr_linear": float(max_snr_linear),

        "mean_snr_db": float(mean_snr_db),
        "min_snr_db": float(np.min(snr_db)),
        "max_snr_db": float(np.max(snr_db)),
    }

    return stats


def compute_power_statistics(received_power: np.ndarray) -> dict:
    """
    Compute basic received-power statistics.

    Parameters
    ----------
    received_power : ndarray, shape (N_grid,)
        Received optical power on all receiver grid points.

    Returns
    -------
    stats : dict
        Received-power statistics.
    """
    received_power = np.asarray(received_power, dtype=float).ravel()

    stats = {
        "mean_received_power": float(np.mean(received_power)),
        "min_received_power": float(np.min(received_power)),
        "max_received_power": float(np.max(received_power)),
    }

    return stats

def compute_power_consumption_ratio(
    selection: np.ndarray,
    config: VLCConfig = CONFIG,
) -> float:
    """
    Compute normalized LED power consumption.

    Since each selected LED has the same transmit power, the normalized
    power consumption is defined as:

        P_norm = number of selected LEDs / total candidate LEDs
    """
    selection = np.asarray(selection, dtype=int).ravel()

    num_selected = int(np.sum(selection))
    power_consumption_ratio = num_selected / config.num_led_total

    return float(power_consumption_ratio)


def selection_to_led_indices(selection: np.ndarray) -> list[int]:
    """
    Convert binary LED selection vector to 1-based LED indices.

    Example
    -------
    selection = [1, 0, 1, 0, 0, 0, 1, 0, 1]

    returns:

    [1, 3, 7, 9]

    Parameters
    ----------
    selection : ndarray, shape (N_led,)
        Binary LED selection vector.

    Returns
    -------
    indices : list[int]
        Selected LED indices using 1-based numbering.
    """
    selection = np.asarray(selection, dtype=int).ravel()
    indices = np.where(selection == 1)[0] + 1

    return indices.tolist()


def compute_all_metrics(
    selection: np.ndarray,
    received_power: np.ndarray,
    snr_linear: np.ndarray,
    snr_db: np.ndarray,
    config: VLCConfig = CONFIG,
) -> dict:
    """
    Compute all performance metrics for one LED selection.

    These metrics are used only for communication-performance evaluation.
    The formal optimization score is qubo_score, which is added later
    by qubo_objective.py.
    """
    coverage_rate = compute_coverage_rate(snr_db, config)
    power_uniformity = compute_power_uniformity(received_power)
    snr_uniformity = compute_snr_uniformity(snr_linear)
    power_consumption_ratio = compute_power_consumption_ratio(selection, config)
    num_selected_leds = int(np.sum(selection))

    power_stats = compute_power_statistics(received_power)
    snr_stats = compute_snr_statistics(snr_linear, snr_db)

    metrics = {
        "selected_leds": selection_to_led_indices(selection),
        "selection_vector": np.asarray(selection, dtype=int).tolist(),

        "num_selected_leds": num_selected_leds,
        "power_consumption_ratio": float(power_consumption_ratio),

        "coverage_rate": float(coverage_rate),
        "power_uniformity": float(power_uniformity),
        "snr_uniformity": float(snr_uniformity),

        "snr_threshold_db": float(config.snr_threshold_db),
    }

    metrics.update(power_stats)
    metrics.update(snr_stats)

    return metrics


def evaluate_selection_with_metrics(
    selection: np.ndarray,
    H: np.ndarray | None = None,
    config: VLCConfig = CONFIG,
) -> dict:
    """
    Evaluate one LED selection and return both physical-layer results
    and communication-performance metrics.

    This function calls evaluate_led_selection() in vlc_model.py first,
    then calculates coverage, uniformity, SNR statistics, and power statistics.
    The formal optimization score is added separately by qubo_objective.py.

    Parameters
    ----------
    selection : ndarray, shape (N_led,)
        Binary LED selection vector.

    H : ndarray, optional, shape (N_led, N_grid)
        Channel gain matrix. If None, it will be computed automatically.

    config : VLCConfig
        Simulation configuration.

    Returns
    -------
    result : dict
        Dictionary containing physical-layer results and metrics.
    """
    physical_result = evaluate_led_selection(
        selection=selection,
        config=config,
        H=H,
    )

    metrics = compute_all_metrics(
        selection=physical_result["selection"],
        received_power=physical_result["received_power"],
        snr_linear=physical_result["snr_linear"],
        snr_db=physical_result["snr_db"],
        config=config,
    )

    result = {}
    result.update(physical_result)
    result.update(metrics)

    return result


def format_metrics_for_print(metrics: dict) -> str:
    """
    Format metrics into readable text.
    """
    lines = [
        f"Selected LEDs: {metrics['selected_leds']}",
        f"Number of selected LEDs: {metrics['num_selected_leds']}",
        f"Normalized power consumption: {metrics['power_consumption_ratio']:.6f}",
    ]

    if "qubo_score" in metrics:
        lines.append(f"QUBO score: {metrics['qubo_score']:.6f}")

    if "qubo_energy" in metrics:
        lines.append(f"QUBO energy: {metrics['qubo_energy']:.6f}")

    if "qaoa_qubo_energy" in metrics:
        lines.append(f"QAOA QUBO energy: {metrics['qaoa_qubo_energy']:.6f}")

    lines.extend([
        f"Coverage rate: {metrics['coverage_rate'] * 100:.2f}%",
        f"Power uniformity: {metrics['power_uniformity']:.6f}",
        f"SNR uniformity: {metrics['snr_uniformity']:.6f}",
        f"Mean SNR: {metrics['mean_snr_db']:.4f} dB",
        f"Minimum SNR: {metrics['min_snr_db']:.4f} dB",
        f"Maximum SNR: {metrics['max_snr_db']:.4f} dB",
        f"Mean received power: {metrics['mean_received_power']:.6e} W",
        f"Minimum received power: {metrics['min_received_power']:.6e} W",
        f"Maximum received power: {metrics['max_received_power']:.6e} W",
    ])

    return "\n".join(lines)


if __name__ == "__main__":
    # Simple test for metrics.py
    from config import get_led_positions, get_receiver_grid
    from vlc_model import (
        compute_channel_gain_matrix,
        selected_indices_to_vector,
    )

    config = CONFIG

    led_positions = get_led_positions(config)
    _, _, _, receiver_points = get_receiver_grid(config)

    H = compute_channel_gain_matrix(
        config=config,
        led_positions=led_positions,
        receiver_points=receiver_points,
    )

    # Example: four-corner LED layout
    selection = selected_indices_to_vector([1, 4, 13, 16], config=config)

    result = evaluate_selection_with_metrics(
        selection=selection,
        H=H,
        config=config,
    )

    print(format_metrics_for_print(result))