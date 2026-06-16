import csv
import itertools
from pathlib import Path

import numpy as np

from config import CONFIG, VLCConfig, get_led_positions, get_receiver_grid
from vlc_model import compute_channel_gain_matrix
from metrics import evaluate_selection_with_metrics, format_metrics_for_print
from qubo_objective import build_qubo_from_vlc_channel, add_qubo_metrics_to_result


def index_to_selection_vector(
    index: int,
    config: VLCConfig = CONFIG,
) -> np.ndarray:
    """
    Convert an integer index to a binary LED selection vector.

    Example for N = 4:
        index = 5 -> binary 0101 in little-endian form
        selection = [1, 0, 1, 0]

    Here selection[i] = 1 means LED_(i+1) is selected.
    """
    selection = np.array(
        [(index >> i) & 1 for i in range(config.num_led_total)],
        dtype=int,
    )

    return selection


def combination_to_selection_vector(
    combination: tuple[int, ...],
    config: VLCConfig = CONFIG,
) -> np.ndarray:
    """
    Convert one LED-index combination to a binary selection vector.

    This function is only used when config.use_fixed_led_number = True.
    """
    selection = np.zeros(config.num_led_total, dtype=int)
    selection[list(combination)] = 1

    return selection


def generate_all_selection_vectors(
    config: VLCConfig = CONFIG,
    include_empty_selection: bool = True,
) -> list[np.ndarray]:
    """
    Generate all LED selection vectors.

    If config.use_fixed_led_number = False:
        enumerate all 2^N binary LED selections.

    If config.use_fixed_led_number = True:
        enumerate all selections satisfying sum_i x_i = config.num_led_select.

    Parameters
    ----------
    config : VLCConfig
        Simulation configuration.

    include_empty_selection : bool
        Whether to include the all-off selection [0, 0, ..., 0].
        For unconstrained QAOA comparison, keeping it True is more consistent.
        If you want to force at least one LED to be used, set it to False.

    Returns
    -------
    selections : list of ndarray
        All binary LED selection vectors.
    """
    selections = []

    if getattr(config, "use_fixed_led_number", False):
        all_combinations = itertools.combinations(
            range(config.num_led_total),
            config.num_led_select,
        )

        for combination in all_combinations:
            selection = combination_to_selection_vector(combination, config)
            selections.append(selection)

    else:
        num_states = 2 ** config.num_led_total

        start_index = 0 if include_empty_selection else 1

        for index in range(start_index, num_states):
            selection = index_to_selection_vector(index, config)
            selections.append(selection)

    return selections


def evaluate_all_selections(
    config: VLCConfig = CONFIG,
    H: np.ndarray | None = None,
    Q: np.ndarray | None = None,
    include_empty_selection: bool = True,
) -> list[dict]:
    """
    Evaluate all LED selections.

    For each selection, this function computes:
        1. Communication-performance metrics
        2. QUBO energy
        3. QUBO score

    The ranking score is:

        qubo_score = - x^T Q x
    """
    if H is None:
        led_positions = get_led_positions(config)
        _, _, _, receiver_points = get_receiver_grid(config)

        H = compute_channel_gain_matrix(
            config=config,
            led_positions=led_positions,
            receiver_points=receiver_points,
        )

    if Q is None:
        Q, _ = build_qubo_from_vlc_channel(
            H=H,
            config=config,
        )

    selections = generate_all_selection_vectors(
        config=config,
        include_empty_selection=include_empty_selection,
    )

    results = []

    for selection in selections:
        result = evaluate_selection_with_metrics(
            selection=selection,
            H=H,
            config=config,
        )

        result = add_qubo_metrics_to_result(
            result=result,
            Q=Q,
        )

        results.append(result)

    return results


def sort_results(results: list[dict]) -> list[dict]:
    """
    Sort enumeration results.

    Sorting priority:
        1. qubo_score, descending
        2. coverage_rate, descending
        3. power_uniformity, descending
        4. power_consumption_ratio, ascending
        5. min_snr_db, descending
        6. mean_snr_db, descending
    """
    sorted_results = sorted(
        results,
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

    return sorted_results


def get_best_result(results: list[dict]) -> dict:
    """
    Get the best LED selection result.
    """
    if len(results) == 0:
        raise ValueError("The result list is empty.")

    sorted_results = sort_results(results)
    best_result = sorted_results[0]

    return best_result


def get_top_k_results(
    results: list[dict],
    k: int = 10,
) -> list[dict]:
    """
    Get the top-k LED selection results.
    """
    sorted_results = sort_results(results)
    return sorted_results[:k]


def build_csv_row(result: dict, rank: int) -> dict:
    """
    Build one CSV row from an evaluation result.

    Large arrays such as received_power, snr_linear, and snr_db are not saved.
    Only scalar metrics and LED-selection information are saved.
    """
    row = {
        "rank": rank,

        "selected_leds": "-".join(map(str, result["selected_leds"])),
        "selection_vector": "b_" + "".join(map(str, result["selection_vector"])),

        "qubo_score": result["qubo_score"],
        "qubo_energy": result["qubo_energy"],

        "num_selected_leds": result["num_selected_leds"],
        "power_consumption_ratio": result["power_consumption_ratio"],

        "coverage_rate": result["coverage_rate"],
        "power_uniformity": result["power_uniformity"],
        "snr_uniformity": result["snr_uniformity"],

        "snr_threshold_db": result["snr_threshold_db"],
        "mean_snr_db": result["mean_snr_db"],
        "min_snr_db": result["min_snr_db"],
        "max_snr_db": result["max_snr_db"],

        "mean_received_power": result["mean_received_power"],
        "min_received_power": result["min_received_power"],
        "max_received_power": result["max_received_power"],
    }

    return row


def save_results_to_csv(
    results: list[dict],
    output_path: str | Path,
) -> None:
    """
    Save sorted enumeration results to a CSV file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sorted_results = sort_results(results)

    rows = [
        build_csv_row(result, rank=rank)
        for rank, result in enumerate(sorted_results, start=1)
    ]

    if len(rows) == 0:
        raise ValueError("No results to save.")

    fieldnames = list(rows[0].keys())

    with output_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_top_k_results(
    results: list[dict],
    k: int = 10,
) -> None:
    """
    Print top-k enumeration results.
    """
    top_results = get_top_k_results(results, k=k)

    print(f"\nTop {len(top_results)} LED selections:")
    print("-" * 140)

    for rank, result in enumerate(top_results, start=1):
        print(
            f"Rank {rank:02d} | "
            f"LEDs: {result['selected_leds']} | "
            f"Num LEDs: {result['num_selected_leds']} | "
            f"Power ratio: {result['power_consumption_ratio']:.4f} | "
            f"QUBO score: {result['qubo_score']:.6f} | "
            f"QUBO energy: {result['qubo_energy']:.6f} | "
            f"Coverage: {result['coverage_rate'] * 100:.2f}% | "
            f"Uniformity: {result['power_uniformity']:.6f} | "
            f"Min SNR: {result['min_snr_db']:.4f} dB | "
            f"Mean SNR: {result['mean_snr_db']:.4f} dB"
        )


def run_enumeration_solver(
    config: VLCConfig = CONFIG,
    save_csv: bool = True,
    include_empty_selection: bool = True,
) -> dict:
    """
    Run the exhaustive enumeration solver.

    The best result is selected by maximizing:

        qubo_score = - x^T Q x
    """
    print("Building VLC channel gain matrix...")

    led_positions = get_led_positions(config)
    _, _, _, receiver_points = get_receiver_grid(config)

    H = compute_channel_gain_matrix(
        config=config,
        led_positions=led_positions,
        receiver_points=receiver_points,
    )

    print("Channel gain matrix shape:", H.shape)

    print("\nBuilding QUBO objective matrix...")

    Q, qubo_info = build_qubo_from_vlc_channel(
        H=H,
        config=config,
    )

    print("QUBO objective information:")
    for key, value in qubo_info.items():
        print(f"  {key}: {value}")

    print("\nEnumerating LED selections...")

    if getattr(config, "use_fixed_led_number", False):
        print(
            f"Mode: fixed LED number, select exactly {config.num_led_select} LEDs."
        )
    else:
        print(
            f"Mode: variable LED number, enumerate all 2^{config.num_led_total} selections."
        )

    results = evaluate_all_selections(
        config=config,
        H=H,
        Q=Q,
        include_empty_selection=include_empty_selection,
    )

    print(f"Total enumerated selections: {len(results)}")

    best_result = get_best_result(results)

    print("\nBest enumeration result:")
    print("-" * 100)
    print(format_metrics_for_print(best_result))

    print_top_k_results(results, k=10)

    if save_csv:
        output_dir = Path(config.output_dir)
        csv_path = output_dir / "enumeration_results.csv"

        save_results_to_csv(results, csv_path)

        print("\nAll enumeration results have been saved to:")
        print(csv_path)

    return best_result


if __name__ == "__main__":
    run_enumeration_solver(
        CONFIG,
        save_csv=True,
        include_empty_selection=True,
    )