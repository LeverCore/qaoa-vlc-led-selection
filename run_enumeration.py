from pathlib import Path

from config import (
    CONFIG,
    VLCConfig,
    get_led_positions,
    get_receiver_grid,
)
from enumeration_solver import (
    evaluate_all_selections,
    get_best_result,
    get_top_k_results,
    save_results_to_csv,
)
from metrics import format_metrics_for_print
from plot_utils import save_result_plots
from vlc_model import compute_channel_gain_matrix


def print_config_summary(
    config: VLCConfig = CONFIG,
    include_empty_selection: bool = True,
) -> None:
    """
    Print key simulation parameters.
    """
    print("=" * 100)
    print("Indoor VLC LED selection: exhaustive enumeration")
    print("=" * 100)

    print(f"Room size: {config.room_length} m × {config.room_width} m × {config.room_height} m")
    print(f"Receiver plane height: {config.receiver_height} m")
    print(f"Receiver grid: {config.grid_size_x} × {config.grid_size_y}")

    print(
        f"Candidate LEDs: {config.num_led_x} × {config.num_led_y} "
        f"= {config.num_led_total}"
    )

    if getattr(config, "use_fixed_led_number", False):
        print(f"LED selection mode: fixed, select exactly {config.num_led_select} LEDs")
        expected_num = "C(N, K)"
    else:
        print("LED selection mode: variable, LED number is not fixed")
        expected_num = 2 ** config.num_led_total
        if not include_empty_selection:
            expected_num -= 1

    print(f"Expected enumerated selections: {expected_num}")
    print(f"Include all-off selection: {include_empty_selection}")

    print(f"Nominal LED transmit power: {config.led_power} W")

    if getattr(config, "enable_led_nonlinearity", False):
        print("LED nonlinearity: enabled")
        print(f"LED saturation power: {config.led_saturation_power} W")
        print(f"Nonlinear distortion factor: {config.nonlinear_distortion_factor}")
    else:
        print("LED nonlinearity: disabled")

    print(f"Background current: {getattr(config, 'background_current', 0.0)} A")
    print(f"Communication bandwidth: {getattr(config, 'communication_bandwidth_hz', 0.0)} Hz")
    print(f"System 3-dB bandwidth: {getattr(config, 'system_3db_bandwidth_hz', 0.0)} Hz")

    print(f"SNR threshold: {config.snr_threshold_db} dB")

    print(
        "Objective: "
        f"{config.alpha_coverage} × coverage "
        f"+ {config.beta_uniformity} × power uniformity "
        f"- {config.lambda_power} × normalized power"
    )

    print("=" * 100)


def print_top_results(
    results: list[dict],
    k: int = 10,
) -> None:
    """
    Print top-k enumeration results.
    """
    top_results = get_top_k_results(results, k=k)

    print(f"\nTop {len(top_results)} enumeration results:")
    print("-" * 130)

    for rank, result in enumerate(top_results, start=1):
        print(
            f"Rank {rank:02d} | "
            f"LEDs: {result['selected_leds']} | "
            f"Num LEDs: {result['num_selected_leds']} | "
            f"Power ratio: {result['power_consumption_ratio']:.4f} | "
            f"Score: {result['objective_score']:.6f} | "
            f"Coverage: {result['coverage_rate'] * 100:.2f}% | "
            f"Uniformity: {result['power_uniformity']:.6f} | "
            f"Min SNR: {result['min_snr_db']:.4f} dB | "
            f"Mean SNR: {result['mean_snr_db']:.4f} dB"
        )


def run(
    save_csv: bool = True,
    save_plots: bool = True,
    include_empty_selection: bool = True,
) -> dict:
    """
    Run the full enumeration experiment.

    Returns
    -------
    best_result : dict
        Best enumeration result.
    """
    config = CONFIG

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print_config_summary(
        config=config,
        include_empty_selection=include_empty_selection,
    )

    print("\nBuilding VLC channel gain matrix...")

    led_positions = get_led_positions(config)
    _, _, _, receiver_points = get_receiver_grid(config)

    H = compute_channel_gain_matrix(
        config=config,
        led_positions=led_positions,
        receiver_points=receiver_points,
    )

    print(f"Channel gain matrix shape: {H.shape}")

    print("\nEnumerating LED selections...")

    results = evaluate_all_selections(
        config=config,
        H=H,
        include_empty_selection=include_empty_selection,
    )

    print(f"Total enumerated selections: {len(results)}")

    best_result = get_best_result(results)

    print("\nBest enumeration result:")
    print("-" * 100)
    print(format_metrics_for_print(best_result))

    print_top_results(results, k=10)

    if save_csv:
        csv_path = output_dir / "enumeration_results.csv"
        save_results_to_csv(results, csv_path)

        print("\nEnumeration results saved to:")
        print(csv_path)

    if save_plots:
        saved_paths = save_result_plots(
            result=best_result,
            config=config,
            output_dir=output_dir,
            prefix="best_enumeration",
        )

        print("\nFigures saved:")
        for name, path in saved_paths.items():
            print(f"{name}: {path}")

    print("\nEnumeration experiment finished.")

    return best_result


if __name__ == "__main__":
    run(
        save_csv=True,
        save_plots=True,
        include_empty_selection=True,
    )