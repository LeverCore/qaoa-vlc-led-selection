# plot_utils.py
# -*- coding: utf-8 -*-
"""
Plot utilities for indoor VLC LED selection simulation.

This file provides:
    1. LED layout plot
    2. SNR heatmap
    3. Received-power heatmap
    4. A wrapper function for saving all plots of one selected result

The code is compatible with the latest model setting:
    - 10 m × 10 m × 3 m room
    - 4 × 4 candidate LED positions, i.e., 16 candidates
    - variable number of selected LEDs
    - objective including coverage, received-power uniformity, and power penalty
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from config import CONFIG, VLCConfig, get_led_positions, get_receiver_grid
from vlc_model import (
    reshape_to_grid,
    selected_indices_to_vector,
    compute_channel_gain_matrix,
)
from metrics import evaluate_selection_with_metrics


def ensure_output_dir(output_dir: str | Path) -> Path:
    """
    Create output directory if it does not exist.

    Parameters
    ----------
    output_dir : str or Path
        Output directory.

    Returns
    -------
    output_dir : Path
        Path object of the output directory.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def get_selection_array(
    result_or_selection,
    config: VLCConfig = CONFIG,
) -> np.ndarray:
    """
    Get binary selection vector from either a result dictionary or an array.

    Parameters
    ----------
    result_or_selection : dict or array-like
        If dict, it should contain "selection" or "selection_vector".
        If array-like, it is directly treated as a selection vector.

    config : VLCConfig
        Simulation configuration.

    Returns
    -------
    selection : ndarray, shape (N_led,)
        Binary LED selection vector.
    """
    if isinstance(result_or_selection, dict):
        if "selection" in result_or_selection:
            selection = result_or_selection["selection"]
        elif "selection_vector" in result_or_selection:
            selection = result_or_selection["selection_vector"]
        else:
            raise KeyError(
                "Result dictionary must contain 'selection' or 'selection_vector'."
            )
    else:
        selection = result_or_selection

    selection = np.asarray(selection, dtype=int).ravel()

    if selection.size != config.num_led_total:
        raise ValueError(
            f"Selection vector length should be {config.num_led_total}, "
            f"but got {selection.size}."
        )

    if not np.all((selection == 0) | (selection == 1)):
        raise ValueError("Selection vector must be binary, containing only 0 and 1.")

    return selection


def plot_led_layout(
    selection,
    config: VLCConfig = CONFIG,
    led_positions: np.ndarray | None = None,
    save_path: str | Path | None = None,
    show: bool = False,
) -> None:
    """
    Plot candidate LED positions and selected LED positions.

    Parameters
    ----------
    selection : array-like, shape (N_led,)
        Binary LED selection vector.

    config : VLCConfig
        Simulation configuration.

    led_positions : ndarray, optional, shape (N_led, 3)
        LED candidate positions. If None, generated from config.

    save_path : str or Path, optional
        Figure saving path. If None, the figure is not saved.

    show : bool
        Whether to display the figure.
    """
    selection = get_selection_array(selection, config)

    if led_positions is None:
        led_positions = get_led_positions(config)

    led_positions = np.asarray(led_positions, dtype=float)

    if led_positions.shape[0] != config.num_led_total:
        raise ValueError(
            f"led_positions should have {config.num_led_total} rows, "
            f"but got {led_positions.shape[0]}."
        )

    fig, ax = plt.subplots(figsize=(6.2, 5.8))

    # Room boundary
    ax.plot(
        [0.0, config.room_length, config.room_length, 0.0, 0.0],
        [0.0, 0.0, config.room_width, config.room_width, 0.0],
        linewidth=1.5,
        color="black",
    )

    # All candidate LED positions
    ax.scatter(
        led_positions[:, 0],
        led_positions[:, 1],
        s=145,
        facecolors="none",
        edgecolors="black",
        linewidths=1.4,
        label="Candidate LEDs",
    )

    # Selected LED positions
    selected_mask = selection == 1
    ax.scatter(
        led_positions[selected_mask, 0],
        led_positions[selected_mask, 1],
        s=175,
        marker="o",
        label="Selected LEDs",
    )

    # LED index labels. Use only numbers to avoid crowding in a 4 × 4 grid.
    label_offset = 0.02 * config.room_width

    for idx, (x, y, _) in enumerate(led_positions, start=1):
        ax.text(
            x,
            y + label_offset,
            f"{idx}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    margin_x = 0.03 * config.room_length
    margin_y = 0.03 * config.room_width

    ax.set_xlim(-margin_x, config.room_length + margin_x)
    ax.set_ylim(-margin_y, config.room_width + margin_y)
    ax.set_aspect("equal", adjustable="box")

    ax.set_xlabel("x position (m)")
    ax.set_ylabel("y position (m)")
    ax.set_title("Selected LED layout")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    ax.legend(loc="upper right")

    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    plt.close(fig)


def plot_snr_heatmap(
    snr_db: np.ndarray,
    selection=None,
    config: VLCConfig = CONFIG,
    led_positions: np.ndarray | None = None,
    save_path: str | Path | None = None,
    show: bool = False,
    title: str = "SNR distribution",
) -> None:
    """
    Plot SNR heatmap on the receiver plane.

    Parameters
    ----------
    snr_db : ndarray, shape (N_grid,)
        SNR values in dB.

    selection : array-like, optional, shape (N_led,)
        Binary LED selection vector. If provided, selected LEDs are marked.

    config : VLCConfig
        Simulation configuration.

    led_positions : ndarray, optional, shape (N_led, 3)
        LED candidate positions.

    save_path : str or Path, optional
        Figure saving path.

    show : bool
        Whether to display the figure.

    title : str
        Figure title.
    """
    snr_grid = reshape_to_grid(snr_db, config)

    if led_positions is None:
        led_positions = get_led_positions(config)

    led_positions = np.asarray(led_positions, dtype=float)

    fig, ax = plt.subplots(figsize=(6.5, 5.6))

    im = ax.imshow(
        snr_grid,
        origin="lower",
        extent=[0.0, config.room_length, 0.0, config.room_width],
        aspect="equal",
    )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("SNR (dB)")

    # Draw SNR threshold contour if it lies inside the data range.
    snr_min = float(np.min(snr_grid))
    snr_max = float(np.max(snr_grid))

    if snr_min <= config.snr_threshold_db <= snr_max:
        contour = ax.contour(
            snr_grid,
            levels=[config.snr_threshold_db],
            origin="lower",
            extent=[0.0, config.room_length, 0.0, config.room_width],
            linewidths=1.2,
        )
        ax.clabel(
            contour,
            fmt={config.snr_threshold_db: f"{config.snr_threshold_db:.1f} dB"},
            fontsize=9,
        )

    # Mark selected LEDs on the ceiling projection.
    if selection is not None:
        selection = get_selection_array(selection, config)
        selected_mask = selection == 1

        ax.scatter(
            led_positions[selected_mask, 0],
            led_positions[selected_mask, 1],
            s=80,
            marker="x",
            linewidths=2.0,
            label="Selected LEDs",
        )

        ax.legend(loc="upper right")

    ax.set_xlabel("x position (m)")
    ax.set_ylabel("y position (m)")
    ax.set_title(title)
    ax.set_xlim(0.0, config.room_length)
    ax.set_ylim(0.0, config.room_width)

    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    plt.close(fig)


def plot_received_power_heatmap(
    received_power: np.ndarray,
    selection=None,
    config: VLCConfig = CONFIG,
    led_positions: np.ndarray | None = None,
    save_path: str | Path | None = None,
    show: bool = False,
    title: str = "Received optical power distribution",
) -> None:
    """
    Plot received optical power heatmap on the receiver plane.

    Parameters
    ----------
    received_power : ndarray, shape (N_grid,)
        Received optical power values.

    selection : array-like, optional, shape (N_led,)
        Binary LED selection vector. If provided, selected LEDs are marked.

    config : VLCConfig
        Simulation configuration.

    led_positions : ndarray, optional, shape (N_led, 3)
        LED candidate positions.

    save_path : str or Path, optional
        Figure saving path.

    show : bool
        Whether to display the figure.

    title : str
        Figure title.
    """
    power_grid = reshape_to_grid(received_power, config)

    if led_positions is None:
        led_positions = get_led_positions(config)

    led_positions = np.asarray(led_positions, dtype=float)

    fig, ax = plt.subplots(figsize=(6.5, 5.6))

    im = ax.imshow(
        power_grid,
        origin="lower",
        extent=[0.0, config.room_length, 0.0, config.room_width],
        aspect="equal",
    )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Received optical power (W)")

    if selection is not None:
        selection = get_selection_array(selection, config)
        selected_mask = selection == 1

        ax.scatter(
            led_positions[selected_mask, 0],
            led_positions[selected_mask, 1],
            s=80,
            marker="x",
            linewidths=2.0,
            label="Selected LEDs",
        )

        ax.legend(loc="upper right")

    ax.set_xlabel("x position (m)")
    ax.set_ylabel("y position (m)")
    ax.set_title(title)
    ax.set_xlim(0.0, config.room_length)
    ax.set_ylim(0.0, config.room_width)

    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    plt.close(fig)


def save_result_plots(
    result: dict,
    config: VLCConfig = CONFIG,
    output_dir: str | Path | None = None,
    prefix: str = "best",
) -> dict:
    """
    Save LED layout, SNR heatmap, and received-power heatmap for one result.

    Parameters
    ----------
    result : dict
        Evaluation result. It should contain:
            - selection or selection_vector
            - snr_db
            - received_power

    config : VLCConfig
        Simulation configuration.

    output_dir : str or Path, optional
        Output directory. If None, config.output_dir is used.

    prefix : str
        Prefix of saved figure filenames.

    Returns
    -------
    paths : dict
        Saved figure paths.
    """
    if output_dir is None:
        output_dir = config.output_dir

    output_dir = ensure_output_dir(output_dir)

    selection = get_selection_array(result, config)

    if "snr_db" not in result:
        raise KeyError("Result dictionary must contain 'snr_db'.")

    if "received_power" not in result:
        raise KeyError("Result dictionary must contain 'received_power'.")

    led_positions = get_led_positions(config)

    layout_path = output_dir / f"{prefix}_led_layout.png"
    snr_path = output_dir / f"{prefix}_snr_heatmap.png"
    power_path = output_dir / f"{prefix}_received_power_heatmap.png"

    plot_led_layout(
        selection=selection,
        config=config,
        led_positions=led_positions,
        save_path=layout_path,
        show=False,
    )

    plot_snr_heatmap(
        snr_db=result["snr_db"],
        selection=selection,
        config=config,
        led_positions=led_positions,
        save_path=snr_path,
        show=False,
        title="SNR distribution of selected LED layout",
    )

    plot_received_power_heatmap(
        received_power=result["received_power"],
        selection=selection,
        config=config,
        led_positions=led_positions,
        save_path=power_path,
        show=False,
        title="Received optical power distribution of selected LED layout",
    )

    paths = {
        "layout": str(layout_path),
        "snr_heatmap": str(snr_path),
        "received_power_heatmap": str(power_path),
    }

    return paths


if __name__ == "__main__":
    # Test plot_utils.py with the current best layout for the 10 m × 10 m room.
    config = CONFIG

    led_positions = get_led_positions(config)
    _, _, _, receiver_points = get_receiver_grid(config)

    H = compute_channel_gain_matrix(
        config=config,
        led_positions=led_positions,
        receiver_points=receiver_points,
    )

    # Current best layout:
    # four corner LEDs plus one near-center LED.
    selection = selected_indices_to_vector(
        [1, 4, 6, 13, 16],
        config=config,
    )

    result = evaluate_selection_with_metrics(
        selection=selection,
        H=H,
        config=config,
    )

    saved_paths = save_result_plots(
        result=result,
        config=config,
        output_dir=config.output_dir,
        prefix="test_best_layout",
    )

    print("Figures saved:")
    for name, path in saved_paths.items():
        print(f"{name}: {path}")
