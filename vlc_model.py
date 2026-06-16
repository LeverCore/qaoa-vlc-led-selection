import numpy as np

from config import (
    VLCConfig,
    CONFIG,
    get_lambertian_order,
    get_led_positions,
    get_receiver_grid,
)


_EPS = 1e-30

def compute_effective_led_power(config: VLCConfig) -> float:
    """
    Compute the effective optical power of one selected LED.

    If LED nonlinearity is enabled, a simple saturation model is used:

        P_eff = P_sat * tanh(P_t / P_sat)

    where P_t is the nominal optical transmit power and P_sat is the
    saturation power.

    When LED nonlinearity is disabled:

        P_eff = P_t
    """
    if not getattr(config, "enable_led_nonlinearity", False):
        return float(config.led_power)

    p_sat = max(float(config.led_saturation_power), _EPS)
    p_eff = p_sat * np.tanh(config.led_power / p_sat)

    return float(p_eff)


def compute_bandwidth_signal_gain(config: VLCConfig) -> float:
    """
    Compute the signal attenuation caused by finite system bandwidth.

    A first-order low-pass response is used:

        G_B = 1 / sqrt(1 + (B / f_3dB)^2)

    where B is the communication bandwidth and f_3dB is the equivalent
    3-dB bandwidth of the LED-receiver system.
    """
    bandwidth = max(float(config.communication_bandwidth_hz), 0.0)
    f_3db = max(float(config.system_3db_bandwidth_hz), _EPS)

    gain = 1.0 / np.sqrt(1.0 + (bandwidth / f_3db) ** 2)

    return float(gain)


def compute_total_noise_variance(
    dc_signal_current: np.ndarray,
    useful_signal_current: np.ndarray,
    config: VLCConfig,
) -> np.ndarray:
    """
    Compute total noise variance in current domain.

    The total noise includes:
        1. Receiver noise floor
        2. Thermal noise
        3. Shot noise induced by signal and background light
        4. Nonlinear distortion noise
    """
    dc_signal_current = np.asarray(dc_signal_current, dtype=float)
    useful_signal_current = np.asarray(useful_signal_current, dtype=float)

    bandwidth = max(float(config.communication_bandwidth_hz), 0.0)

    # 1. Receiver noise floor
    receiver_noise = float(config.noise_variance)

    # 2. Thermal noise: 4 k_B T B / R_L
    thermal_noise = (
        4.0
        * config.boltzmann_constant
        * config.temperature_k
        * bandwidth
        / config.load_resistance_ohm
    )

    # 3. Shot noise: 2 q (I_signal + I_background) B
    background_current = max(float(config.background_current), 0.0)

    total_dc_current = np.maximum(
        dc_signal_current + background_current,
        0.0,
    )

    shot_noise = (
        2.0
        * config.electron_charge
        * total_dc_current
        * bandwidth
    )

    # 4. LED nonlinear distortion noise
    distortion_factor = max(float(config.nonlinear_distortion_factor), 0.0)

    nonlinear_noise = (
        distortion_factor * useful_signal_current
    ) ** 2

    total_noise = (
        receiver_noise
        + thermal_noise
        + shot_noise
        + nonlinear_noise
    )

    return np.maximum(total_noise, _EPS)


def compute_concentrator_gain(config: VLCConfig) -> float:
    """
    Compute optical concentrator gain.

    Standard VLC model:
        g(psi) = n^2 / sin^2(Psi_c),  0 <= psi <= Psi_c

    Parameters
    ----------
    config : VLCConfig
        Simulation configuration.

    Returns
    -------
    g : float
        Optical concentrator gain.
    """
    fov_rad = np.deg2rad(config.fov_deg)

    if fov_rad <= 0.0 or fov_rad >= np.pi / 2 + 1e-12:
        raise ValueError("FOV should be in the range (0, 90] degrees.")

    g = config.refractive_index ** 2 / (np.sin(fov_rad) ** 2)
    return float(g)


def compute_channel_gain_matrix(
    config: VLCConfig = CONFIG,
    led_positions: np.ndarray | None = None,
    receiver_points: np.ndarray | None = None,
) -> np.ndarray:
    """
    Compute LOS DC channel gain matrix H.

    H[i, j] is the channel gain from LED i to receiver point j.

    The adopted LOS VLC channel model is:

        H_ij = ((m + 1) A) / (2 pi d_ij^2)
               * cos^m(phi_ij)
               * T_s(psi_ij)
               * g(psi_ij)
               * cos(psi_ij)

    when 0 <= psi_ij <= Psi_c. Otherwise, H_ij = 0.

    In this simplified indoor setting:
        - LEDs point vertically downward.
        - The receiver points vertically upward.
        - Therefore, phi_ij = psi_ij.

    Parameters
    ----------
    config : VLCConfig
        Simulation configuration.

    led_positions : ndarray, optional, shape (N_led, 3)
        Candidate LED positions. If None, generated from config.

    receiver_points : ndarray, optional, shape (N_grid, 3)
        Receiver grid points. If None, generated from config.

    Returns
    -------
    H : ndarray, shape (N_led, N_grid)
        LOS channel gain matrix.
    """
    if led_positions is None:
        led_positions = get_led_positions(config)

    if receiver_points is None:
        _, _, _, receiver_points = get_receiver_grid(config)

    led_positions = np.asarray(led_positions, dtype=float)
    receiver_points = np.asarray(receiver_points, dtype=float)

    if led_positions.ndim != 2 or led_positions.shape[1] != 3:
        raise ValueError("led_positions must have shape (N_led, 3).")

    if receiver_points.ndim != 2 or receiver_points.shape[1] != 3:
        raise ValueError("receiver_points must have shape (N_grid, 3).")

    # Shape: (N_led, N_grid, 3)
    diff = led_positions[:, None, :] - receiver_points[None, :, :]

    # Distance d_ij
    distance = np.linalg.norm(diff, axis=2)

    # Vertical separation between LED and receiver plane
    dz = led_positions[:, None, 2] - receiver_points[None, :, 2]

    # cos(phi_ij) and cos(psi_ij)
    # For downward LED and upward receiver, both are dz / d_ij.
    cos_phi = dz / np.maximum(distance, _EPS)
    cos_psi = cos_phi.copy()

    # Incident angle psi_ij
    psi_rad = np.arccos(np.clip(cos_psi, -1.0, 1.0))
    fov_rad = np.deg2rad(config.fov_deg)

    # Valid LOS links within receiver FOV
    valid = (
        (distance > 0.0)
        & (cos_phi > 0.0)
        & (cos_psi > 0.0)
        & (psi_rad <= fov_rad)
    )

    m = get_lambertian_order(config)
    g = compute_concentrator_gain(config)

    H = np.zeros_like(distance, dtype=float)

    H[valid] = (
        ((m + 1.0) * config.pd_area)
        / (2.0 * np.pi * distance[valid] ** 2)
        * (cos_phi[valid] ** m)
        * config.optical_filter_gain
        * g
        * cos_psi[valid]
    )

    return H


def validate_selection_vector(
    selection: np.ndarray,
    config: VLCConfig = CONFIG,
    require_fixed_number: bool | None = None,
) -> np.ndarray:
    """
    Validate LED selection vector.

    Parameters
    ----------
    selection : ndarray, shape (N_led,)
        Binary vector. selection[i] = 1 means LED i is selected.

    config : VLCConfig
        Simulation configuration.

    require_fixed_number : bool or None
        If True, require sum(selection) = config.num_led_select.
        If False, do not constrain the number of selected LEDs.
        If None, use config.use_fixed_led_number.

    Returns
    -------
    selection : ndarray, shape (N_led,)
        Validated binary selection vector.
    """
    selection = np.asarray(selection, dtype=int).ravel()

    if selection.size != config.num_led_total:
        raise ValueError(
            f"Selection vector length should be {config.num_led_total}, "
            f"but got {selection.size}."
        )

    if not np.all((selection == 0) | (selection == 1)):
        raise ValueError("Selection vector must be binary, containing only 0 and 1.")

    if require_fixed_number is None:
        require_fixed_number = getattr(config, "use_fixed_led_number", True)

    if require_fixed_number:
        if int(np.sum(selection)) != config.num_led_select:
            raise ValueError(
                f"Exactly {config.num_led_select} LEDs should be selected, "
                f"but got {int(np.sum(selection))}."
            )

    return selection


def selected_indices_to_vector(
    selected_indices,
    config: VLCConfig = CONFIG,
    one_based: bool = True,
    require_fixed_number: bool | None = None,
) -> np.ndarray:
    """
    Convert selected LED indices to a binary selection vector.

    Parameters
    ----------
    selected_indices : list, tuple, or ndarray
        Selected LED indices.

        If one_based=True:
            [1, 4, 13, 16] means LED_1, LED_4, LED_13, LED_16.

        If one_based=False:
            [0, 3, 12, 15] means LED_1, LED_4, LED_13, LED_16.

    config : VLCConfig
        Simulation configuration.

    one_based : bool
        Whether selected_indices uses 1-based LED numbering.

    require_fixed_number : bool or None
        If True, require exactly config.num_led_select LEDs.
        If False, allow any number of selected LEDs.
        If None, use config.use_fixed_led_number.

    Returns
    -------
    selection : ndarray, shape (N_led,)
        Binary selection vector.
    """
    indices = np.asarray(list(selected_indices), dtype=int)

    if one_based:
        indices = indices - 1

    if require_fixed_number is None:
        require_fixed_number = getattr(config, "use_fixed_led_number", True)

    if require_fixed_number and indices.size != config.num_led_select:
        raise ValueError(
            f"Exactly {config.num_led_select} LED indices should be given, "
            f"but got {indices.size}."
        )

    if np.any(indices < 0) or np.any(indices >= config.num_led_total):
        if one_based:
            raise ValueError(
                f"LED indices should be within 1 to {config.num_led_total}."
            )
        else:
            raise ValueError(
                f"LED indices should be within 0 to {config.num_led_total - 1}."
            )

    if len(np.unique(indices)) != indices.size:
        raise ValueError("Selected LED indices should not contain duplicates.")

    selection = np.zeros(config.num_led_total, dtype=int)
    selection[indices] = 1

    return selection


def compute_received_power(
    selection: np.ndarray,
    H: np.ndarray,
    config: VLCConfig = CONFIG,
) -> np.ndarray:
    """
    Compute received optical power on all receiver grid points.

    No-interference model:
        P_r,j = sum_i x_i * P_t * H_ij

    Parameters
    ----------
    selection : ndarray, shape (N_led,)
        Binary LED selection vector.

    H : ndarray, shape (N_led, N_grid)
        Channel gain matrix.

    config : VLCConfig
        Simulation configuration.

    Returns
    -------
    received_power : ndarray, shape (N_grid,)
        Received optical power at each receiver grid point.
    """
    selection = validate_selection_vector(
        selection,
        config=config,
        require_fixed_number=None,
    )

    H = np.asarray(H, dtype=float)

    if H.ndim != 2:
        raise ValueError("H must be a 2D matrix with shape (N_led, N_grid).")

    if H.shape[0] != config.num_led_total:
        raise ValueError(
            f"H should have {config.num_led_total} rows, "
            f"but got {H.shape[0]}."
        )

    # Each selected LED has the same equivalent optical transmit power.
    effective_led_power = compute_effective_led_power(config)

    led_power_vector = selection * effective_led_power

    # Shape: (N_grid,)
    received_power = led_power_vector @ H

    return received_power


def compute_snr(
    received_power: np.ndarray,
    config: VLCConfig = CONFIG,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute SNR from received optical power.

    The useful signal current is reduced by finite bandwidth:

        I_sig = G_B * R * P_r

    The total noise variance includes:
        - receiver noise floor
        - thermal noise
        - background-light shot noise
        - signal shot noise
        - LED nonlinear distortion noise
    """
    received_power = np.asarray(received_power, dtype=float)

    # DC photocurrent generated by received optical power
    dc_signal_current = config.responsivity * received_power

    # Bandwidth-limited useful signal current
    bandwidth_gain = compute_bandwidth_signal_gain(config)
    useful_signal_current = bandwidth_gain * dc_signal_current

    # Total current-domain noise variance
    total_noise_variance = compute_total_noise_variance(
        dc_signal_current=dc_signal_current,
        useful_signal_current=useful_signal_current,
        config=config,
    )

    snr_linear = (useful_signal_current ** 2) / total_noise_variance
    snr_db = 10.0 * np.log10(np.maximum(snr_linear, _EPS))

    return snr_linear, snr_db


def evaluate_led_selection(
    selection: np.ndarray,
    config: VLCConfig = CONFIG,
    H: np.ndarray | None = None,
) -> dict:
    """
    Evaluate the VLC physical-layer quantities of one LED selection.

    This function only calculates:
        - received optical power
        - SNR in linear scale
        - SNR in dB

    Coverage rate, uniformity, and objective function should be calculated
    in metrics.py.

    Parameters
    ----------
    selection : ndarray, shape (N_led,)
        Binary LED selection vector.

    config : VLCConfig
        Simulation configuration.

    H : ndarray, optional, shape (N_led, N_grid)
        Channel gain matrix. If None, it will be computed automatically.

    Returns
    -------
    result : dict
        Dictionary containing:
            selection
            received_power
            snr_linear
            snr_db
    """
    selection = validate_selection_vector(
        selection,
        config=config,
        require_fixed_number=None,
    )

    if H is None:
        H = compute_channel_gain_matrix(config)

    received_power = compute_received_power(selection, H, config)
    snr_linear, snr_db = compute_snr(received_power, config)

    result = {
        "selection": selection,
        "received_power": received_power,
        "snr_linear": snr_linear,
        "snr_db": snr_db,
    }

    return result


def reshape_to_grid(
    values: np.ndarray,
    config: VLCConfig = CONFIG,
) -> np.ndarray:
    """
    Reshape a 1D receiver-grid result into a 2D grid.

    This is useful for plotting heatmaps.

    Parameters
    ----------
    values : ndarray, shape (N_grid,)
        Values on receiver points, such as received power or SNR.

    config : VLCConfig
        Simulation configuration.

    Returns
    -------
    grid_values : ndarray, shape (grid_size_y, grid_size_x)
        Reshaped 2D grid values.
    """
    values = np.asarray(values, dtype=float).ravel()

    expected_size = config.grid_size_x * config.grid_size_y

    if values.size != expected_size:
        raise ValueError(
            f"Expected {expected_size} grid values, but got {values.size}."
        )

    return values.reshape(config.grid_size_y, config.grid_size_x)


if __name__ == "__main__":
    # Simple test for the VLC model.
    config = CONFIG

    led_positions = get_led_positions(config)
    grid_x, grid_y, grid_z, receiver_points = get_receiver_grid(config)

    H = compute_channel_gain_matrix(
        config=config,
        led_positions=led_positions,
        receiver_points=receiver_points,
    )

    # Example: select LED 1, 3, 7, 9.
    selection = selected_indices_to_vector([1, 4, 6, 11, 13, 16], config=config)

    result = evaluate_led_selection(selection, config=config, H=H)

    print("LED positions shape:", led_positions.shape)
    print("Receiver points shape:", receiver_points.shape)
    print("Channel gain matrix shape:", H.shape)
    print("Selected LEDs:", np.where(selection == 1)[0] + 1)
    print("Received power shape:", result["received_power"].shape)
    print("SNR dB shape:", result["snr_db"].shape)
    print("Average SNR:", np.mean(result["snr_db"]))
    print("Minimum SNR:", np.min(result["snr_db"]))