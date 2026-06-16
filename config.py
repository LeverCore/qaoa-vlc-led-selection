from dataclasses import dataclass
import numpy as np


@dataclass
class VLCConfig:
    # ============================================================
    # 1. Room geometry
    # ============================================================
    room_length: float = 10.0     # L, unit: m
    room_width: float = 10.0      # W, unit: m
    room_height: float = 3.0     # H, unit: m

    # Receiver plane height
    receiver_height: float = 0.85    # z_r, unit: m

    # LED installation height
    led_height: float = 3.0          # unit: m

    # ============================================================
    # 2. LED candidate positions
    # ============================================================
    # 4 × 4 candidate LED grid
    num_led_x: int = 4
    num_led_y: int = 4
    num_led_total: int = 16

    # Whether to force a fixed number of selected LEDs.
    # False means that the optimizer can decide how many LEDs to use.
    use_fixed_led_number: bool = False

    # Only used when use_fixed_led_number = True.
    # It is kept here for compatibility with previous enumeration code.
    num_led_select: int = 4

    # Equivalent optical transmit power of one selected LED.
    led_power: float = 1.0     # P_t = 1 W

    # Whether to consider LED optical-power saturation.
    enable_led_nonlinearity: bool = True

    # Saturation optical power of one equivalent LED source.
    # Larger value means weaker nonlinearity.
    led_saturation_power: float = 2.0     # unit: W

    # Nonlinear distortion coefficient.
    # This introduces an additional signal-dependent noise term.
    nonlinear_distortion_factor: float = 0.02

    # LED semi-angle at half power
    half_power_angle_deg: float = 70.0

    # ============================================================
    # 3. Receiver parameters
    # ============================================================
    pd_area: float = 1.0e-4       # A = 1 cm^2 = 1e-4 m^2
    responsivity: float = 0.53    # R, unit: A/W

    fov_deg: float = 60.0         # Psi_c, unit: degree

    # Optical filter gain
    optical_filter_gain: float = 1.0

    # Optical concentrator refractive index
    refractive_index: float = 1.5

    # Optical concentrator gain reference value.
    # In vlc_model.py, g(psi) can also be computed from n and FOV.
    concentrator_gain: float = 3.0

    # ============================================================
    # 4. Receiver grid
    # ============================================================
    grid_size_x: int = 50
    grid_size_y: int = 50

    # ============================================================
    # 5. Noise, bandwidth, and SNR settings
    # ============================================================
    # Receiver noise floor in current domain.
    # This term is kept as a baseline receiver noise variance.
    noise_variance: float = 1.0e-14      # unit: A^2

    # Background-light-induced photocurrent.
    # Larger background current leads to stronger shot noise.
    background_current: float = 5.0e-6   # unit: A

    # Communication bandwidth.
    # Noise power increases with bandwidth.
    communication_bandwidth_hz: float = 20.0e6

    # Equivalent 3-dB bandwidth of LED + receiver.
    # If communication_bandwidth_hz is close to this value,
    # the useful signal is attenuated.
    system_3db_bandwidth_hz: float = 50.0e6

    # Physical constants for noise calculation.
    electron_charge: float = 1.602176634e-19
    boltzmann_constant: float = 1.380649e-23
    temperature_k: float = 300.0
    load_resistance_ohm: float = 50.0

    # Communication coverage threshold.
    snr_threshold_db: float = 13.6

    # ============================================================
    # 6. QUBO objective weights
    # ============================================================
    # Target-power-tracking QUBO objective:
    #
    # E_QUBO(x) =
    #     tracking_weight / M * sum_j (P_r,j(x) / P_tar - 1)^2
    #     + power_weight * normalized_power
    #
    # where:
    #   P_r,j(x)         : received optical power at receiver point j
    #   P_tar            : target received optical power
    #   normalized_power : number of selected LEDs / total candidate LEDs
    #
    # P_tar is computed from the SNR threshold using the current VLC noise model:
    #
    #   P_tar = target_power_margin * P_req
    #
    # where P_req is the received power required to reach snr_threshold_db.
    #
    tracking_weight: float = 1.0

    # Target received-power margin.
    # A small value makes selected LEDs easily exceed the target power,
    # which may make the all-off solution artificially competitive.
    target_power_margin: float = 7.0

    # LED power-consumption penalty.
    # Larger value favors fewer LEDs; smaller value allows more LEDs.
    power_weight: float = 0.2

    # ============================================================
    # 7. Output path
    # ============================================================
    output_dir: str = "outputs"


def get_lambertian_order(config: VLCConfig) -> float:
    """
    Compute Lambertian emission order:
        m = -ln(2) / ln(cos(Phi_{1/2}))
    """
    phi_half = np.deg2rad(config.half_power_angle_deg)
    return -np.log(2.0) / np.log(np.cos(phi_half))


def get_led_positions(config: VLCConfig) -> np.ndarray:
    """
    Generate candidate LED positions.

    For a 4 × 4 grid, the LED candidate positions are uniformly placed
    on the ceiling with equal margins from the room boundary.

    Returns
    -------
    led_positions : ndarray, shape (num_led_total, 3)
        Each row is [x_i, y_i, z_i].
    """
    # Use equal-margin placement.
    # For example, if room_length = 5 m and num_led_x = 4,
    # x positions are approximately {1, 2, 3, 4}.
    x_positions = np.linspace(
        config.room_length / (config.num_led_x + 1),
        config.room_length * config.num_led_x / (config.num_led_x + 1),
        config.num_led_x
    )

    y_positions = np.linspace(
        config.room_width / (config.num_led_y + 1),
        config.room_width * config.num_led_y / (config.num_led_y + 1),
        config.num_led_y
    )

    led_positions = []

    for y in y_positions:
        for x in x_positions:
            led_positions.append([x, y, config.led_height])

    return np.array(led_positions, dtype=float)


def get_receiver_grid(config: VLCConfig):
    """
    Generate receiver grid on the working plane z = receiver_height.

    Returns
    -------
    grid_x : ndarray, shape (grid_size_y, grid_size_x)
    grid_y : ndarray, shape (grid_size_y, grid_size_x)
    grid_z : ndarray, shape (grid_size_y, grid_size_x)
    receiver_points : ndarray, shape (grid_size_x * grid_size_y, 3)
        Each row is [x_j, y_j, z_j].
    """
    x = np.linspace(0.0, config.room_length, config.grid_size_x)
    y = np.linspace(0.0, config.room_width, config.grid_size_y)

    grid_x, grid_y = np.meshgrid(x, y)
    grid_z = np.full_like(grid_x, config.receiver_height)

    receiver_points = np.column_stack([
        grid_x.ravel(),
        grid_y.ravel(),
        grid_z.ravel()
    ])

    return grid_x, grid_y, grid_z, receiver_points


# Create a default global config object.
CONFIG = VLCConfig()