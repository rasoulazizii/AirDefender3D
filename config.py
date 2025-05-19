# config.py

"""
Simulation configuration for radar intercept scenario.
Units: meters (m) and seconds (s).
"""

import math
import random
from dataclasses import dataclass, field
from typing import Tuple, TypeAlias, NamedTuple

import numpy as _np

# 3d vector
Vec3: TypeAlias = Tuple[float, float, float]


class Rect(NamedTuple):
    """Axis-aligned rectangle defined by min/max in x and y."""
    x_min: float
    x_max: float
    y_min: float
    y_max: float


@dataclass
class Config:
    """
    Holds all parameters for the simulation.
    - Time settings
    - Ballistic missile settings
    - Interceptor settings
    - Radar settings
    - Environment limits
    """
    # --- Time settings ---
    dt: float = 1.0                                  # Time per simulation step (s)
    substeps: int = 10                               # Physics substeps per step
    # dt / substep
    dt_phys: float = field(init=False)               # Time per physics substep (s)

    # --- Physics constants ---
    g: float = 9.81                                  # Gravity acceleration (m/s^2)
    drag_coeff: float = 0.0                          # Linear drag coefficient

    # --- Ballistic missile ---
    # init speed (0 to infinite)
    v0_range: Tuple[float, float] = (0.0, float('inf'))
    bm_flight_angle_deg: float = 45.0                 # Launch angle (degrees)
    bm_start_pos: Vec3 = (5000.0, 5000.0, 0.0)        # Launch coordinates
    bm_target_pos: Vec3 = (35000.0, 35000.0, 0.0)     # Target coordinates

    # --- Multiple missiles ---
    num_missiles: int = 5
    spawn_circle_radius: float = 10000.0               # Radius around start for spawn (m)
    spawn_delay_rng: Tuple[float, float] = (0.0, 5.0) # Random delay range for each missile (s)

    # --- Interceptors ---
    num_interceptors: int = 5
    launch_pos: Vec3 = (30000.0, 30000.0, 0.0)        # Interceptor launch point
    max_speed: float = 600.0                         # Max interceptor speed (m/s)
    fuse_radius: float = 50.5                        # Proximity fuse radius (m)
    lead_steps: int = 1                              # Steps ahead for lead prediction

    # --- Interceptor fuel ---
    interceptor_fuel_capacity: float = 30000.0       # Fuel capacity (m of movement)
    interceptor_fuel_consumption: float = 1.0        # Fuel per meter moved

    # --- Radar sensor  ---
    radar_radius: float = 10000.0                    # Detection dome radius (m)
    radar_sigma: Vec3 = (1.0, 1.0, 1.0)              # Noise standard deviations (m)
    radar_miss_prob: float = 0.01                    # Base miss probability
    radar_load_noise_factor: float = 0.05            # Noise increase per extra target
    radar_load_miss_inc: float = 0.005               # Miss probability increase per extra target

    # --- Kalman filter ---
    process_var: float = 0.02                        # Process variance for velocity drag

    # --- Environment bounds ---
    x_range: Tuple[float, float] = (0.0, 40000.0)
    y_range: Tuple[float, float] = (0.0, 40000.0)
    z_range: Tuple[float, float] = (0.0, 40000.0)
    max_steps: int = 12000                           # Max simulation frames

    # --- Miscellaneous ---
    random_seed: int | None = 42                     # Seed for reproducibility
    debug: bool = True                              # Enable debug mode

    def __post_init__(self):
        """Compute derived values and seed random generators."""
        self.dt_phys = self.dt / self.substeps
        if self.random_seed is not None:
            random.seed(self.random_seed)
            _np.random.seed(self.random_seed)

    def random_spawn_point(self) -> Vec3:
        """
        Sample a uniform random point on the circle perimeter
        around the missile start position.
        """
        theta = random.uniform(0.0, 2.0 * math.pi)
        r = self.spawn_circle_radius
        cx, cy, cz = self.bm_start_pos
        x = cx + r * math.cos(theta)
        y = cy + r * math.sin(theta)
        return (x, y, cz)

    def random_spawn_delay(self) -> float:
        """Return a random delay within the configured range."""
        lo, hi = self.spawn_delay_rng
        return random.uniform(lo, hi)
