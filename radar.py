# radar.py

"""
Radar sensor model for 3D position measurements with detection dome,
Gaussian noise, and missed detections under load.
"""

from __future__ import annotations
import numpy as np
import random
from typing import Callable

from config import Config


class Radar:
    """
    Radar that measures target positions within a dome radius.

    Attributes:
        cfg: Simulation configuration.
        _pos: Function returning current radar platform position.
    """

    def __init__(self, cfg: Config, get_platform_pos: Callable[[], np.ndarray]):
        self.cfg = cfg
        self._pos = get_platform_pos

    def measure(self, tgt_pos: np.ndarray) -> np.ndarray | None:
        """
        Return a noisy measurement of the target position or None.

        Steps:
        1. Check if target is within radar radius.
        2. Apply miss probability, increasing with number of missiles.
        3. Add Gaussian noise scaled by load factor.
        4. Return noisy position or None.
        """
        # Compute target relative to radar
        rel = tgt_pos -self._pos() 
        if np.linalg.norm(rel) > self.cfg.radar_radius:
            return None

        # Increase miss chance under load
        n = self.cfg.num_missiles
        p_miss = min(
            1.0,
            self.cfg.radar_miss_prob + (n - 1) * self.cfg.radar_load_miss_inc
        )
        if random.random() < p_miss:
            return None

        # Noise scale increases with load
        base_sigma = np.array(self.cfg.radar_sigma)
        sigma = base_sigma * (1 + (n - 1) * self.cfg.radar_load_noise_factor)
        noise = np.random.normal(0.0, sigma, size=3)

        # Return noisy measurement
        return tgt_pos + noise
