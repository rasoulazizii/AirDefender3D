# entities.py

"""
Defines the moving objects in the simulation:
- BallisticMissile3D: simple 3D projectile under gravity and linear drag.
- Interceptor3D: guided interceptor with an Extended Kalman Filter for target tracking.
"""

from __future__ import annotations
import math
import numpy as np
from typing import Optional, Callable

from config import Config


class BallisticMissile3D:
    """
    A 3D ballistic missile under gravity and linear drag.

    Initial velocity is calculated to reach the target at a given launch angle.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.start = np.array(cfg.bm_start_pos, dtype=float)
        self.pos = self.start.copy()
        self.t = 0.0

        # Horizontal direction toward target
        target = np.array(cfg.bm_target_pos, float)
        d_xy = target[:2] - self.start[:2]
        R = np.linalg.norm(d_xy) + 1e-12
        dir_xy = d_xy / R

        # Launch angle in radians
        θ = math.radians(cfg.bm_flight_angle_deg)
        sinθ, cosθ = math.sin(θ), math.cos(θ)
        h0 = self.start[2]
        eps = 1e-8

        # Compute base speed to reach target
        if abs(sinθ) < eps:
            # Horizontal launch
            if h0 > 0:
                t_fall = math.sqrt(2 * h0 / cfg.g)
                v0_base = R / t_fall
            else:
                v0_base = 1.0
        else:
            denom = 2 * cosθ**2 * (R * math.tan(θ) + h0)
            if denom > 0:
                v0_base = math.sqrt((R**2 * cfg.g) / denom)
            else:
                v0_base = math.sqrt(R * cfg.g / (math.sin(2 * θ) + eps))

        # Enforce minimum speed
        v0_base = max(v0_base, cfg.v0_range[0])

        # Apply drag
        v0 = v0_base / (1.0 + cfg.drag_coeff)

        # Initial velocity components
        vx = v0 * cosθ * dir_xy[0]
        vy = v0 * cosθ * dir_xy[1]
        vz = v0 * sinθ

        self.vel = np.array([vx, vy, vz], float)
        self.vx, self.vy, self.vz = vx, vy, vz

    def move(self, dt: float) -> bool:
        """
        Advance by one physics substep.
        Update position under gravity.
        Returns False if missile has hit the ground (z <= 0).
        """
        dt_eff = self.cfg.dt_phys
        self.t += dt_eff

        # Linear motion + vertical drop
        self.pos[0] = self.start[0] + self.vx * self.t
        self.pos[1] = self.start[1] + self.vy * self.t
        self.pos[2] = self.start[2] + self.vz * self.t - 0.5 * self.cfg.g * self.t**2

        return self.pos[2] > 0.0


class Interceptor3D:
    """
    A point-mass interceptor with:
    - 7-state EKF (position, velocity, drag)
    - Delayed measurement handling
    - Fuel consumption
    - Self-deactivation on failure criteria
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.pos = np.array(cfg.launch_pos, float)

        # Fuel system
        self.fuel_remaining = cfg.interceptor_fuel_capacity

        # For detecting divergence
        self.last_distance = float('inf')

        # EKF initialization
        self.last_update_time = 0.0
        self.state_dim = 7
        self.H = np.hstack([np.eye(3), np.zeros((3, 4))])
        q = cfg.process_var
        self.Q = np.diag([0, 0, 0, q, q, q, q])
        sx, sy, sz = cfg.radar_sigma
        self.R = np.diag([sx**2, sy**2, sz**2])

        self.x: Optional[np.ndarray] = None
        self.P: Optional[np.ndarray] = None
        self.pred_next = self.pos.copy()

        # Radar sensor attached to this interceptor
        from radar import Radar
        self.radar = Radar(cfg, get_platform_pos=lambda: self.pos)

    def kalman(self, meas: Optional[np.ndarray], t_meas: float):
        """
        Update EKF with optional measurement:
        - Predict to measurement time, then correct if measurement exists.
        - Predict ahead to current physics time.
        - Compute next interception point.
        """
        # Initialize state on first measurement
        if self.x is None:
            if meas is None:
                return
            self.x = np.hstack([meas, (0.0, 0.0, 0.0), self.cfg.drag_coeff])
            sx2, sy2, sz2 = np.square(self.cfg.radar_sigma)
            self.P = np.diag([sx2, sy2, sz2, 1, 1, 1, 1])
            self.last_update_time = t_meas
            self.last_distance = np.linalg.norm(meas - self.pos)
            return

        # Prediction to measurement time
        dt1 = t_meas - self.last_update_time
        F6 = np.block([[np.eye(3), dt1 * np.eye(3)],
                       [np.zeros((3, 3)), np.eye(3)]])
        F = np.block([[F6, np.zeros((6, 1))],
                      [np.zeros((1, 6)), np.eye(1)]])
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self.Q

        # Correction step
        if meas is not None:
            y = meas - self.H @ self.x
            S = self.H @ self.P @ self.H.T + self.R
            K = self.P @ self.H.T @ np.linalg.inv(S)
            self.x += K @ y
            self.P = (np.eye(self.state_dim) - K @ self.H) @ self.P

        # Predict ahead one physics substep
        dt2 = self.cfg.dt_phys
        F6b = np.block([[np.eye(3), dt2 * np.eye(3)],
                        [np.zeros((3, 3)), np.eye(3)]])
        Fb = np.block([[F6b, np.zeros((6, 1))],
                       [np.zeros((1, 6)), np.eye(1)]])
        self.x = Fb @ self.x
        self.P = Fb @ self.P @ Fb.T + self.Q
        self.last_update_time = t_meas

        # Compute dynamic intercept horizon
        rel = self.x[:3] - self.pos
        dist = np.linalg.norm(rel)
        t_c = dist / (self.cfg.max_speed + 1e-6)
        n = max(1, math.ceil(t_c / self.cfg.dt_phys))
        nom = self.x[:3] + self.x[3:6] * t_c

        # Limit by max reachable distance
        dir_vec = nom - self.pos
        dist_nom = np.linalg.norm(dir_vec)
        max_reach = self.cfg.max_speed * self.cfg.dt_phys * n
        if dist_nom > max_reach:
            nom = self.pos + dir_vec / dist_nom * max_reach

        self.pred_next = nom

    def move(self, dt: float):
        """
        Move toward the predicted intercept point.
        Consume fuel and deactivate if out of fuel or diverging.
        """
        direction = self.pred_next - self.pos
        dist = np.linalg.norm(direction)
        if dist == 0:
            return

        # Step size limited by max speed
        step = min(self.cfg.max_speed * dt, dist)

        # Fuel consumption
        self.fuel_remaining -= step * self.cfg.interceptor_fuel_consumption
        if self.fuel_remaining <= 0:
            self.active = False
            return

        # Update position
        self.pos += (direction / dist) * step
        self.pos[2] = max(0.0, self.pos[2])

        # Check for divergence
        new_dist = np.linalg.norm(self.pred_next - self.pos)
        if new_dist > self.last_distance + 1e-6:
            self.active = False
        else:
            self.last_distance = new_dist

    def hit(self, tgt: BallisticMissile3D) -> bool:
        """
        Return True if within fuse radius of target.
        """
        return np.linalg.norm(tgt.pos - self.pos) <= self.cfg.fuse_radius
