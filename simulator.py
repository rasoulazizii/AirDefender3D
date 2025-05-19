# simulator.py

"""
Core simulation engine for radar intercept scenario.

Manages multiple ballistic missiles and interceptors,
advances physics, handles radar detection, and logs trajectories.
"""

from __future__ import annotations
from dataclasses import replace
from typing import List, Optional
import numpy as np

from config import Config
from entities import BallisticMissile3D, Interceptor3D


class _MslWrap:
    """Wrapper for a missile with spawn timing and trajectory log."""
    def __init__(self, obj: BallisticMissile3D, spawn_time: float):
        self.obj = obj
        self.spawn_time = spawn_time
        self.active = False
        self.grounded = False
        self.traj: List[np.ndarray] = []


class _IntWrap:
    """Wrapper for an interceptor with target assignment and trajectory log."""
    def __init__(self, obj: Interceptor3D, target_idx: Optional[int]):
        self.obj = obj
        self.target_idx = target_idx
        self.active = target_idx is not None
        self.traj: List[np.ndarray] = []
        self.has_been_in_range = False  # To detect exit after entering radar dome


class Simulator:
    """
    Simulation of multiple missiles and interceptors.

    - Maps interceptor[i] → missile[i] for i < min(num_interceptors, num_missiles).
    - Interceptors wait until their assigned missile is active.
    - Advances in time steps with finer physics substeps.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.substep_idx = 0

        # Initialize missiles with random spawn times and positions
        self.missiles: List[_MslWrap] = []
        t_cursor = 0.0
        used_points = set()
        for _ in range(cfg.num_missiles):
            # Ensure unique spawn positions
            while True:
                start = cfg.random_spawn_point()
                key = (round(start[0], 6), round(start[1], 6), round(start[2], 6))
                if key not in used_points:
                    used_points.add(key)
                    break
            m_cfg = replace(cfg, bm_start_pos=start)
            missile = BallisticMissile3D(m_cfg)
            self.missiles.append(_MslWrap(missile, t_cursor))
            t_cursor += cfg.random_spawn_delay()

        # Initialize interceptors assigned to missiles by index
        self.interceptors: List[_IntWrap] = []
        for i in range(cfg.num_interceptors):
            interceptor = Interceptor3D(cfg)
            tgt_idx = i if i < len(self.missiles) else None
            self.interceptors.append(_IntWrap(interceptor, tgt_idx))

        # Radar measurements log
        self.meas: List[np.ndarray] = []

        # Simulation state
        self.time = 0.0
        self.frame_count = 0
        self.intercept = False
        self.impact_int: Optional[np.ndarray] = None
        self.impact_msl: Optional[np.ndarray] = None
        self.hit_ground = False
        self.hit_target = False
        self._target_tol = 0.5  # Ground collision tolerance (m)

    def _physics_substep(self) -> bool:
        """
        Perform one physics substep:
        - Activate missiles when spawn_time is reached.
        - Move active missiles under gravity.
        - Activate and move interceptors with EKF updates.
        - Check for intercept or ground hits.
        Returns True if simulation should end.
        """
        self.substep_idx += 1
        dt = self.cfg.dt_phys
        self.time += dt

        # Activate missiles
        for mw in self.missiles:
            if not mw.active and self.time >= mw.spawn_time:
                mw.active = True

        # Move missiles
        for mw in self.missiles:
            if not mw.active or mw.grounded:
                continue
            alive = mw.obj.move(dt)
            if not alive:
                mw.grounded = True
                # Check if hit the intended target location
                dist = np.linalg.norm(mw.obj.pos - np.array(self.cfg.bm_target_pos))
                if dist <= self._target_tol:
                    self.hit_target = True
                else:
                    self.hit_ground = True
            else:
                mw.traj.append(mw.obj.pos.copy())

        # Guide interceptors
        for iw in self.interceptors:
            if not iw.active:
                continue

            # Retrieve assigned missile
            tgt_idx = iw.target_idx
            mw = self.missiles[tgt_idx]  # type: ignore

            # Deactivate if missile has landed
            if mw.grounded:
                iw.active = False
                iw.target_idx = None
                continue

            # Wait until missile spawns
            if not mw.active:
                continue

            # Radar range check and exit detection
            dist_to_radar = np.linalg.norm(mw.obj.pos - np.array(self.cfg.launch_pos))
            if dist_to_radar <= self.cfg.radar_radius:
                iw.has_been_in_range = True
            elif iw.has_been_in_range and dist_to_radar > self.cfg.radar_radius:
                iw.active = False
                iw.target_idx = None
                continue

            # Measure and update EKF
            meas = iw.obj.radar.measure(mw.obj.pos)
            if meas is not None:
                self.meas.append(meas)
            iw.obj.kalman(meas, self.time)

            # Move interceptor and log trajectory
            iw.obj.move(dt)
            iw.traj.append(iw.obj.pos.copy())

            # Check for intercept event
            if iw.obj.hit(mw.obj):
                self.intercept = True
                self.impact_int = iw.obj.pos.copy()
                self.impact_msl = mw.obj.pos.copy()
                mw.grounded = True
                iw.active = False
                iw.target_idx = None

        # End if all missiles grounded or max frames reached
        done = (
            all(mw.grounded for mw in self.missiles) or
            self.frame_count >= self.cfg.max_steps
        )
        return done

    def step(self) -> bool:
        """
        Advance the simulation by one frame (multiple substeps).
        Returns True if simulation is complete.
        """
        if self.hit_ground or self.hit_target:
            return True
        for _ in range(self.cfg.substeps):
            if self._physics_substep():
                return True
        self.frame_count += 1
        return False

    @property
    def m_traj(self) -> List[np.ndarray]:
        """Concatenated trajectories of all missiles."""
        pts: List[np.ndarray] = []
        for mw in self.missiles:
            pts.extend(mw.traj)
        return pts

    @property
    def i_traj_lists(self) -> List[List[np.ndarray]]:
        """Separate trajectory lists for each interceptor."""
        return [iw.traj for iw in self.interceptors]
