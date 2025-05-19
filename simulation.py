# simulation.py

"""
Run a full simulation with detailed radar measurement logging,
and compute summary reports for analysis.
"""

import numpy as np
import pandas as pd

from config import Config
from simulator import Simulator
from radar import Radar


def simulate_and_record(cfg: Config):
    """
    Execute simulation while capturing each Radar.measure call.

    Returns:
        sim: Simulator after completion.
        measurement_records: list of tuples (true_pos, measured_pos or None).
    """
    # Save original measure method
    orig_measure = Radar.measure
    # empty list( real location, measere location or none)
    measurement_records: list[tuple[np.ndarray, np.ndarray | None]] = []

    def measure_and_record(self, tgt_pos: np.ndarray):
        """
        Replacement for Radar.measure that logs true and measured positions.
        """
        meas = orig_measure(self, tgt_pos)
        measurement_records.append((
            tgt_pos.copy(),
            meas.copy() if meas is not None else None
        ))
        return meas

    # Monkey-patch Radar.measure
    Radar.measure = measure_and_record

    # Run simulation
    sim = Simulator(cfg)
    # if true sim end 
    while not sim.step():
        pass

    # Restore original method
    Radar.measure = orig_measure
    return sim, measurement_records


def compute_reports(sim, measurement_records):
    """
    Generate numeric reports from simulation results.

    Per missile:
      - spawn_time: when missile was launched
      - t_flight: flight duration
      - range: horizontal distance traveled
      - h_max: maximum altitude reached

    Global:
      - intercept: whether an intercept occurred
      - intercept_time: time of intercept
      - impact_distance: distance between interceptor and missile at intercept
      - radar_miss_rate: fraction of missed measurements

    Returns:
        df: pandas DataFrame of per-missile metrics.
        globals_: dict of global metrics.
    """
    rows = []
    for idx, w in enumerate(sim.missiles):
        t_flight = w.obj.t
        start = w.obj.start
        final = w.obj.pos
        R = np.linalg.norm(final[:2] - start[:2])
        h_max = max([p[2] for p in w.traj] + [start[2]])
        rows.append({
            'missile_id': idx,
            'spawn_time': w.spawn_time,
            't_flight': t_flight,
            'range': R,
            'h_max': h_max
        })
    df = pd.DataFrame(rows)

    # Compute radar miss rate
    total = len(measurement_records)
    missed = sum(1 for _, meas in measurement_records if meas is None)
    miss_rate = missed / total if total > 0 else np.nan

    # Global intercept metrics
    if sim.intercept:
        intercept_time = sim.time
        impact_distance = np.linalg.norm(sim.impact_int - sim.impact_msl)
    else:
        intercept_time = np.nan
        impact_distance = np.nan

    globals_ = {
        'intercept': sim.intercept,
        'intercept_time': intercept_time,
        'impact_distance': impact_distance,
        'radar_miss_rate': miss_rate
    }
    return df, globals_
