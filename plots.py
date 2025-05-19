# plots.py

"""
Statistical plots for radar simulation results using Matplotlib.
Each function returns a Figure or None if no data is available.
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_noise_distribution(records, cfg):
    """
    Histogram of measurement error distribution on each axis.
    Args:
        records: list of (true_pos, measured_pos or None)
        cfg: simulation Config
    Returns:
        Matplotlib Figure or None
    """
    # Compute errors where measurements exist
    errors = np.array([meas - true for true, meas in records if meas is not None])
    if errors.size == 0:
        return None

    fig, axs = plt.subplots(3, 1, figsize=(6, 8))
    axes = ['x', 'y', 'z']
    for i, ax in enumerate(axs):
        ax.hist(errors[:, i], bins=30, edgecolor='black')
        ax.set_title(f"{axes[i]}-axis Error (σ={cfg.radar_sigma[i]:.1f} m)")
        ax.set_xlabel("Error (m)")
        ax.set_ylabel("Count")
    plt.tight_layout()
    return fig


def plot_error_vs_index(records):
    """
    Plot error magnitude versus measurement index.
    Args:
        records: list of (true_pos, measured_pos or None)
    Returns:
        Matplotlib Figure
    """
    idxs = np.arange(len(records))
    mags = np.array([
        np.linalg.norm(meas - true) if meas is not None else np.nan
        for true, meas in records
    ])

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(idxs, mags, marker='o', linestyle='-')
    ax.set_title("Error Magnitude vs. Index")
    ax.set_xlabel("Measurement Index")
    ax.set_ylabel("Error (m)")
    return fig


def plot_survival_curve(sim):
    """
    Survival curve (1 - CDF) of missile flight time until intercept or ground.
    Args:
        sim: Simulator instance
    Returns:
        Matplotlib Figure
    """
    times = []
    for w in sim.missiles:
        if sim.intercept and np.allclose(w.obj.pos, sim.impact_msl):
            times.append(sim.time)
        else:
            times.append(w.obj.t)
    times = np.sort(times)
    n = len(times)
    surv = 1 - np.arange(1, n + 1) / n

    fig, ax = plt.subplots()
    ax.step(times, surv, where='post')
    ax.set_title("Missile Survival Curve")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Survival Probability")
    return fig


def plot_impact_heatmap(sim, cfg):
    """
    2D heatmap of final missile impact points on the ground.
    Args:
        sim: Simulator instance
        cfg: simulation Config
    Returns:
        Matplotlib Figure or None
    """
    pts = np.array([w.obj.pos[:2] for w in sim.missiles])
    if pts.size == 0:
        return None

    fig, ax = plt.subplots(figsize=(6, 6))
    h = ax.hist2d(
        pts[:, 0], pts[:, 1],
        bins=30, range=[cfg.x_range, cfg.y_range],
        cmap='hot'
    )
    fig.colorbar(h[3], ax=ax, label='Count')
    ax.set_title("Impact Points Heatmap")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    return fig


def plot_success_vs_distance(sim, cfg):
    """
    Bar chart of intercept success rate vs. initial spawn distance bins.
    Args:
        sim: Simulator instance
        cfg: simulation Config
    Returns:
        Matplotlib Figure
    """
    # Compute initial distances
    starts = np.array([w.obj.start[:2] for w in sim.missiles])
    launch = np.array(cfg.launch_pos[:2])
    d0 = np.linalg.norm(starts - launch, axis=1)

    # Success flags: 1 if intercepted, else 0
    success = [
        int(sim.intercept and np.allclose(w.obj.pos, sim.impact_msl))
        for w in sim.missiles
    ]

    # Bin data
    bins = np.linspace(0, max(d0.max(), cfg.radar_radius), 5)
    inds = np.digitize(d0, bins) - 1

    rates, labels = [], []
    for i in range(len(bins) - 1):
        idxs = np.where(inds == i)[0]
        if idxs.size:
            rates.append(np.mean([success[j] for j in idxs]))
            labels.append(f"{int(bins[i])}-{int(bins[i+1])}")
    fig, ax = plt.subplots()
    ax.bar(labels, rates)
    ax.set_title("Intercept Rate vs. Spawn Distance")
    ax.set_xlabel("Distance Range (m)")
    ax.set_ylabel("Intercept Rate")
    return fig


def plot_detection_rate_vs_distance(records, cfg, num_bins: int = 10):
    """
    Bar chart of detection probability vs. target distance bins.
    Args:
        records: list of (true_pos, measured_pos or None)
        cfg: simulation Config
        num_bins: number of bins
    Returns:
        Matplotlib Figure or None
    """
    data = [
        (np.linalg.norm(true - np.array(cfg.launch_pos)), meas is not None)
        for true, meas in records
    ]
    if not data:
        return None

    dists, detected = zip(*data)
    bins = np.linspace(0, max(dists), num_bins + 1)
    inds = np.digitize(dists, bins) - 1

    rates, labels = [], []
    for i in range(num_bins):
        idxs = [j for j, b in enumerate(inds) if b == i]
        if idxs:
            rates.append(sum(detected[j] for j in idxs) / len(idxs))
        else:
            rates.append(0.0)
        labels.append(f"{int(bins[i])}-{int(bins[i+1])}")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(labels, rates, edgecolor='black')
    ax.set_title("Detection Rate vs. Distance")
    ax.set_xlabel("Distance Bin (m)")
    ax.set_ylabel("Detection Probability")
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    plt.tight_layout()
    return fig


def plot_noise_vs_range(records, cfg):
    """
    Scatter plot of measurement error magnitude vs. target range.
    Args:
        records: list of (true_pos, measured_pos or None)
        cfg: simulation Config
    Returns:
        Matplotlib Figure or None
    """
    data = [
        (np.linalg.norm(true - np.array(cfg.launch_pos)),
         np.linalg.norm(meas - true) if meas is not None else np.nan)
        for true, meas in records
    ]
    if not data:
        return None

    ranges, errs = zip(*data)
    fig, ax = plt.subplots()
    ax.scatter(ranges, errs, alpha=0.6)
    ax.set_title("Error vs. Range")
    ax.set_xlabel("Range (m)")
    ax.set_ylabel("Error (m)")
    return fig
