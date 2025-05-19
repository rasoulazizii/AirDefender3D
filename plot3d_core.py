# plot3d_core.py

"""
Core function to create a standalone 3D plot of trajectories.

Includes:
- Missile paths
- Interceptor paths
- Radar dome and horizontal range circle
- Radar measurements
- Fuse impact marker
"""

import colorsys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def plot_trajectories(sim, cfg):
    """
    Generate a 3D Matplotlib figure showing:
    - Missile and interceptor trajectories
    - Radar detection dome and range circle
    - Measured points
    - Impact (fuse) event marker

    Returns:
        matplotlib.figure.Figure
    """
    fig = plt.figure(figsize=(8, 6))
    ax: Axes3D = fig.add_subplot(111, projection='3d')

    # Set axis limits and labels
    ax.set_xlim(cfg.x_range)
    ax.set_ylim(cfg.y_range)
    ax.set_zlim(cfg.z_range)
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_zlabel('z (m)')
    ax.set_title(f"Trajectories (Missiles = {cfg.num_missiles})")

    # Draw radar dome
    cx, cy, cz = cfg.launch_pos
    r = cfg.radar_radius
    u, v = np.mgrid[0:2*np.pi:18j, 0:np.pi:9j]
    xs = cx + r * np.cos(u) * np.sin(v)
    ys = cy + r * np.sin(u) * np.sin(v)
    zs = cz + r * np.cos(v)
    ax.plot_wireframe(xs, ys, zs, color='cyan', lw=0.4, alpha=0.3)

    # Draw horizontal range circle
    theta = np.linspace(0, 2*np.pi, 100)
    xc = cx + r * np.cos(theta)
    yc = cy + r * np.sin(theta)
    zc = np.full_like(theta, cz)
    ax.plot(xc, yc, zc, color='cyan', lw=1.0, linestyle=':')

    # Mark the target
    ax.scatter(*cfg.bm_target_pos, c='g', marker='o', s=60, label="Target")

    # Plot missile trajectories
    for idx, wrap in enumerate(sim.missiles):
        if not wrap.traj:
            continue
        pts = np.array(wrap.traj)
        hue = 0.55 * idx / max(1, len(sim.missiles))
        color = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2],
                color=color, lw=1.8, label=f"Missile {idx+1}")

    # Plot interceptor trajectories
    for idx, iw in enumerate(sim.interceptors):
        if not iw.traj:
            continue
        pts = np.array(iw.traj)
        hue = 0.6 + 0.4 * idx / max(1, len(sim.interceptors))
        color = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2],
                color=color, lw=2.0, linestyle='--', label=f"Interceptor {idx+1}")

    # Plot radar measurements
    if sim.meas:
        meas = np.array(sim.meas)
        ax.scatter(meas[:, 0], meas[:, 1], meas[:, 2],
                   c='gray', marker='x', s=20, label="Radar meas.")

    # Mark impact (fuse) event
    if getattr(sim, 'intercept', False) and sim.impact_int is not None:
        ix, iy, iz = sim.impact_int
        ax.scatter(ix, iy, iz,
                   c='red', marker='*', s=200, label="Intercept Fuse")

    ax.legend(loc="upper right")
    return fig
