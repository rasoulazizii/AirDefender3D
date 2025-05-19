# live_simulation_pyqt.py

import sys
import numpy as np
from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QHBoxLayout, QLabel, QDoubleSpinBox, QPushButton
)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

from config import Config
from simulator import Simulator

class LiveWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Live 3D Radar Simulation")
        self.resize(900, 700)

        # 1. Create Config with no spawn delay and 45° launch angle
        cfg = Config(
            spawn_delay_rng=(0.0, 0.0),
            bm_flight_angle_deg=45.0
        )
        self.sim = Simulator(cfg)

        # 2. Record initial positions into trajectory logs
        for mw in self.sim.missiles:
            mw.traj.append(mw.obj.pos.copy())
        for iw in self.sim.interceptors:
            iw.traj.append(iw.obj.pos.copy())

        # 3. Set up Matplotlib figure and 3D axes
        self.fig = plt.Figure(figsize=(6, 5))
        self.canvas = FigureCanvas(self.fig)  # store the canvas
        self.ax: Axes3D = self.fig.add_subplot(111, projection='3d')
        self.ax.set_box_aspect([1,1,1])
        self.ax.set_xlim(cfg.x_range)
        self.ax.set_ylim(cfg.y_range)
        self.ax.set_zlim(cfg.z_range)
        self.ax.set_xlabel('x (m)')
        self.ax.set_ylabel('y (m)')
        self.ax.set_zlabel('z (m)')
        self.ax.set_title("Live 3D Radar Simulation")
        self.ax.view_init(elev=30, azim=45)

        # 4. Draw the radar dome and horizontal range circle
        cx, cy, cz = cfg.launch_pos
        r = cfg.radar_radius
        u, v = np.mgrid[0:2*np.pi:18j, 0:np.pi:9j]
        xs = cx + r * np.cos(u) * np.sin(v)
        ys = cy + r * np.sin(u) * np.sin(v)
        zs = cz + r * np.cos(v)
        self.ax.plot_wireframe(xs, ys, zs, color='cyan', lw=0.4, alpha=0.3)
        theta = np.linspace(0, 2*np.pi, 100)
        self.ax.plot(
            cx + r*np.cos(theta),
            cy + r*np.sin(theta),
            cz*np.zeros_like(theta),
            color='cyan', lw=1.0, linestyle=':'
        )
        # mark the target point
        self.ax.scatter(*cfg.bm_target_pos, c='g', marker='o', s=60, label="Target")

        # 5. Prepare line objects for missile and interceptor trajectories
        self.missile_lines = [
            self.ax.plot([], [], [], lw=1.8, label=f"Missile {i+1}")[0]
            for i in range(len(self.sim.missiles))
        ]
        self.interceptor_lines = [
            self.ax.plot([], [], [], lw=2.0, linestyle='--', label=f"Interceptor {i+1}")[0]
            for i in range(len(self.sim.interceptors))
        ]
        self.ax.legend(loc='upper right')

        # 6. Layout: canvas plus controls
        speed_label = QLabel("FPS:")
        self.spin_speed = QDoubleSpinBox()
        self.spin_speed.setRange(0.1, 60.0)
        self.spin_speed.setSingleStep(0.1)
        self.spin_speed.setValue(5.0)
        btn_start = QPushButton("Start")
        btn_start.clicked.connect(self.start)

        controls = QHBoxLayout()
        controls.addWidget(speed_label)
        controls.addWidget(self.spin_speed)
        controls.addWidget(btn_start)

        central = QWidget()
        layout = QVBoxLayout(central)
        layout.addWidget(self.canvas)  # add the canvas to the layout
        layout.addLayout(controls)
        self.setCentralWidget(central)

        # 7. Timer for frame-by-frame updates
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

    def start(self):
        # convert interval to integer milliseconds
        interval_ms = int(1000.0 / self.spin_speed.value())
        self.timer.start(interval_ms)

    def update_frame(self):
        # perform one full simulation step (equivalent to multiple substeps)
        done = self.sim.step()

        # update missile trajectories
        for idx, mw in enumerate(self.sim.missiles):
            pts = np.array(mw.traj)
            self.missile_lines[idx].set_data(pts[:,0], pts[:,1])
            self.missile_lines[idx].set_3d_properties(pts[:,2])

        # update interceptor trajectories
        for idx, iw in enumerate(self.sim.interceptors):
            pts = np.array(iw.traj)
            self.interceptor_lines[idx].set_data(pts[:,0], pts[:,1])
            self.interceptor_lines[idx].set_3d_properties(pts[:,2])

        # redraw the canvas and process events
        self.canvas.draw()
        self.canvas.flush_events()

        # stop the timer when simulation completes
        if done:
            self.timer.stop()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = LiveWindow()
    win.show()
    sys.exit(app.exec())
