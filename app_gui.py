# app_gui.py

"""
GUI for Radar Simulation using PyQt6.

Provides a main window with controls to configure and run the simulation,
and tabs to display various statistical plots. Launches a separate window
for the 3D trajectory visualization.
"""

import sys
import matplotlib.pyplot as plt
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QFormLayout, QSpinBox,
    QDoubleSpinBox, QPushButton, QTabWidget, QLabel
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

from config import Config
import simulation
import plots
import plot3d_core


class MainWindow(QMainWindow):
    """Main application window with simulation controls and result tabs."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Radar Simulation GUI")
        self.resize(1280, 720)

        # (parameters input)Build form layout for simulation parameters
        form = QFormLayout()
        self.spin_missiles = QSpinBox()
        self.spin_missiles.setRange(1, 50)
        self.spin_missiles.setValue(3)

        self.spin_interceptors = QSpinBox()
        self.spin_interceptors.setRange(1, 50)
        self.spin_interceptors.setValue(1)

        self.spin_angle = QDoubleSpinBox()
        self.spin_angle.setRange(0, 90)
        self.spin_angle.setValue(45)

        self.spin_radius = QDoubleSpinBox()
        self.spin_radius.setRange(0, 20000)
        self.spin_radius.setValue(1000)

        self.spin_max_speed = QDoubleSpinBox()
        self.spin_max_speed.setRange(0, 2000)
        self.spin_max_speed.setValue(600)

        self.spin_fuse = QDoubleSpinBox()
        self.spin_fuse.setRange(0, 1000)
        self.spin_fuse.setValue(40.5)

        self.spin_radar_range = QDoubleSpinBox()
        self.spin_radar_range.setRange(0, 50000)
        self.spin_radar_range.setValue(20000)

        self.spin_sigma_x = QDoubleSpinBox()
        self.spin_sigma_x.setRange(0, 100)
        self.spin_sigma_x.setValue(1)

        self.spin_sigma_y = QDoubleSpinBox()
        self.spin_sigma_y.setRange(0, 100)
        self.spin_sigma_y.setValue(1)

        self.spin_sigma_z = QDoubleSpinBox()
        self.spin_sigma_z.setRange(0, 100)
        self.spin_sigma_z.setValue(1)

        self.spin_miss_prob = QDoubleSpinBox()
        self.spin_miss_prob.setRange(0, 1)
        self.spin_miss_prob.setSingleStep(0.01)
        self.spin_miss_prob.setValue(0.01)

        form.addRow("Num Missiles:", self.spin_missiles)
        form.addRow("Num Interceptors:", self.spin_interceptors)
        form.addRow("Flight Angle (deg):", self.spin_angle)
        form.addRow("Spawn Radius (m):", self.spin_radius)
        form.addRow("Interceptor Max Speed (m/s):", self.spin_max_speed)
        form.addRow("Fuse Radius (m):", self.spin_fuse)
        form.addRow("Radar Radius (m):", self.spin_radar_range)
        form.addRow("Radar Noise σ_x (m):", self.spin_sigma_x)
        form.addRow("Radar Noise σ_y (m):", self.spin_sigma_y)
        form.addRow("Radar Noise σ_z (m):", self.spin_sigma_z)
        form.addRow("Radar Miss Prob:", self.spin_miss_prob)

        # Start button
        self.btn_start = QPushButton("Start Simulation")
        self.btn_start.clicked.connect(self.on_start)

        # Titles for result tabs
        self.titles = [
            "Noise Dist.", "Error vs Index", "Survival",
            "Impact Heatmap", "Success vs Dist",
            "Detection Rate", "Error vs Range"
        ]

        # Create tabs placeholder
        self.tabs = QTabWidget()
        for title in self.titles:
            self.tabs.addTab(QWidget(), title)

        # Layout assembly: left controls, then tabs
        left = QWidget()
        lv = QVBoxLayout(left)
        lv.addLayout(form)
        lv.addWidget(self.btn_start)

        container = QWidget()
        main_layout = QVBoxLayout(container)
        main_layout.addWidget(left)
        main_layout.addWidget(self.tabs)

        self.setCentralWidget(container)

    def on_start(self):
        """Run simulation and update all plots."""
        # Close any existing Matplotlib windows
        plt.close('all')
        self.btn_start.setEnabled(False)

        # Build configuration from input widgets
        cfg = Config(
            num_missiles=self.spin_missiles.value(),
            num_interceptors=self.spin_interceptors.value(),
            bm_flight_angle_deg=self.spin_angle.value(),
            spawn_circle_radius=self.spin_radius.value(),
            max_speed=self.spin_max_speed.value(),
            fuse_radius=self.spin_fuse.value(),
            radar_radius=self.spin_radar_range.value(),
            radar_sigma=(
                self.spin_sigma_x.value(),
                self.spin_sigma_y.value(),
                self.spin_sigma_z.value()
            ),
            radar_miss_prob=self.spin_miss_prob.value()
        )

        # Run simulation and record measurements
        sim, records = simulation.simulate_and_record(cfg)

        # Show separate 3D plot window
        fig3d = plot3d_core.plot_trajectories(sim, cfg)
        fig3d.show()

        # List of (plot function, args) for tabs
        plot_funcs = [
            (plots.plot_noise_distribution,        (records, cfg)),
            (plots.plot_error_vs_index,            (records,)),
            (plots.plot_survival_curve,            (sim,)),
            (plots.plot_impact_heatmap,            (sim, cfg)),
            (plots.plot_success_vs_distance,       (sim, cfg)),
            (plots.plot_detection_rate_vs_distance,(records, cfg)),
            (plots.plot_noise_vs_range,            (records, cfg)),
        ]

        # Rebuild tabs with new figures
        self.tabs.clear()
        for fn_args, title in zip(plot_funcs, self.titles):
            fn, args = fn_args
            fig = fn(*args)
            tab = QWidget()
            layout = QVBoxLayout(tab)
            if fig is None:
                layout.addWidget(QLabel("No data to display"))
            else:
                canvas = FigureCanvas(fig)
                layout.addWidget(canvas)
            self.tabs.addTab(tab, title)

        self.btn_start.setEnabled(True)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())
