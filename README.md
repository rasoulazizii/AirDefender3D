Radar and Missile Interception Simulation

📘 Overview
This is a 3D radar and missile interception simulator featuring:

3D ballistic missiles under gravity and linear drag

Point-mass interceptors guided by an Extended Kalman Filter

Radar sensor model with Gaussian noise and missed detections

Configurable parameters for physics, radar, and intercept logic

2D and 3D graphical output via Matplotlib and PyQt6

⭐ Key Features
Accurate Physics

Projectile motion with gravity and drag

Adjustable launch angle and initial speed

EKF-Based Interception

7-state Extended Kalman Filter

Lead prediction and divergence handling

Realistic Radar

Detection dome with range limit

Load-dependent noise increase and miss probability

PyQt6 GUI

Form inputs for all simulation parameters

Tabbed display of statistical plots

Live 3D Visualization

Real-time trajectory updates in a 3D view

Vibe Coding Methodology

Clear module boundaries (GUI, core simulator, plotting)

Consistent naming and documentation

🛠️ Prerequisites
Python 3.8+

Install dependencies:
pip install numpy scipy matplotlib pandas PyQt6

📂 Project Structure
.
├── app_gui.py # Main GUI application with controls and result tabs
├── live_simulation_pyqt.py # Live 3D simulation window
├── plot3d_core.py # 3D trajectory plotting utilities
├── plots.py # 2D statistical plot functions
├── config.py # Simulation parameters and helper methods
├── entities.py # BallisticMissile3D & Interceptor3D classes
├── radar.py # Radar sensor model with noise and miss logic
├── simulation.py # Simulation runner with measurement logging
├── simulator.py # Core simulation engine
└── README.md # This documentation

⚙️ Configuration
All simulation parameters are defined in config.py:

dt: time step (s)

substeps: physics substeps per frame

bm_flight_angle_deg: missile launch angle (deg)

spawn_circle_radius: missile spawn radius (m)

max_speed: interceptor max speed (m/s)

fuse_radius: proximity fuse radius (m)

radar_sigma: radar noise std dev in x, y, z (m)

radar_miss_prob: base miss probability

Plus interceptor fuel, Kalman process variance, environment bounds, etc.

▶️ Usage
Clone the repository:
git clone <repo_url>
cd <repo_dir>

Run the main GUI:
python app_gui.py

Run the live 3D view:
python live_simulation.py

🚀 Extending the Project
New guidance algorithm
Add a class in entities.py and integrate in simulator.py.

Custom plots
Implement in plots.py and register in app_gui.py.

Additional parameters
Define in config.py and add inputs in app_gui.py.
