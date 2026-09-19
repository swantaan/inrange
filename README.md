# Inrange Golf Ball Flight Predictor and 3D Visualizer

This project predicts the full flight of a golf ball from launch to landing, using physics, machine learning, and deep learning for the Inrange Student Competition.

---

## Overview

Driving ranges with safety nets only track golf balls for the first few meters (up to a 60-meter net). Our goal is to predict what happens after the ball passes the net:
- **launch_spin_rate**: Ball spin at launch (RPM).
- **apex_x, apex_y, apex_z, apex_t**: The highest point of the shot and when it happens.
- **landing_x, landing_y, landing_z, landing_t**: Where and when the ball first hits the ground.

---

## How It Works

We combine three models to get the best accuracy:

1. **Physics Model (`src/physics-model.py`)**:
   - Calculates the flight using real aerodynamic formulas (air resistance, spin lift, and gravity).
   - Accounts for local weather and elevation in Stellenbosch, South Africa.
   - Simulates how the ball bounces and rolls on local Kikuyu grass turf.

2. **Machine Learning Model (`src/machine-learning-model.py`)**:
   - Uses LightGBM and CatBoost tree models.
   - Learns the small differences between the physics predictions and the actual radar data.
   - Uses 5-fold cross-validation to prevent overfitting.

3. **Deep Learning Model (`src/deep-learning-model.py`)**:
   - Uses a PyTorch neural network to capture complex patterns in ball speed, angle, and curvature.

4. **Hybrid Pipeline (`src/hybrid.py`)**:
   - Blends the physics base predictions with the machine learning and deep learning corrections.
   - Produces the final competition submission file.

---

## 3D Interactive Golf Flight Simulator

You can open `visualizer/golf_simulator_3d.html` (or `visualizer/index.html`) in any web browser for a full 3D flight experience built with Three.js, GSAP, and custom WebGL shaders.

### Features & Requirements:
- **Accurate 60m Cylindrical Safety Net Arc**:
  - Net is modelled along the exact $60.0\text{m}$ Euclidean radius arc from the tee bay with translucent netting, neon boundary line, and checkpoint pillars.
- **Multi-Tier Range Topography**:
  - Ground level hitting bays ($z \approx 0.06\text{m}$) and elevated upper deck hitting bays ($z \approx 4.125\text{m}$).
- **Pre-Net vs. Post-Net Trajectory Differentiation**:
  - **Radar Tracking Zone (0 - 60m)**: Highlighted in glowing cyan with checkpoint rings ($15\text{m}, 30\text{m}, 45\text{m}, 60\text{m}$).
  - **Predicted Aerodynamic Flight (60m+)**: Smooth aerodynamic parabolic path up to apex and landing.
- **Kikuyu Grass Bounce & Rollout Physics**:
  - Parabolic multi-hop bounces with turf impact shockwaves and expanding green ripple rings on contact.
- **Minimalist Telemetry HUD & Audio**:
  - Web Audio API synthesizer for the driver crack, net pass ping, bounce thumps, and chime.
  - Rotating 3D wind dial, 2D top-down minimap radar, and celebratory particle effects.
- **Dynamic Camera Modes**:
  - **Follow Cam** (behind ball), **TV Cam** (spectator tower), **Tee Cam**, **60m Net Cam** (on top of the safety net), and **Green Cam**.

---

## Folder Structure

```
inrange/
├── data/
│   ├── train.csv                # Training shots with complete radar tracking
│   ├── test.csv                 # Test shots cut off at the 60m net
│   └── sample_submission.csv    # Example submission format
├── src/
│   ├── physics.py               # Core Stellenbosch dimpled flight & turf physics engine
│   ├── features.py              # Physics-informed aerodynamic & kinematic feature extraction
│   ├── models.py                # Tree ensemble (LightGBM/CatBoost/XGBoost) & Deep Neural Net
│   ├── hybrid.py                # Hybrid orchestrator combining physics baseline & learned residuals
│   ├── trajectory_visualizer.py # Interactive Plotly dual-path comparison 3D visualizer
│   ├── build_3d_simulator.py    # 3D interactive simulator generator
│   ├── generate_funky_textures.py # Funky cyberpunk radar ball & turf texture generator
│   ├── physics-model.py         # Physics CLI entrypoint
│   ├── machine-learning-model.py# Tree model CLI entrypoint
│   └── deep-learning-model.py   # Deep learning CLI entrypoint
├── results/
│   ├── submission.csv           # Final competition submission file
│   ├── hybrid_predictions.csv   # Winning hybrid ensemble predictions
│   └── hybrid_oof.csv           # 5-fold cross-validation out-of-fold predictions
├── visualizer/
│   ├── golf_simulator_3d.html   # Standalone 3D golf flight simulator
│   ├── index.html               # Visualizer entrypoint
│   ├── textures/                # High-res procedural PNG textures
│   └── legacy/                  # Archived legacy visualizers
├── report/
│   ├── figures/                 # High-resolution report graphics & architecture schematics
│   └── KAGGLE_WRITEUP.md        # Detailed competition report
├── generate_report_diagrams.py  # Generates publication-ready report figures & charts
├── make_submission.py           # Script to validate and generate results/submission.csv
└── README.md
```

---

## Quick Start

### Install Dependencies
```bash
pip install numpy scipy pandas scikit-learn lightgbm catboost torch plotly
```

### Run the Pipeline
```bash
# Run the model pipeline and build the visualizer:
python src/hybrid.py

# Check that the submission file is valid:
python make_submission.py
```

### View the 3D Flight
Double-click `visualizer/hybrid_flight_animation.html` or open it in any web browser.
