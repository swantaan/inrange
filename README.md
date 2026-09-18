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

## 3D Interactive Visualizer

You can open `visualizer/hybrid_flight_animation.html` in any web browser to see the shots in 3D.

### Features:
- **Choose Any Shot**: A dropdown menu lets you switch between different test shots.
- **Two Flight Paths**:
  - **Ideal Flight Path (Cyan Line)**: Shows the smooth, natural path calculated by physics.
  - **Radar-Fitted Path (Orange Line)**: Shows the path when forced to hit every measured checkpoint, including radar noise.
- **Measurement Differences (Red Dotted Lines)**: Shows the difference between radar points and the pure physics flight.
- **Realistic Bounce and Roll**: Shows how the ball lands and rolls to a stop on Stellenbosch grass.
- **Playback Controls**: Play, pause, or drag the slider to watch the ball fly through the air.

---

## Folder Structure

```
inrange/
├── data/
│   ├── train.csv                # Training shots with complete radar tracking
│   ├── test.csv                 # Test shots cut off at the 60m net
│   └── sample_submission.csv    # Example submission format
├── src/
│   ├── physics-model.py         # Flight physics and turf bounce simulation
│   ├── machine-learning-model.py# Tree-based residual models
│   ├── deep-learning-model.py   # Neural network model
│   └── hybrid.py                # Main script that combines models and makes the visualizer
├── results/                     # Model predictions and outputs
├── visualizer/
│   └── hybrid_flight_animation.html # 3D flight visualizer
├── report/
│   └── KAGGLE_WRITEUP.md        # Detailed competition report
├── submission.csv               # Final submission file
├── make_submission.py           # Script to check and create submission files
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
