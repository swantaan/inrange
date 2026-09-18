# Inrange Golf Ball Trajectory Prediction & 3D Flight Visualizer

An end-to-end Physics-Informed, Machine Learning, and Deep Learning solution for predicting full 3D golf ball flight trajectories, apex milestones, landing coordinates, and Stellenbosch turf rollout for the **Inrange Student Competition**.

---

## Overview

Modern golf tracking systems (such as Inrange and TrackMan) capture the initial launch window of a golf ball before netting, urban barriers, or range boundaries obstruct full-flight radar tracking. The challenge is to extrapolate the complete 3D trajectory from trimmed early-flight checkpoints ($15\text{m}$, $30\text{m}$, $45\text{m}$, and $60\text{m}$ Urban Net), predicting:
- **`launch_spin_rate`**: Backspin / total spin rate in RPM.
- **`apex_x`, `apex_y`, `apex_z`, `apex_t`**: 3D coordinates and timestamp of the maximum flight elevation.
- **`landing_x`, `landing_y`, `landing_z`, `landing_t`**: First ground contact coordinates and impact time.

---

## Architecture & Modeling Approach

The project implements a **Hybrid Physics-Guided Ensemble**:

1. **Aerodynamic Physics Engine (`src/physics-model.py`)**:
   - Numerical ODE integration of 3D ballistic equations.
   - Dynamic Quintavalla dimpled drag model: $C_d(v) = C_{d0} + k C_l^2$.
   - Magnus lift force with continuous exponential spin decay: $\omega(t) = \omega_0 e^{-t/\tau}$.
   - Calibrated air density ($P = 997.0\text{ hPa}$, $T = 20^\circ\text{C}$).
   - Stellenbosch Kikuyu turf plastic deformation model (Clegg Impact Value $\approx 75-80\text{ CIV}$), backspin shear bite, and rolling resistance.

2. **Gradient Boosted Machine Learning (`src/machine-learning-model.py`)**:
   - 5-Fold Cross-Validation with LightGBM and CatBoost.
   - High-order kinetic features, curvature vectors, and ballistic energy invariants.
   - Out-of-fold residual learning targeting physics prediction deltas.

3. **Deep Learning ResNet (`src/deep-learning-model.py`)**:
   - Deep PyTorch residual neural network with Swish activations and LayerNorm.
   - Models non-linear aerodynamic boundary layer transitions and turbulence regimes.

4. **Hybrid Ensemble & Trajectory Pipeline (`src/hybrid.py`)**:
   - Blends physics base predictions with ML and DL residuals using target-specific optimal grid weights.
   - Generates final submission files compliant with competition specifications.

---

## 3D Interactive Visualizer

The interactive 3D visualizer is located in `visualizer/hybrid_flight_animation.html`.

### Key Features:
- **Multi-Shot Dropdown**: Select and inspect any predicted shot from the test dataset.
- **Dual Flight Path Comparison**:
  - **Ideal Aerodynamic Path (Cyan Solid)**: Pure fluid-dynamic ODE trajectory without radar noise.
  - **Radar-Constrained Path (Orange Dashed)**: Path fitted strictly through noisy radar checkpoints.
- **Measurement Error Vectors (Crimson Dotted)**: Visualizes the exact 3D measurement residuals ($\Delta$) at each radar checkpoint.
- **Stellenbosch Kikuyu Turf Ballistics**: Authentic post-landing micro-hop ($0.25\text{m} - 0.40\text{m}$) and rollout with backspin check.
- **Play/Pause & Time Scrubbing**: Interactive playback controls for complete flight simulation.

---

## Repository Structure

```
inrange/
├── data/
│   ├── train.csv                # 492 recorded full-flight radar shots (Stellenbosch, SA)
│   ├── test.csv                 # 500 trimmed radar test shots
│   └── sample_submission.csv    # Submission schema template
├── src/
│   ├── physics-model.py         # 3D aerodynamic ODE simulator & turf ballistics
│   ├── machine-learning-model.py# LightGBM / CatBoost residual regression
│   ├── deep-learning-model.py   # PyTorch ResNet residual architecture
│   └── hybrid.py                # Pipeline execution, ensembling & 3D visualizer
├── results/                     # Out-of-fold and test predictions
├── visualizer/
│   └── hybrid_flight_animation.html # Standalone interactive 3D visualizer
├── report/
│   └── KAGGLE_WRITEUP.md        # Technical documentation and methodology writeup
├── submission.csv               # Competition submission file
├── make_submission.py           # Verification and submission generation script
└── README.md
```

---

## Getting Started

### Prerequisites
```bash
pip install numpy scipy pandas scikit-learn lightgbm catboost torch plotly
```

### Running the Pipeline
```bash
# Run the complete hybrid pipeline and generate predictions:
python src/hybrid.py

# Verify and format submission:
python make_submission.py
```

### Viewing the Visualizer
Open `visualizer/hybrid_flight_animation.html` in any web browser.
