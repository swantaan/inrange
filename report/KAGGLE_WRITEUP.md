# Inrange Golf Ball Trajectory Prediction

**Competition**: Inrange Competition 
**Author**: Christiaan Swanepoel  
**Approach**: Combining Physics, Tree Models, and Deep Learning  

---

# Input
1. Launch information at the tee: ball speed, angles, and starting position.
2. Checkpoints at 15 meters, 30 meters, 45 meters, and the 60-meter net.

The goal is to predict what happens after the ball passes the net for 559 test shots:
- **Launch Spin Rate**: Ball spin in RPM.
- **Apex**: The highest point of the flight (X, Y, Z coordinates and time).
- **Landing**: Where and when the ball touches the ground.
- **Visualizer**: A 3D animation showing the full flight, landing, and rollout.

I built a simple and effective pipeline:
1. A physics model calculates the flight using real aerodynamic formulas (gravity, air drag, and spin lift).
2. Machine learning models (LightGBM, CatBoost) and a neural network predict the remaining differences between the physics calculation and real radar observations.
3. Combining them gives us accurate predictions that follow the laws of physics.

---

### A. Bay Heights and Ground Levels
Looking at the data from the 491 training shots, we found that shots come from four different hitting bays:
- **Bay 1**: Ground level (height around 0.07 meters).
- **Bay 2**: Upper deck (height around 4.125 meters).
- **Bay 3**: Ground level (height around 0.06 meters).
- **Bay 4**: Ground level (height around 0.05 meters).


### B. Grouping by Club Type
Different golf clubs create different types of shots:
- **Drivers**: High ball speed (65 to 78 m/s), low launch angle (10 to 15 degrees), low spin (2,000 to 3,000 RPM).
- **Mid-Irons**: Medium ball speed (45 to 60 m/s), medium launch angle (15 to 22 degrees), medium spin (5,000 to 7,000 RPM).
- **Wedges**: Lower ball speed (30 to 40 m/s), high launch angle (25 to 32 degrees), high backspin (8,000 to 12,000 RPM).

I created a simple ratio:
$$\text{Club Ratio} = \frac{\text{vertical speed}}{(\text{total speed})^2}$$
This ratio easily separates drivers from wedges and improved our spin prediction error from 784 RPM down to 730 RPM.

### C. Slowing Down at the Net
The data shows that balls lose about 31% of their speed by the time they reach the 60-meter net. Measuring this slowdown helps the models understand the air resistance for each shot.

---

## 3. The Physics Model

### A. Local Air in Stellenbosch
The shots were recorded near Stellenbosch, South Africa (about 136 meters above sea level, with an average temperature of about 20 degrees Celsius). At this height, the air pressure is about 997 hPa and air density is around 1.187 kg/m^3. This is about 3% thinner than sea-level air, which means the ball travels slightly farther because there is less air resistance.

### B. Drag, Lift, and Spin
- **Air Resistance (Drag)**: Dimples on a golf ball reduce drag, allowing it to fly smoothly.
- **Lift (Magnus Effect)**: Backspin pushes air downward, which creates an upward lift force holding the ball in the air.
- **Spin Decay**: Spin slowly decreases during flight as the ball moves through the air.

### C. Turf Bounce and Rollout in Stellenbosch
The driving range in Stellenbosch uses Kikuyu grass. Kikuyu grass has thick, spongy roots and blades. When a golf ball lands from high up:
- The soft grass absorbs most of the downward energy, giving a small bounce of 20 to 40 centimeters instead of a huge bounce.
- Backspin grabs the grass and slows down forward motion.
- Rolling resistance on Kikuyu grass brings wedge shots to a stop within 1 to 3 meters, while driver shots roll 12 to 18 meters.

---

## 4. How the Models Work Together

1. **Step 1 (Physics Baseline)**: The physics model calculates an initial guess for apex, landing, and spin.
2. **Step 2 (Finding the Errors)**: We calculate the difference between the true measurements and the physics guess.
3. **Step 3 (Machine Learning)**: The tree models and neural network learn to predict these small differences.
4. **Step 4 (Final Blend)**: We add the corrections to the physics guess to get the final predictions.

---

## 5. Test Results (5-Fold Cross-Validation)

| Measurement | Physics Alone (MAE) | Tree Models (MAE) | Neural Net (MAE) | Combined Final (MAE) | Final R^2 Score |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **launch_spin_rate** (RPM) | 1360.58 | 763.74 | 778.19 | **730.46** | **0.8295** |
| **apex_t** (seconds) | 0.3343 | 0.1215 | 0.1011 | **0.0961** | **0.9428** |
| **apex_x** (meters) | 10.8238 | 3.7286 | 2.9034 | **2.7481** | **0.9816** |
| **apex_y** (meters) | 5.4257 | 2.0400 | 1.5139 | **1.4489** | **0.9850** |
| **apex_z** (meters) | 3.9195 | 1.1765 | 1.0103 | **0.9292** | **0.9754** |
| **landing_t** (seconds) | 0.6258 | 0.2004 | 0.1882 | **0.1723** | **0.9472** |
| **landing_x** (meters) | 18.1355 | 4.9235 | 4.2769 | **3.9409** | **0.9787** |
| **landing_y** (meters) | 9.3260 | 3.4745 | 2.9026 | **2.8417** | **0.9728** |
| **landing_z** (meters) | 0.0761 | 0.0789 | 0.0818 | **0.0761** | **0.9976** |

The combined model performs significantly better than any single model on its own.

---

## 6. Interactive 3D Visualizer

I developed an interactive 3D visualizer suite:
- **Interactive 3D Golf Flight Simulator (`visualizer/golf_simulator_3d.html` / `visualizer/index.html`)**:
  - Realistic 3D flight physics built on Three.js and custom shaders.
  - Exact 60m netted range cylindrical arc boundary, checkpoint gates (15m, 30m, 45m, 60m), and elevated bay platforms.
  - Multi-hop Kikuyu turf bounce and rollout dynamics with impact shockwave ripples.
  - Dynamic cinematic camera controls (Follow Cam, TV Cam, Net Cam, Green Cam).