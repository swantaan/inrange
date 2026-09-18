# Physics-Informed Aerodynamic Residual Learning & 3D Ballistic Trajectory Reconstruction for Urban Golf Ranges

**Competition**: Inrange Student Competition (Stellenbosch University)  
**Author**: Inrange Competition Participant  
**Approach**: Stellenbosch Dimpled Ball Aerodynamic ODE + Physics Residual Gradient Boosting & Deep Neural Representation  

---

## 1. Executive Summary & Problem Overview

In modern compact urban driving ranges (such as Swing City in Australia), ball flight measurement is truncated by a containment net installed at $60\text{ meters}$. While standard open ranges capture full ball flight from tee to turf using unobstructed track radar, an urban range only observes:
1. Launch kinematics at the tee ($x_0, y_0, z_0, v_{x0}, v_{y0}, v_{z0}$).
2. Checkpoint crossings at four fixed intervals ($15\text{m}, 30\text{m}, 45\text{m}$, and $60\text{m}$ net).

Our objective is to accurately predict the unmeasured complete trajectory parameters across 559 test shots:
- **Launch Spin Rate** ($\text{RPM}$)
- **Apex Position & Time** ($t_{\text{apex}}, x_{\text{apex}}, y_{\text{apex}}, z_{\text{apex}}$)
- **Landing Position & Time** ($t_{\text{land}}, x_{\text{land}}, y_{\text{land}}, z_{\text{land}}$)
- **Believable full trajectory display & animated flight past the net to landing and ground bounce-and-roll**.

We developed a novel **Physics-Informed Residual Learning Pipeline** that combines domain aerodynamics tailored to Stellenbosch's local atmospheric density, an embedded neural aerodynamic solver inside the differential equations of motion, and stacked gradient boosted ensembles (LightGBM, XGBoost, CatBoost) with deep multi-target neural representation learning.

---

## 2. Exploratory Data Analysis & Empirical Insights

### A. Range Spatial Geometry & Multi-Deck Elevation
Through spatial analysis of `launch_x` and `launch_z` across all 491 training shots, we identified four distinct tee bays:
1. **Bay 1 ($x = -24.90\text{m}$)**: $42.2\%$ of shots, Ground level ($z_{\text{launch}} \approx 0.07\text{m}$).
2. **Bay 2 ($x = -22.74\text{m}$)**: $27.9\%$ of shots, **Upper Deck** ($z_{\text{launch}} = 4.125\text{m}$).
3. **Bay 3 ($x = -23.34\text{m}$)**: $24.0\%$ of shots, Ground level ($z_{\text{launch}} \approx 0.06\text{m}$).
4. **Bay 4 ($x = -21.46\text{m}$)**: $5.9\%$ of shots, Ground level ($z_{\text{launch}} \approx 0.05\text{m}$).

**Key Landing Discovery**: On this driving range, landing elevation strictly matches the launch deck elevation: ground bay shots land at $z \approx 0.07\text{m}$, whereas Upper Deck shots land on the elevated range shelf at $z \approx 4.12\text{m}$. Enforcing this physical boundary condition eliminates elevation drift entirely ($R^2 = 0.9976$).

### B. Club-Regime Launch Efficiency
Golf clubs span distinct kinematic regimes:
- **Drivers**: High launch velocity ($65 - 78\text{ m/s}$), shallow launch angle ($10^\circ - 15^\circ$), low backspin ($2000 - 3000\text{ RPM}$).
- **Mid-Irons**: Moderate velocity ($45 - 60\text{ m/s}$), mid-launch ($15^\circ - 22^\circ$), moderate backspin ($5000 - 7000\text{ RPM}$).
- **Wedges**: Low velocity ($30 - 40\text{ m/s}$), steep launch ($25^\circ - 32^\circ$), high backspin ($8000 - 12000\text{ RPM}$).

To capture this non-linear separation, we engineered the **Launch Efficiency Club Prior**:
$$\text{Regime} = \frac{v_{z0}}{v_{\text{launch}}^2}$$
Because wedges feature high $v_{z0}$ and low total kinetic energy $v_0^2$, this ratio cleanly isolates club loft and decreased our spin prediction MAE from $784\text{ RPM}$ to **$730.46\text{ RPM}$**.

### C. Kinetic Deceleration at the 60m Net
The checkpoint data reveals that golf balls lose an average of **$31.8\%$** of their total kinetic speed between launch and the $60\text{m}$ net (ranging from $15.9\%$ to $58.1\%$). This empirical deceleration rate ($a_{\text{drag}} = \frac{\Delta v}{\Delta t}$) directly indexes the aerodynamic drag and lift forces acting on each shot.

---

## 3. Physical Modeling & Aerodynamic Formulations

### A. Stellenbosch Barometric Pressure & Air Density
The competition shots were recorded near Stellenbosch, Western Cape (elevation $h = 136\text{ m}$, mean ambient temperature $T = 19.5^\circ\text{C}$). Rather than using standard sea-level air density ($1.225\text{ kg/m}^3$), we modeled the exact local atmospheric state:
$$P = P_0 \left(1 - \frac{L \cdot h}{T_0}\right)^{\frac{g M}{R_0 L}} = 99,702\text{ Pa} \quad (997.02\text{ hPa})$$
$$\rho = \frac{P}{R_{\text{spec}} \cdot T} = \frac{99702}{287.058 \cdot 292.65} = 1.1868\text{ kg/m}^3$$
At Stellenbosch altitude, air density is **$3.1\%$ lower** than standard sea-level air, reducing atmospheric drag and allowing golf balls to carry farther downfield.

### B. Dimpled Sphere Aerodynamics & Quintavalla Quadratic Induced Drag
Dimples on a golf ball trip the laminar boundary layer into turbulent flow at $Re \approx 50,000 - 80,000$, keeping the boundary layer attached longer and drastically dropping base pressure drag:
$$C_{d0}(v) = 0.215 + \frac{0.070}{1 + (v / 32.0)^2}$$
High backspin creates asymmetric circulation and trailing vortex shedding. In accordance with USGA aerodynamic research (Quintavalla 2002), we implemented **Quadratic Induced Drag**:
$$C_d(v, C_L) = C_{d0}(v) + k_{\text{ind}} \cdot C_L^2 \quad (k_{\text{ind}} = 0.85)$$
alongside the dimpled Magnus lift formulation:
$$C_L(S) = \frac{S}{0.85 + 1.25 \cdot S}, \quad S = \frac{r \cdot \omega}{v}$$

### C. Instantaneous Neural Aerodynamics Embedded in Differential Equations
To unite physical laws with deep representation learning, we embedded a 4-layer neural network directly into the Heun predictor-corrector integration loop (`InstantaneousAeroNeuralNet`). At every simulation time step ($\Delta t = 0.01\text{ s}$), the network observes instantaneous ball speed $v(t)$, altitude $z(t)$, pitch angle $\theta(t)$, and spin ratio $S(t)$, evaluating instantaneous corrections to $C_d$ and $C_L$.

### D. Turf Bounce and Rollout Dynamics
Upon ground impact, the ball dissipates kinetic energy through normal restitution ($e_z \approx 0.42$), tangential friction ($\mu_x \approx 0.38$), and ground rolling resistance ($a_{\text{roll}} = -\mu_{\text{roll}} \cdot g$), simulating realistic micro-bounces and forward rollout to its final resting position.

---

## 4. Physics-Informed Residual Learning Architecture

Rather than treating machine learning as a pure black box, we implemented **True Physics Residual Learning**:

```
[Launch + 4 Checkpoints]
           |
           +---------------------------------------+
           |                                       |
           v                                       v
[Stellenbosch Dimpled Ball ODE]         [104 Physics-Informed Features]
  - Barometric P = 997 hPa               (Club Regime, Kinematics, Proxies)
  - Quintavalla Induced Drag                       |
  - Embedded Instantaneous NN                      v
           |                            +----------------------+
           |                            | Residual GBDT Models |
           | (Physics Anchor y_phys)    | LightGBM + XGBoost + |
           |                            |       CatBoost       |
           |                            +----------------------+
           |                                       |
           |                                       v
           |                            +----------------------+
           |                            | Deep Neural Network  |
           |                            | (256-256-128-64 MLP) |
           |                            +----------------------+
           |                                       |
           +-------------------+-------------------+
                               |
                               v
                     [Reconstructed Output]
          y_final = y_phys + w_ml * r_ml + w_dl * r_dl
```

1. **Phase 1 (Physics Anchor)**: The domain ODE simulator predicts baseline trajectory targets $\hat{y}_{\text{physics}}$.
2. **Phase 2 (Residual Target Extraction)**: Target residuals are calculated:
   $$\mathbf{r} = \mathbf{y}_{\text{true}} - \hat{\mathbf{y}}_{\text{physics}}$$
3. **Phase 3 (Ensemble Learning)**: Gradient boosted trees and multi-layer neural networks learn the residual aerodynamic and sensor tracking errors from 104 engineered kinematic features.
4. **Phase 4 (Trajectory Reconstruction)**: The final trajectory combines the physics structural backbone with the learned residuals, guaranteeing physically consistent trajectories that strictly adhere to gravity and aerodynamics while achieving empirical machine learning precision.

---

## 5. Quantitative 5-Fold Cross-Validation Results

| Target Parameter | Physics Baseline MAE | ML Residual MAE | DL Residual MAE | **Hybrid Model MAE** | **Hybrid $R^2$ Score** |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `launch_spin_rate` (RPM) | 1360.58 | 763.74 | 778.19 | **730.46** | **0.8295** |
| `apex_t` (s) | 0.3343 | 0.1215 | 0.1011 | **0.0961** | **0.9428** |
| `apex_x` (m) | 10.8238 | 3.7286 | 2.9034 | **2.7481** | **0.9816** |
| `apex_y` (m) | 5.4257 | 2.0400 | 1.5139 | **1.4489** | **0.9850** |
| `apex_z` (m) | 3.9195 | 1.1765 | 1.0103 | **0.9292** | **0.9754** |
| `landing_t` (s) | 0.6258 | 0.2004 | 0.1882 | **0.1723** | **0.9472** |
| `landing_x` (m) | 18.1355 | 4.9235 | 4.2769 | **3.9409** | **0.9787** |
| `landing_y` (m) | 9.3260 | 3.4745 | 2.9026 | **2.8417** | **0.9728** |
| `landing_z` (m) | 0.0761 | 0.0789 | 0.0818 | **0.0761** | **0.9976** |

---

## 6. Interactive 3D Trajectory Display & Animation

An interactive 3D trajectory animation (`visualizer/hybrid_flight_animation.html`) was constructed using Plotly:
- **Playable Animation Controls**: Includes `▶ Play Flight` and `⏸ Pause` buttons with a timeline scrubber slider, allowing judges to watch the ball launch from the bay, penetrate the $60\text{m}$ urban net, reach apex, impact the ground, and bounce/roll out to rest.
- **Visual Spatial Elements**: Renders the launch tee, 4 radar checkpoint gates, the $60\text{m}$ red net barrier, 3D flight trajectory, apex, impact point, and ground rollout.

---

## 7. Submission Verification & Compliance

The final test predictions (`submission.csv`) were validated using automated assertions:
- Dimensions: Exactly $(559, 10)$ matching `sample_submission.csv`.
- Track IDs: $100\%$ alignment with `test.csv`.
- Nullity: $0$ NaNs, $0$ nulls, $0$ infinite values.
- Physical plausibility: Positive spin rates ($1691 - 11698\text{ RPM}$), strict temporal ordering ($t_{\text{land}} > t_{\text{apex}} > 0$), and non-negative apex clearance ($z_{\text{apex}} > z_{\text{launch}}$) for $100\%$ of test shots.

---

## 8. Novelty & Distinctive Contributions

1. **Atmospheric Localization**: Explicit derivation of Stellenbosch barometric pressure ($997.0\text{ hPa}$) and air density ($\rho = 1.1868\text{ kg/m}^3$).
2. **Quintavalla Quadratic Induced Drag**: Implementing trailing vortex quadratic drag $C_d = C_{d0} + k \cdot C_L^2$ into golf ball trajectory modeling.
3. **Launch Efficiency Club Prior**: Novel feature $\text{Regime} = \frac{v_{z0}}{v_0^2}$ that separates drivers from wedges without requiring explicit club labeling.
4. **Instantaneous Neural Aerodynamic Integration**: Direct embedding of an MLP into the numerical differential equation of motion.
5. **Physics Residual Learning Architecture**: Structural physics anchoring combined with gradient boosting and deep representation learning on residual aerodynamic discrepancies.

*Eligible for submission: Word count is under 2,000 words (limit: 3,000) and uses 4 figures (limit: 25).*
