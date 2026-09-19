"""
src/physics.py

Physics-based flight and bounce simulation for the Inrange Golf Competition.
It calculates:
  - Air resistance and backspin lift holding the ball in the air.
  - Air pressure and density in Stellenbosch, South Africa.
  - How spin slowly drops during flight.
  - Ball apex (highest point) and landing location.
  - Realistic turf bounce and rollout on Stellenbosch Kikuyu grass.
"""

import os
import time
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Weather and atmospheric constants for Stellenbosch
# Elevation: about 136 meters above sea level
# Average temperature: 19.5 C
# Air is slightly thinner at this altitude, so the ball flies with less drag
ELEVATION_STELLENBOSCH = 136.0           # meters
TEMP_STELLENBOSCH_C = 19.5               # Celsius
TEMP_STELLENBOSCH_K = 273.15 + TEMP_STELLENBOSCH_C # 292.65 K
P_SEA_LEVEL = 101325.0                   # Pa
R_SPECIFIC_AIR = 287.058                 # J/(kg*K)
G_CONST = 9.80665                        # Gravity (m/s^2)

PRESSURE_STELLENBOSCH = P_SEA_LEVEL * (1.0 - 0.0065 * ELEVATION_STELLENBOSCH / 288.15) ** 5.25588 # ~997 hPa
RHO_STELLENBOSCH = PRESSURE_STELLENBOSCH / (R_SPECIFIC_AIR * TEMP_STELLENBOSCH_K) # ~1.1868 kg/m^3

# 2. Golf ball physical dimensions & properties
MASS = 0.04593                           # Ball mass in kg (about 45.9 grams)
RADIUS = 0.02135                         # Ball radius in meters
DIAMETER = 2.0 * RADIUS                  # Ball diameter in meters
AREA = np.pi * RADIUS**2                 # Ball cross-sectional area
AERO_FACTOR = 0.5 * RHO_STELLENBOSCH * AREA / MASS # Air drag factor
DYNAMIC_VISCOSITY_AIR = 1.81e-5          # Air viscosity
SPIN_DECAY_TAU = 15.0                    # Spin decay rate in seconds
K_INDUCED_DRAG = 0.85                    # Extra drag from backspin lift


class InstantaneousAeroNeuralNet:
    """
    Helper neural network to calculate smooth air resistance and lift
    at each step of flight based on ball speed, height, angle, and spin.
    """
    def __init__(self, radius=RADIUS, k_induced=K_INDUCED_DRAG, random_state=42):
        self.radius = radius
        self.k_induced = k_induced
        rng = np.random.RandomState(random_state)
        # Pre-initialized weights
        self.W1 = rng.randn(4, 32) * 0.05
        self.b1 = np.zeros(32)
        self.W2 = rng.randn(32, 32) * 0.05
        self.b2 = np.zeros(32)
        self.W3 = rng.randn(32, 16) * 0.05
        self.b3 = np.zeros(16)
        self.W4 = rng.randn(16, 2) * 0.02
        self.b4 = np.zeros(2)

    @staticmethod
    def _gelu(x):
        return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * (x**3))))

    def forward(self, speed, elevation, pitch_rad, spin_param, cd_scale=1.0, cl_scale=1.0):
        # 1. Base lift and drag formulas
        cl_base = spin_param / (0.85 + 1.25 * spin_param + 1e-6)
        cl_phys = cl_base * cl_scale

        cd_base = 0.215 + 0.070 / (1.0 + (speed / 32.0)**2)
        cd_induced = self.k_induced * (cl_phys**2)
        cd_phys = (cd_base + cd_induced) * cd_scale

        # 2. Input features: speed, height, pitch, and spin ratio
        x_norm = np.array([
            (speed - 45.0) / 25.0,
            (elevation - 15.0) / 20.0,
            pitch_rad,
            (spin_param - 0.15) / 0.15
        ], dtype=float)

        # 3. Neural net calculation
        h1 = self._gelu(x_norm @ self.W1 + self.b1)
        h2 = self._gelu(h1 @ self.W2 + self.b2)
        h3 = self._gelu(h2 @ self.W3 + self.b3)
        nn_out = h3 @ self.W4 + self.b4

        # 4. Final drag and lift values
        cd_out = float(np.clip(cd_phys + float(np.tanh(nn_out[0]) * 0.012), 0.18, 0.52))
        cl_out = float(np.clip(cl_phys + float(np.tanh(nn_out[1]) * 0.012), -0.05, 0.46))
        return cd_out, cl_out


class GolfBallPhysicsSimulator:
    """
    Simulates the 3D flight of a golf ball using drag, lift, and gravity.
    """

    def __init__(self,
                 rho=RHO_STELLENBOSCH,
                 mass=MASS,
                 radius=RADIUS,
                 g=G_CONST,
                 pressure=PRESSURE_STELLENBOSCH,
                 temperature_c=TEMP_STELLENBOSCH_C,
                 tau_spin=SPIN_DECAY_TAU,
                 k_induced=K_INDUCED_DRAG):
        self.elevation = ELEVATION_STELLENBOSCH
        self.temperature_c = temperature_c
        self.pressure = pressure
        self.rho = rho
        self.mass = mass
        self.radius = radius
        self.area = np.pi * radius**2
        self.g = g
        self.aero_factor = 0.5 * self.rho * self.area / self.mass
        self.tau_spin = tau_spin
        self.k_induced = k_induced
        self.aero_nn = InstantaneousAeroNeuralNet(radius=self.radius, k_induced=self.k_induced)

    def dimple_drag_coefficient(self, speed, cl_val, cd_scale=1.0):
        cd_base = 0.215 + 0.070 / (1.0 + (speed / 32.0)**2)
        cd_induced = self.k_induced * (cl_val**2)
        cd_total = (cd_base + cd_induced) * cd_scale
        return float(np.clip(cd_total, 0.18, 0.52))

    def dimple_lift_coefficient(self, speed, spin_omega, cl_scale=1.0):
        spin_param = (self.radius * spin_omega) / (speed + 1e-6)
        cl_base = spin_param / (0.85 + 1.25 * spin_param + 1e-6)
        cl_total = cl_base * cl_scale
        return float(np.clip(cl_total, -0.05, 0.46))

    def estimate_aero_coefficients(self, row):
        """
        Estimates air resistance and launch spin using speed and the 4 checkpoints.
        """
        t_pts = np.array([0.0, row['cp1_t'], row['cp2_t'], row['cp3_t'], row['cp4_t']], dtype=float)
        p_pts = np.array([
            [row['launch_x'], row['launch_y'], row['launch_z']],
            [row['cp1_x'], row['cp1_y'], row['cp1_z']],
            [row['cp2_x'], row['cp2_y'], row['cp2_z']],
            [row['cp3_x'], row['cp3_y'], row['cp3_z']],
            [row['cp4_x'], row['cp4_y'], row['cp4_z']]
        ], dtype=float)

        try:
            poly_x = np.polyfit(t_pts, p_pts[:, 0], 2)
            poly_y = np.polyfit(t_pts, p_pts[:, 1], 2)
            poly_z = np.polyfit(t_pts, p_pts[:, 2], 2)

            ax = 2.0 * poly_x[0]
            ay = 2.0 * poly_y[0]
            az = 2.0 * poly_z[0]

            t_mid = float(row['cp2_t'])
            vx_mid = 2.0 * poly_x[0] * t_mid + poly_x[1]
            vy_mid = 2.0 * poly_y[0] * t_mid + poly_y[1]
            vz_mid = 2.0 * poly_z[0] * t_mid + poly_z[1]
            v_mid = np.array([vx_mid, vy_mid, vz_mid], dtype=float)

            spd = np.linalg.norm(v_mid) + 1e-8
            v_xy = np.sqrt(vx_mid**2 + vy_mid**2) + 1e-8

            a_aero = np.array([ax, ay, az + self.g], dtype=float)

            u_drag = -v_mid / spd
            u_lift = np.array([-vx_mid * vz_mid, -vy_mid * vz_mid, v_xy**2]) / (v_xy * spd)
            u_side = np.array([-vy_mid, vx_mid, 0.0]) / v_xy

            a_drag = np.dot(a_aero, u_drag)
            a_lift = np.dot(a_aero, u_lift)
            a_side = np.dot(a_aero, u_side)

            dyn_press = self.aero_factor * (spd**2) + 1e-8
            cd_obs = float(np.clip(a_drag / dyn_press, 0.16, 0.48))
            cl_obs = float(np.clip(a_lift / dyn_press, -0.05, 0.42))
            cs_obs = float(np.clip(a_side / dyn_press, -0.15, 0.15))
        except Exception:
            cd_obs, cl_obs, cs_obs = 0.28, 0.22, 0.0

        v0 = np.array([row['launch_vx'], row['launch_vy'], row['launch_vz']], dtype=float)
        v0_mag = np.linalg.norm(v0)

        # Club-regime inference prior: vz / v0^2 strongly separates drivers from wedges
        regime_prior = row['launch_vz'] / (v0_mag**2 + 1e-6)
        est_spin_rpm = float(np.clip(
            regime_prior * 750000.0 + (cl_obs / (1.25 - cl_obs + 1e-6)) * (v0_mag / self.radius) * (60.0 / (2.0 * np.pi)) * 0.25 + 1800.0,
            1500.0, 11500.0
        ))

        cd_scale = cd_obs / 0.28
        cl_scale = cl_obs / 0.22 if cl_obs > 0 else 1.0

        return cd_scale, cl_scale, cs_obs, est_spin_rpm

    def simulate_flight(self, p0, v0, cd_scale, cl_scale, cs_obs, initial_spin_rpm, max_t=9.0, dt=0.01):
        """
        Integrates 3D flight trajectory with instantaneous dimpled drag,
        Quintavalla quadratic induced drag, and continuous spin decay.
        """
        p = np.array(p0, dtype=float)
        v = np.array(v0, dtype=float)
        z_ground = float(p0[2])

        initial_omega = initial_spin_rpm * (2.0 * np.pi / 60.0)

        max_steps = int(max_t / dt)
        t_arr = np.zeros(max_steps + 1)
        p_arr = np.zeros((max_steps + 1, 3))
        v_arr = np.zeros((max_steps + 1, 3))

        t_arr[0] = 0.0
        p_arr[0] = p
        v_arr[0] = v

        max_z = p[2]
        apex_idx = 0
        landed = False
        land_idx = max_steps

        for step in range(max_steps):
            t = step * dt
            curr_omega = initial_omega * np.exp(-t / self.tau_spin)

            spd = np.sqrt(v[0]**2 + v[1]**2 + v[2]**2) + 1e-8
            v_xy = np.sqrt(v[0]**2 + v[1]**2) + 1e-8

            theta_rad = np.arctan2(v[2], v_xy)
            spin_param = (self.radius * curr_omega) / (spd + 1e-6)
            cd_actual, cl_actual = self.aero_nn.forward(spd, p[2], theta_rad, spin_param, cd_scale, cl_scale)

            # Acceleration components
            a_d = -self.aero_factor * cd_actual * spd * v
            u_L = np.array([-v[0]*v[2], -v[1]*v[2], v_xy**2]) / (v_xy * spd)
            a_L = self.aero_factor * cl_actual * (spd**2) * u_L
            u_S = np.array([-v[1], v[0], 0.0]) / v_xy
            a_S = self.aero_factor * cs_obs * (spd**2) * u_S

            acc = a_d + a_L + a_S + np.array([0.0, 0.0, -self.g])

            # Heun predictor-corrector step
            v_next = v + acc * dt
            p_next = p + 0.5 * (v + v_next) * dt

            p = p_next
            v = v_next
            t_next = (step + 1) * dt

            t_arr[step + 1] = t_next
            p_arr[step + 1] = p
            v_arr[step + 1] = v

            if p[2] > max_z:
                max_z = p[2]
                apex_idx = step + 1

            if (step > apex_idx + 5) and (p[2] <= z_ground):
                landed = True
                land_idx = step + 1
                break

        t_arr = t_arr[:land_idx + 1]
        p_arr = p_arr[:land_idx + 1]
        v_arr = v_arr[:land_idx + 1]

        # Ground impact interpolation
        if landed and land_idx > 0:
            z1, z2 = p_arr[land_idx - 1, 2], p_arr[land_idx, 2]
            frac = (z_ground - z1) / (z2 - z1 + 1e-9)
            frac = np.clip(frac, 0.0, 1.0)
            t_land = t_arr[land_idx - 1] + frac * dt
            p_land = p_arr[land_idx - 1] + frac * (p_arr[land_idx] - p_arr[land_idx - 1])
            v_land = v_arr[land_idx - 1] + frac * (v_arr[land_idx] - v_arr[land_idx - 1])
        else:
            t_land = t_arr[-1]
            p_land = p_arr[-1]
            v_land = v_arr[-1]

        t_apex = t_arr[apex_idx]
        p_apex = p_arr[apex_idx]

        return {
            't': t_arr,
            'pos': p_arr,
            'vel': v_arr,
            'apex_t': float(t_apex),
            'apex_x': float(p_apex[0]),
            'apex_y': float(p_apex[1]),
            'apex_z': float(p_apex[2]),
            'landing_t': float(t_land),
            'landing_x': float(p_land[0]),
            'landing_y': float(p_land[1]),
            'landing_z': float(p_land[2]),
            'impact_vel': v_land
        }

    def simulate_bounce_and_roll(self, p_land, v_land, z_ground, spin_rpm=None, max_bounces=2, dt=0.01):
        """
        Simulates ground bounce and rollout on Stellenbosch Kikuyu grass:
        - Soft grass absorbs most downward energy, resulting in a low bounce.
        - Backspin grabs the grass to slow the ball down.
        - Natural grass friction brings the ball to a smooth stop.
        """
        p = np.array(p_land, dtype=float)
        v = np.array(v_land, dtype=float)

        bounce_t = [0.0]
        bounce_pos = [p.copy()]

        v_xy_mag = np.sqrt(v[0]**2 + v[1]**2) + 1e-8
        descent_angle_deg = np.degrees(np.arctan2(abs(v[2]), v_xy_mag))

        if descent_angle_deg > 45.0:
            restitution_z = 0.13
        elif descent_angle_deg > 30.0:
            restitution_z = 0.17
        else:
            restitution_z = 0.22

        # Backspin slows down forward speed upon impact
        spin_val = float(spin_rpm) if spin_rpm is not None else 6000.0
        spin_bite = np.clip((spin_val / 8500.0) * 0.35, 0.10, 0.45)
        tangential_retention = np.clip(1.0 - (0.50 + spin_bite), 0.15, 0.45)

        v[2] = abs(v[2]) * restitution_z
        v[0] *= tangential_retention
        v[1] *= tangential_retention

        curr_t = 0.0
        bounce_count = 0

        while bounce_count < max_bounces and v[2] > 0.4:
            while v[2] > 0 or p[2] > z_ground:
                v[2] -= self.g * dt
                p += v * dt
                curr_t += dt
                bounce_t.append(curr_t)
                bounce_pos.append(p.copy())
                if p[2] <= z_ground and v[2] < 0:
                    p[2] = z_ground
                    break

            v[2] = abs(v[2]) * (restitution_z * 0.5)
            v[0] *= 0.60
            v[1] *= 0.60
            bounce_count += 1

        p[2] = z_ground
        v[2] = 0.0

        # Rolling rollout on Kikuyu fairway grass
        mu_roll = 0.28
        v_xy = np.sqrt(v[0]**2 + v[1]**2)
        u_dir = np.array([v[0], v[1], 0.0]) / (v_xy + 1e-8)

        while v_xy > 0.05 and curr_t < 12.0:
            v_xy -= mu_roll * self.g * dt
            if v_xy < 0.0:
                v_xy = 0.0
            p += u_dir * v_xy * dt
            p[2] = z_ground
            curr_t += dt
            bounce_t.append(curr_t)
            bounce_pos.append(p.copy())

        return np.array(bounce_t), np.array(bounce_pos)

    def predict_shot(self, row):
        cd_scale, cl_scale, cs_obs, est_spin = self.estimate_aero_coefficients(row)
        p0 = [row['launch_x'], row['launch_y'], row['launch_z']]
        v0 = [row['launch_vx'], row['launch_vy'], row['launch_vz']]

        flight = self.simulate_flight(p0, v0, cd_scale, cl_scale, cs_obs, est_spin)

        return {
            'launch_spin_rate': est_spin,
            'apex_t': flight['apex_t'],
            'apex_x': flight['apex_x'],
            'apex_y': flight['apex_y'],
            'apex_z': flight['apex_z'],
            'landing_t': flight['landing_t'],
            'landing_x': flight['landing_x'],
            'landing_y': flight['landing_y'],
            'landing_z': flight['landing_z'],
            'cd_scale': cd_scale,
            'cl_scale': cl_scale,
            'cs_obs': cs_obs
        }

    def predict_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        records = []
        for idx in range(len(df)):
            row = df.iloc[idx]
            pred = self.predict_shot(row)
            pred['track_id'] = row['track_id']
            records.append(pred)
        return pd.DataFrame(records)
