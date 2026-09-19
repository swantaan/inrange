"""
src/features.py

Feature engineering module for the Inrange Golf Trajectory competition.
Extracts physically motivated kinematics, energies, aerodynamic proxies,
and multi-checkpoint polynomial features from launch monitors and checkpoints.
"""

import numpy as np
import pandas as pd

TARGET_COLUMNS = [
    'launch_spin_rate',
    'apex_t', 'apex_x', 'apex_y', 'apex_z',
    'landing_t', 'landing_x', 'landing_y', 'landing_z'
]


def extract_ml_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates helpful features from ball launch conditions and checkpoints.
    
    Features extracted:
    1. Launch speed, 3D velocity vectors, kinetic energy, elevation and azimuth angles.
    2. Dimensionless club regime ratios (separating drivers vs wedges).
    3. Segment-wise velocities, accelerations, and aerodynamic proxies (15m, 30m, 45m, 60m net).
    4. Cumulative deceleration and energy dissipation up to the 60m net.
    5. Polynomial trajectory fits predicting analytical apex and landing bounds.
    """
    feats = pd.DataFrame(index=df.index)

    # 1. Launch speed and angles
    vx0 = df['launch_vx']
    vy0 = df['launch_vy']
    vz0 = df['launch_vz']
    vxy0 = np.sqrt(vx0**2 + vy0**2)
    vtot0 = np.sqrt(vxy0**2 + vz0**2)

    feats['launch_speed'] = vtot0
    feats['launch_vxy'] = vxy0
    feats['launch_vz'] = vz0
    feats['launch_elevation_angle'] = np.arctan2(vz0, vxy0)
    feats['launch_azimuth_angle'] = np.arctan2(vy0, vx0)
    feats['launch_ke'] = 0.5 * vtot0**2
    feats['launch_ke_z'] = 0.5 * vz0**2
    feats['launch_ke_xy'] = 0.5 * vxy0**2
    feats['launch_vz_ratio'] = vz0 / (vtot0 + 1e-8)

    # Club type indicator: separates fast low-spin drivers from slower high-spin wedges
    feats['regime_launch_efficiency'] = vz0 / (vtot0**2 + 1e-6)
    feats['regime_ratio_vxy'] = vz0 / (vxy0 + 1e-6)
    feats['kinetic_loft_ratio'] = (vz0**2) / (vtot0**2 + 1e-6)
    feats['kinetic_efficiency_xy'] = (vxy0**2) / (vtot0**2 + 1e-6)

    feats['launch_x'] = df['launch_x']
    feats['launch_y'] = df['launch_y']
    feats['launch_z'] = df['launch_z']
    feats['is_upper_deck'] = (df['launch_z'] > 2.0).astype(float)

    # 2. Checkpoint Intervals (0->1, 1->2, 2->3, 3->4)
    cp_times = [np.zeros(len(df)), df['cp1_t'], df['cp2_t'], df['cp3_t'], df['cp4_t']]
    cp_xs = [df['launch_x'], df['cp1_x'], df['cp2_x'], df['cp3_x'], df['cp4_x']]
    cp_ys = [df['launch_y'], df['cp1_y'], df['cp2_y'], df['cp3_y'], df['cp4_y']]
    cp_zs = [df['launch_z'], df['cp1_z'], df['cp2_z'], df['cp3_z'], df['cp4_z']]

    prev_vx = vx0.copy()
    prev_vy = vy0.copy()
    prev_vz = vz0.copy()

    for i in range(1, 5):
        dt = cp_times[i] - cp_times[i - 1]
        dx = cp_xs[i] - cp_xs[i - 1]
        dy = cp_ys[i] - cp_ys[i - 1]
        dz = cp_zs[i] - cp_zs[i - 1]
        dxy = np.sqrt(dx**2 + dy**2)
        d3d = np.sqrt(dxy**2 + dz**2)

        vx_seg = dx / (dt + 1e-8)
        vy_seg = dy / (dt + 1e-8)
        vz_seg = dz / (dt + 1e-8)
        vxy_seg = np.sqrt(vx_seg**2 + vy_seg**2)
        vtot_seg = np.sqrt(vxy_seg**2 + vz_seg**2)

        feats[f'dt_seg{i}'] = dt
        feats[f'dx_seg{i}'] = dx
        feats[f'dy_seg{i}'] = dy
        feats[f'dz_seg{i}'] = dz
        feats[f'dxy_seg{i}'] = dxy
        feats[f'd3d_seg{i}'] = d3d

        feats[f'vx_seg{i}'] = vx_seg
        feats[f'vy_seg{i}'] = vy_seg
        feats[f'vz_seg{i}'] = vz_seg
        feats[f'vxy_seg{i}'] = vxy_seg
        feats[f'vtot_seg{i}'] = vtot_seg

        # Accelerations
        ax_seg = (vx_seg - prev_vx) / (dt + 1e-8)
        ay_seg = (vy_seg - prev_vy) / (dt + 1e-8)
        az_seg = (vz_seg - prev_vz) / (dt + 1e-8)
        feats[f'ax_seg{i}'] = ax_seg
        feats[f'ay_seg{i}'] = ay_seg
        feats[f'az_seg{i}'] = az_seg

        # Lift & Drag Proxies
        feats[f'lift_proxy_seg{i}'] = az_seg + 9.81
        feats[f'drag_proxy_seg{i}'] = (prev_vx**2 + prev_vy**2)**0.5 - vxy_seg

        # Curvature
        feats[f'curvature_seg{i}'] = (vx_seg * ay_seg - vy_seg * ax_seg) / (vxy_seg**3 + 1e-8)

        prev_vx = vx_seg
        prev_vy = vy_seg
        prev_vz = vz_seg

    # 3. Cumulative Launch to Checkpoint 4 (The 60m Net)
    tot_dt = df['cp4_t']
    feats['tot_dt_to_net'] = tot_dt
    feats['tot_dx_to_net'] = df['cp4_x'] - df['launch_x']
    feats['tot_dy_to_net'] = df['cp4_y'] - df['launch_y']
    feats['tot_dz_to_net'] = df['cp4_z'] - df['launch_z']
    feats['tot_dxy_to_net'] = np.sqrt(feats['tot_dx_to_net']**2 + feats['tot_dy_to_net']**2)
    feats['net_speed_loss'] = vtot0 - feats['vtot_seg4']
    feats['net_speed_ratio'] = feats['vtot_seg4'] / (vtot0 + 1e-8)
    feats['drag_decel_rate'] = (vtot0 - feats['vtot_seg4']) / (feats['tot_dt_to_net'] + 1e-6)
    feats['quintavalla_induced_proxy'] = 0.85 * (np.maximum(0.0, feats['lift_proxy_seg4']) / 9.81)**2

    feats['cp4_x'] = df['cp4_x']
    feats['cp4_y'] = df['cp4_y']
    feats['cp4_z'] = df['cp4_z']

    # 4. Polynomial Fit Features Across Checkpoints
    poly_apex_t = []
    poly_apex_z = []
    poly_apex_x = []
    poly_apex_y = []
    poly_land_t = []
    poly_land_x = []
    poly_land_y = []

    for idx in range(len(df)):
        t_arr = np.array([0.0, df.iloc[idx]['cp1_t'], df.iloc[idx]['cp2_t'], df.iloc[idx]['cp3_t'], df.iloc[idx]['cp4_t']])
        x_arr = np.array([df.iloc[idx]['launch_x'], df.iloc[idx]['cp1_x'], df.iloc[idx]['cp2_x'], df.iloc[idx]['cp3_x'], df.iloc[idx]['cp4_x']])
        y_arr = np.array([df.iloc[idx]['launch_y'], df.iloc[idx]['cp1_y'], df.iloc[idx]['cp2_y'], df.iloc[idx]['cp3_y'], df.iloc[idx]['cp4_y']])
        z_arr = np.array([df.iloc[idx]['launch_z'], df.iloc[idx]['cp1_z'], df.iloc[idx]['cp2_z'], df.iloc[idx]['cp3_z'], df.iloc[idx]['cp4_z']])

        z_ground = float(df.iloc[idx]['launch_z'])

        # Fit quadratic for z(t) = a*t^2 + b*t + c
        pz = np.polyfit(t_arr, z_arr, 2)
        px = np.polyfit(t_arr, x_arr, 2)
        py = np.polyfit(t_arr, y_arr, 2)

        # Apex: t = -b / (2a)
        if pz[0] < -0.1:
            t_ap = -pz[1] / (2.0 * pz[0])
            z_ap = pz[0] * t_ap**2 + pz[1] * t_ap + pz[2]
            x_ap = px[0] * t_ap**2 + px[1] * t_ap + px[2]
            y_ap = py[0] * t_ap**2 + py[1] * t_ap + py[2]

            # Landing: root of a*t^2 + b*t + (c - z_ground) = 0
            discriminant = pz[1]**2 - 4.0 * pz[0] * (pz[2] - z_ground)
            if discriminant >= 0:
                t_ld = (-pz[1] - np.sqrt(discriminant)) / (2.0 * pz[0])
                if t_ld < t_ap:
                    t_ld = (-pz[1] + np.sqrt(discriminant)) / (2.0 * pz[0])
            else:
                t_ld = 2.0 * t_ap

            x_ld = px[0] * t_ld**2 + px[1] * t_ld + px[2]
            y_ld = py[0] * t_ld**2 + py[1] * t_ld + py[2]
        else:
            t_ap, z_ap, x_ap, y_ap = 2.5, 20.0, 60.0, 70.0
            t_ld, x_ld, y_ld = 5.0, 110.0, 100.0

        poly_apex_t.append(t_ap)
        poly_apex_z.append(z_ap)
        poly_apex_x.append(x_ap)
        poly_apex_y.append(y_ap)
        poly_land_t.append(t_ld)
        poly_land_x.append(x_ld)
        poly_land_y.append(y_ld)

    feats['poly_apex_t'] = poly_apex_t
    feats['poly_apex_z'] = poly_apex_z
    feats['poly_apex_x'] = poly_apex_x
    feats['poly_apex_y'] = poly_apex_y
    feats['poly_land_t'] = poly_land_t
    feats['poly_land_x'] = poly_land_x
    feats['poly_land_y'] = poly_land_y

    return feats
