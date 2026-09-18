"""
hybrid.py

Main pipeline for the Inrange Golf Competition.
Combines:
  1. Flight physics model for Stellenbosch conditions.
  2. Tree-based machine learning models (LightGBM, CatBoost).
  3. Deep learning neural network.
  4. Best weighting to combine all models.
  5. Interactive 3D visualizer showing both the ideal flight and radar-fitted path.
"""

import os
import sys
import time
import importlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import plotly.graph_objects as go

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

phys_module = importlib.import_module('physics-model')
ml_module = importlib.import_module('machine-learning-model')
dl_module = importlib.import_module('deep-learning-model')

GolfBallPhysicsSimulator = phys_module.GolfBallPhysicsSimulator
RHO_STELLENBOSCH = phys_module.RHO_STELLENBOSCH
PRESSURE_STELLENBOSCH = phys_module.PRESSURE_STELLENBOSCH
extract_ml_features = ml_module.extract_ml_features
TARGET_COLUMNS = ml_module.TARGET_COLUMNS
MLTrajectoryModel = ml_module.MLTrajectoryModel
DeepTrajectoryNetwork = dl_module.DeepTrajectoryNetwork


class HybridTrajectoryPipeline:
    """
    Main pipeline combining physics, machine learning, and deep learning.
    """

    def __init__(self, random_state=42):
        self.random_state = random_state
        self.simulator = GolfBallPhysicsSimulator()
        self.ml_model = MLTrajectoryModel(n_splits=5, random_state=random_state)
        self.dl_model = DeepTrajectoryNetwork(hidden_layer_sizes=(256, 256, 128, 64), n_splits=5, random_state=random_state)
        self.optimal_weights = {} # target -> (w_phys, w_ml, w_dl)
        self.oof_predictions = None

    def fit_and_evaluate(self, train_df):
        print("=" * 70, flush=True)
        print("INRANGE COMPETITION: HYBRID PIPELINE (STELLENBOSCH DIMPLED PHYSICS)", flush=True)
        print("=" * 70, flush=True)
        print(f"Stellenbosch atmospheric settings: P = {PRESSURE_STELLENBOSCH/100:.1f} hPa | rho = {RHO_STELLENBOSCH:.4f} kg/m^3", flush=True)

        results_dir = 'results'
        os.makedirs(results_dir, exist_ok=True)

        # 1. Physics Baseline & Residual Extraction
        print("\n[Phase 1/4] Running Stellenbosch Dimpled Physics Simulation Structural Anchor...", flush=True)
        t0 = time.time()
        phys_preds = self.simulator.predict_dataset(train_df)
        print(f"Physics simulation anchor evaluated in {time.time() - t0:.2f}s.", flush=True)

        # Compute physics residuals: r = y_true - y_physics
        print("Computing physics residuals (delta_y = y_true - y_physics)...", flush=True)
        residuals_train = train_df[TARGET_COLUMNS] - phys_preds[TARGET_COLUMNS]

        # 2. Machine Learning Residual Learning
        print("\n[Phase 2/4] Training Gradient Boosting Ensemble to predict Physics Residuals...", flush=True)
        t0 = time.time()
        X_train = extract_ml_features(train_df)
        self.ml_model.train_and_evaluate(X_train, residuals_train)
        ml_res_oof = self.ml_model.oof_predictions
        print(f"Gradient boosting residual model completed in {time.time() - t0:.2f}s.", flush=True)

        # 3. Deep Learning Residual Learning
        print("\n[Phase 3/4] Training Deep Multi-Layer Network to predict Physics Residuals...", flush=True)
        t0 = time.time()
        self.dl_model.train_and_evaluate(X_train, residuals_train)
        dl_res_oof = self.dl_model.oof_predictions
        print(f"Deep neural network residual model completed in {time.time() - t0:.2f}s.", flush=True)

        # 4. Physics Residual Reconstruction & Stacking
        print("\n[Phase 4/4] Reconstructing Trajectory from Physics Anchor + Learned Residuals...", flush=True)
        self.oof_predictions = pd.DataFrame(index=train_df.index)

        print("-" * 75, flush=True)
        print(f"{'Target':<18} | {'Phys MAE':<9} | {'Residual Weights':<18} | {'Hybrid MAE':<10} | {'R2 Score':<8}", flush=True)
        print("-" * 75, flush=True)

        scores = {}

        grid_weights = [
            (0.55, 0.45),
            (0.50, 0.50),
            (0.60, 0.40),
            (0.65, 0.35),
            (0.70, 0.30),
            (0.40, 0.60),
            (0.35, 0.65),
            (0.80, 0.20),
            (0.20, 0.80),
            (1.00, 0.00),
            (0.00, 1.00)
        ]

        for tgt in TARGET_COLUMNS:
            y_true = train_df[tgt].values
            p_phys = phys_preds[tgt].values
            r_ml = ml_res_oof[tgt].values
            r_dl = dl_res_oof[tgt].values

            phys_mae = mean_absolute_error(y_true, p_phys)

            best_mae = float('inf')
            best_w = (0.55, 0.45)
            best_pred = None

            for w_m, w_d in grid_weights:
                candidate = p_phys + w_m * r_ml + w_d * r_dl
                mae_cand = mean_absolute_error(y_true, candidate)
                if mae_cand < best_mae:
                    best_mae = mae_cand
                    best_w = (w_m, w_d)
                    best_pred = candidate

            if tgt == 'landing_z':
                best_pred = train_df['launch_z'].values

            self.optimal_weights[tgt] = best_w
            self.oof_predictions[tgt] = best_pred

            rmse = np.sqrt(mean_squared_error(y_true, best_pred))
            r2 = r2_score(y_true, best_pred)
            scores[tgt] = {'MAE': best_mae, 'RMSE': rmse, 'R2': r2, 'Weights': best_w, 'Phys_MAE': phys_mae}

            w_str = f"ML:{best_w[0]:.2f}, DL:{best_w[1]:.2f}"
            print(f"{tgt:<18} | {phys_mae:<9.4f} | {w_str:<18} | {best_mae:<10.4f} | {r2:<8.4f}", flush=True)

        return scores

    def predict(self, test_df):
        print("\nGenerating physics residual predictions on test set...", flush=True)
        phys_preds = self.simulator.predict_dataset(test_df)

        X_test = extract_ml_features(test_df)
        ml_res_preds = self.ml_model.predict(X_test)
        dl_res_preds = self.dl_model.predict(X_test)

        test_sub = pd.DataFrame(index=test_df.index)
        test_sub['track_id'] = test_df['track_id']

        for tgt in TARGET_COLUMNS:
            w_m, w_d = self.optimal_weights[tgt]
            test_sub[tgt] = phys_preds[tgt].values + w_m * ml_res_preds[tgt].values + w_d * dl_res_preds[tgt].values

        test_sub['landing_z'] = test_df['launch_z'].values

        return test_sub

    def _build_shot_figure(self, sample_row, shot_num=None, is_prediction=True):
        """
        Builds the 3D Plotly figure with interactive playback controls for a single shot.
        """
        cd_scale, cl_scale, cs_obs, est_spin = self.simulator.estimate_aero_coefficients(sample_row)
        # If the sample row contains our model's predicted launch spin rate, use it!
        if 'launch_spin_rate' in sample_row and pd.notnull(sample_row['launch_spin_rate']):
            est_spin = float(sample_row['launch_spin_rate'])

        from scipy.interpolate import CubicSpline

        # Gather measured & predicted milestones: Launch -> CP1 -> CP2 -> CP3 -> CP4 -> Apex -> Landing
        p_launch = [float(sample_row['launch_x']), float(sample_row['launch_y']), float(sample_row['launch_z'])]
        p_cp1 = [float(sample_row['cp1_x']), float(sample_row['cp1_y']), float(sample_row['cp1_z'])]
        p_cp2 = [float(sample_row['cp2_x']), float(sample_row['cp2_y']), float(sample_row['cp2_z'])]
        p_cp3 = [float(sample_row['cp3_x']), float(sample_row['cp3_y']), float(sample_row['cp3_z'])]
        p_cp4 = [float(sample_row['cp4_x']), float(sample_row['cp4_y']), float(sample_row['cp4_z'])]

        # Apex and landing coordinates
        if 'apex_x' in sample_row and pd.notnull(sample_row['apex_x']):
            p_apex = [float(sample_row['apex_x']), float(sample_row['apex_y']), float(sample_row['apex_z'])]
            t_apex = float(sample_row['apex_t'])
        else:
            sim_flight = self.simulator.simulate_flight(p_launch, [sample_row['launch_vx'], sample_row['launch_vy'], sample_row['launch_vz']], cd_scale, cl_scale, cs_obs, est_spin, dt=0.01)
            p_apex = [sim_flight['apex_x'], sim_flight['apex_y'], sim_flight['apex_z']]
            t_apex = sim_flight['apex_t']

        if 'landing_x' in sample_row and pd.notnull(sample_row['landing_x']):
            p_landing = [float(sample_row['landing_x']), float(sample_row['landing_y']), float(sample_row.get('landing_z', sample_row['launch_z']))]
            t_landing = float(sample_row['landing_t'])
        else:
            sim_flight = self.simulator.simulate_flight(p_launch, [sample_row['launch_vx'], sample_row['launch_vy'], sample_row['launch_vz']], cd_scale, cl_scale, cs_obs, est_spin, dt=0.01)
            p_landing = [sim_flight['landing_x'], sim_flight['landing_y'], sim_flight['landing_z']]
            t_landing = sim_flight['landing_t']

        # 1. Physical flight path: smooth natural flight without radar noise ("How it should actually be")
        v_launch = [float(sample_row['launch_vx']), float(sample_row['launch_vy']), float(sample_row['launch_vz'])]
        sim_flight = self.simulator.simulate_flight(p_launch, v_launch, cd_scale, cl_scale, cs_obs, est_spin, dt=0.01)
        phys_t = sim_flight['t']
        phys_pos = sim_flight['pos']
        phys_bt, phys_bpos = self.simulator.simulate_bounce_and_roll(
            phys_pos[-1].tolist(),
            sim_flight['vel'][-1],
            sample_row['launch_z'],
            spin_rpm=est_spin
        )

        # 2. Radar-fitted path: forced through measured checkpoints to show the effect of radar noise
        milestones = [
            (0.0, p_launch),
            (float(sample_row['cp1_t']), p_cp1),
            (float(sample_row['cp2_t']), p_cp2),
            (float(sample_row['cp3_t']), p_cp3),
            (float(sample_row['cp4_t']), p_cp4),
            (t_apex, p_apex),
            (t_landing, p_landing)
        ]
        milestones.sort(key=lambda pt: pt[0])

        target_times = np.array([pt[0] for pt in milestones])
        target_coords = np.array([pt[1] for pt in milestones])

        target_tau = target_times / t_landing
        sim_tau = phys_t / phys_t[-1]

        phys_at_milestones = np.zeros((len(target_times), 3))
        for d in range(3):
            phys_at_milestones[:, d] = np.interp(target_tau, sim_tau, phys_pos[:, d])

        res_at_milestones = target_coords - phys_at_milestones
        res_spline = CubicSpline(target_tau, res_at_milestones, bc_type='natural')

        flight_t = np.sort(np.unique(np.concatenate([
            np.linspace(0.0, t_landing, 120),
            target_times
        ])))
        dense_tau = flight_t / t_landing

        dense_phys = np.zeros((len(flight_t), 3))
        for d in range(3):
            dense_phys[:, d] = np.interp(dense_tau, sim_tau, phys_pos[:, d])

        f_pos = dense_phys + res_spline(dense_tau)
        dt_impact = flight_t[-1] - flight_t[-2]
        v_impact = (f_pos[-1] - f_pos[-2]) / dt_impact

        b_t, b_pos = self.simulator.simulate_bounce_and_roll(
            f_pos[-1].tolist(),
            v_impact,
            sample_row['launch_z'],
            spin_rpm=est_spin
        )

        # 3. Measurement differences: lines connecting measured radar points to the pure physics path
        err_x, err_y, err_z = [], [], []
        milestones_info = [
            ('CP1', float(sample_row['cp1_t']), p_cp1),
            ('CP2', float(sample_row['cp2_t']), p_cp2),
            ('CP3', float(sample_row['cp3_t']), p_cp3),
            ('CP4', float(sample_row['cp4_t']), p_cp4),
            ('Apex', t_apex, p_apex),
            ('Landing', t_landing, p_landing)
        ]
        for name, t_m, pt_m in milestones_info:
            idx_m = np.argmin(np.abs(phys_t - t_m))
            pt_ideal = phys_pos[idx_m]
            err_x.extend([pt_m[0], pt_ideal[0], None])
            err_y.extend([pt_m[1], pt_ideal[1], None])
            err_z.extend([pt_m[2], pt_ideal[2], None])

        # Animation follows the Ideal Pure Aerodynamic flight path and Stellenbosch turf rollout
        full_pos = np.vstack([phys_pos, phys_bpos[1:]])
        full_t = np.concatenate([phys_t, phys_t[-1] + phys_bt[1:]])
        n_flight = len(phys_pos)

        fig = go.Figure()

        # Trace 0: Ideal Aerodynamic Flight Path (Pure Physics)
        fig.add_trace(go.Scatter3d(
            x=phys_pos[:, 0], y=phys_pos[:, 1], z=phys_pos[:, 2],
            mode='lines',
            line=dict(color='#00f0ff', width=5),
            name='Ideal Aerodynamic Path (Pure Physics - True Flight)'
        ))

        # Trace 1: Ideal Bounce & Rollout (Stellenbosch Turf)
        fig.add_trace(go.Scatter3d(
            x=phys_bpos[:, 0], y=phys_bpos[:, 1], z=phys_bpos[:, 2],
            mode='lines',
            line=dict(color='rgba(0, 240, 255, 0.60)', width=3, dash='dot'),
            name='Ideal Turf Rollout (Stellenbosch Turf)'
        ))

        # Trace 2: Radar-Constrained Path (Fitting Noisy Checkpoints)
        fig.add_trace(go.Scatter3d(
            x=f_pos[:, 0], y=f_pos[:, 1], z=f_pos[:, 2],
            mode='lines',
            line=dict(color='#ff7700', width=4, dash='dash'),
            name='Radar-Constrained Path (Fitting Noisy Checkpoints)'
        ))

        # Trace 3: Radar Path Rollout
        fig.add_trace(go.Scatter3d(
            x=b_pos[:, 0], y=b_pos[:, 1], z=b_pos[:, 2],
            mode='lines',
            line=dict(color='rgba(255, 119, 0, 0.50)', width=2, dash='dot'),
            name='Radar Path Rollout'
        ))

        # Trace 4: Measurement Error Vectors (Radar Gate Noise Offsets)
        fig.add_trace(go.Scatter3d(
            x=err_x, y=err_y, z=err_z,
            mode='lines',
            line=dict(color='rgba(255, 60, 60, 0.85)', width=3, dash='dot'),
            name='Measurement Error Vectors (Radar Gate Offsets Δ)'
        ))

        # Trace 5: Launch Tee
        fig.add_trace(go.Scatter3d(
            x=[p_launch[0]], y=[p_launch[1]], z=[p_launch[2]],
            mode='markers+text',
            marker=dict(size=9, color='#ff007f', symbol='diamond'),
            text=['Launch Tee'],
            textposition='top center',
            name='Launch Tee'
        ))

        # Trace 6: Checkpoints (15m, 30m, 45m)
        fig.add_trace(go.Scatter3d(
            x=[p_cp1[0], p_cp2[0], p_cp3[0]],
            y=[p_cp1[1], p_cp2[1], p_cp3[1]],
            z=[p_cp1[2], p_cp2[2], p_cp3[2]],
            mode='markers',
            marker=dict(size=8, color='#ffcc00', symbol='circle'),
            name='Checkpoints (15m, 30m, 45m)'
        ))

        # Trace 7: CP4 (60m Urban Net)
        fig.add_trace(go.Scatter3d(
            x=[p_cp4[0]], y=[p_cp4[1]], z=[p_cp4[2]],
            mode='markers+text',
            marker=dict(size=12, color='#ff2222', symbol='square'),
            text=['CP4 (60m Urban Net)'],
            textposition='top center',
            name='Urban Net (60m Barrier)'
        ))

        # Trace 8: Predicted Apex
        fig.add_trace(go.Scatter3d(
            x=[p_apex[0]], y=[p_apex[1]], z=[p_apex[2]],
            mode='markers+text',
            marker=dict(size=10, color='#aa00ff', symbol='cross'),
            text=[f"Predicted Apex ({p_apex[2]:.1f}m @ {t_apex:.2f}s)"],
            textposition='top center',
            name='Predicted Trajectory Apex'
        ))

        # Trace 9: Predicted Landing
        fig.add_trace(go.Scatter3d(
            x=[p_landing[0]], y=[p_landing[1]], z=[p_landing[2]],
            mode='markers+text',
            marker=dict(size=10, color='#00ff66', symbol='diamond'),
            text=[f"Predicted Landing ({p_landing[0]:.1f}m @ {t_landing:.2f}s)"],
            textposition='bottom center',
            name='Predicted Ground Impact'
        ))

        # Trace 10: Ideal Resting Position
        fig.add_trace(go.Scatter3d(
            x=[phys_bpos[-1, 0]], y=[phys_bpos[-1, 1]], z=[phys_bpos[-1, 2]],
            mode='markers+text',
            marker=dict(size=9, color='#00f0ff', symbol='circle'),
            text=[f"Ideal Rest ({phys_bpos[-1, 0]:.1f}m)"],
            textposition='bottom center',
            name='Ideal Resting Position'
        ))

        # Trace 11: Dynamic Animated Golf Ball
        fig.add_trace(go.Scatter3d(
            x=[full_pos[0, 0]], y=[full_pos[0, 1]], z=[full_pos[0, 2]],
            mode='markers+text',
            marker=dict(size=12, color='#ffffff', symbol='circle', line=dict(color='#00f0ff', width=3)),
            text=['Golf Ball (Launch)'],
            textposition='top center',
            name='Animated Golf Ball'
        ))

        # Trace 12: Dynamic Motion Trail
        fig.add_trace(go.Scatter3d(
            x=[full_pos[0, 0]], y=[full_pos[0, 1]], z=[full_pos[0, 2]],
            mode='lines',
            line=dict(color='#00f0ff', width=6),
            name='Live Motion Trail'
        ))

        # Build Animation Frames
        n_frames = 65
        frame_indices = np.linspace(0, len(full_pos) - 1, n_frames, dtype=int)
        frames = []
        slider_steps = []

        for i, idx in enumerate(frame_indices):
            is_flying = idx < n_flight
            t_curr = full_t[idx]
            status_text = f"Flight: {full_pos[idx, 0]:.1f}m (t={t_curr:.2f}s)" if is_flying else f"Rollout: {full_pos[idx, 0]:.1f}m (t={t_curr:.2f}s)"
            ball_col = '#00f0ff' if is_flying else '#00ff88'

            frame_data = [
                go.Scatter3d(
                    x=[full_pos[idx, 0]],
                    y=[full_pos[idx, 1]],
                    z=[full_pos[idx, 2]],
                    marker=dict(size=12, color=ball_col, line=dict(color='#ffffff', width=2)),
                    text=[status_text]
                ),
                go.Scatter3d(
                    x=full_pos[:idx + 1, 0],
                    y=full_pos[:idx + 1, 1],
                    z=full_pos[:idx + 1, 2],
                    line=dict(color='#00f0ff' if is_flying else '#00ff88', width=6)
                )
            ]

            frame_name = f"frame_{i}"
            frames.append(go.Frame(data=frame_data, traces=[11, 12], name=frame_name))

            slider_steps.append(dict(
                method="animate",
                label=f"{t_curr:.1f}s",
                args=[[frame_name], dict(mode="immediate", frame=dict(duration=25, redraw=True), transition=dict(duration=0))]
            ))

        fig.frames = frames

        shot_type = "Predicted Test Shot" if is_prediction else "Shot"
        shot_label = f"{shot_type} #{shot_num}" if shot_num is not None else f"Shot ID: {str(sample_row['track_id'])[:12]}"
        spin_label = f"Model Predicted Spin: {est_spin:.0f} RPM" if is_prediction else f"Spin: {est_spin:.0f} RPM"

        fig.update_layout(
            title=dict(
                text=f"Inrange 3D Flight Comparison: Ideal Aerodynamic Physics vs Radar Gate-Constrained Path<br><sup>{shot_label} | {spin_label} | Cyan: Pure Physics Flight | Orange Dash: Checkpoint Fit with Noise | Red Dots: Radar Errors Δ</sup>",
                font=dict(size=14, color='#ffffff')
            ),
            template='plotly_dark',
            paper_bgcolor='#0b0f19',
            scene=dict(
                xaxis_title='X - Downfield (m)',
                yaxis_title='Y - Lateral Deviation (m)',
                zaxis_title='Z - Elevation (m)',
                camera=dict(eye=dict(x=1.65, y=1.65, z=0.85)),
                aspectmode='data'
            ),
            legend=dict(x=0.01, y=0.98, bgcolor='rgba(15,23,42,0.88)'),
            updatemenus=[dict(
                type="buttons",
                direction="left",
                x=0.10, y=0.05,
                bgcolor="rgba(30, 41, 59, 0.9)",
                font=dict(color="#ffffff"),
                buttons=[
                    dict(
                        label="▶ Play Flight",
                        method="animate",
                        args=[None, dict(frame=dict(duration=35, redraw=True), fromcurrent=True, mode="immediate", transition=dict(duration=0))]
                    ),
                    dict(
                        label="⏸ Pause",
                        method="animate",
                        args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate", transition=dict(duration=0))]
                    )
                ]
            )],
            sliders=[dict(
                steps=slider_steps,
                active=0,
                x=0.10, y=0.0,
                len=0.80,
                font=dict(color="#ffffff"),
                currentvalue=dict(prefix="Flight Time: ", visible=True, font=dict(color="#00f0ff", size=13))
            )]
        )
        return fig

    def create_interactive_trajectory_animation(self, sample_row, output_html_path, df=None, is_prediction=True):
        """
        Generates interactive 3D trajectory animation allowing the user to select
        and visualize any predicted shot from the dataset.
        """
        import json

        shot_idx = 0
        if isinstance(sample_row, pd.Series):
            row = sample_row
            if df is not None:
                matches = df.index[df['track_id'] == row['track_id']].tolist()
                if matches:
                    shot_idx = matches[0]
        elif df is not None:
            if isinstance(sample_row, int) or (isinstance(sample_row, str) and sample_row.isdigit()):
                shot_idx = int(sample_row) % len(df)
                row = df.iloc[shot_idx]
            else:
                matches = df[df['track_id'].astype(str).str.contains(str(sample_row), case=False, na=False)]
                if len(matches) > 0:
                    shot_idx = matches.index[0]
                    row = matches.iloc[0]
                else:
                    shot_idx = 0
                    row = df.iloc[0]
        else:
            row = sample_row

        fig = self._build_shot_figure(row, shot_num=shot_idx, is_prediction=is_prediction)

        os.makedirs(os.path.dirname(output_html_path), exist_ok=True)

        if df is None or len(df) == 0:
            fig.write_html(output_html_path)
            print(f"Saved interactive 3D trajectory animation to {output_html_path}", flush=True)
            return

        # Pre-compute selectable shots (first 10 diverse shots + currently selected)
        n_selectable = min(10, len(df))
        selectable_indices = list(range(n_selectable))
        if shot_idx not in selectable_indices:
            selectable_indices.append(shot_idx)
            selectable_indices.sort()

        shots_cache = {}
        options_html = []

        for idx in selectable_indices:
            row_i = df.iloc[idx]
            fig_i = self._build_shot_figure(row_i, shot_num=idx, is_prediction=is_prediction)
            fig_dict = fig_i.to_dict()
            v_mag = np.sqrt(row_i['launch_vx']**2 + row_i['launch_vy']**2 + row_i['launch_vz']**2)
            spin_val = float(row_i['launch_spin_rate']) if 'launch_spin_rate' in row_i else 0.0
            carry_val = np.hypot(row_i['landing_x'] - row_i['launch_x'], row_i['landing_y'] - row_i['launch_y']) if 'landing_x' in row_i else 0.0

            prefix = "Predicted Shot" if is_prediction else "Shot"
            shot_label = f"{prefix} {idx}: {str(row_i['track_id'])[:8]} (Carry: {carry_val:.1f}m, Spin: {spin_val:.0f} RPM)"
            meta_info = f"{prefix} #{idx} | Track ID: {row_i['track_id']} | Speed: {v_mag:.1f} m/s | Pred Carry: {carry_val:.1f}m | Pred Spin: {spin_val:.0f} RPM"

            shots_cache[str(idx)] = {
                'data': fig_dict.get('data', []),
                'layout': fig_dict.get('layout', {}),
                'frames': fig_dict.get('frames', []),
                'meta': meta_info
            }

            selected_attr = 'selected' if idx == shot_idx else ''
            options_html.append(f'<option value="{idx}" {selected_attr}>{shot_label}</option>')

        base_html = fig.to_html(include_plotlyjs=True, full_html=True)

        v_init = np.sqrt(row['launch_vx']**2 + row['launch_vy']**2 + row['launch_vz']**2)
        spin_init = float(row['launch_spin_rate']) if 'launch_spin_rate' in row else 0.0
        carry_init = np.hypot(row['landing_x'] - row['launch_x'], row['landing_y'] - row['launch_y']) if 'landing_x' in row else 0.0
        prefix_init = "Predicted Test Shot" if is_prediction else "Shot"
        initial_meta = f"{prefix_init} #{shot_idx} | Track: {row['track_id'][:12]}... | Speed: {v_init:.1f} m/s | Pred Carry: {carry_init:.1f}m | Pred Spin: {spin_init:.0f} RPM"

        toolbar_title = "INRANGE 3D VISUALIZER (MODEL PREDICTIONS)" if is_prediction else "INRANGE 3D SHOT VISUALIZER"
        dropdown_label = "Choose Predicted Shot:" if is_prediction else "Choose Shot:"

        toolbar_html = f'''
        <div id="shot-selector-bar" style="position: sticky; top: 0; left: 0; right: 0; z-index: 99999; background: #0f172a; border-bottom: 2px solid #00f0ff; padding: 12px 24px; display: flex; align-items: center; justify-content: space-between; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; color: #ffffff; box-shadow: 0 4px 12px rgba(0,0,0,0.5);">
            <div style="display: flex; align-items: center; gap: 14px;">
                <span style="color: #00f0ff; font-weight: 700; font-size: 15px; letter-spacing: 0.5px;">{toolbar_title}</span>
                <label for="shot-select-dropdown" style="font-size: 13px; color: #94a3b8; font-weight: 600;">{dropdown_label}</label>
                <select id="shot-select-dropdown" onchange="onSelectShot(this.value)" style="background: #1e293b; color: #ffffff; border: 1px solid #38bdf8; padding: 7px 14px; border-radius: 6px; font-size: 13px; font-weight: 600; cursor: pointer; outline: none;">
                    {''.join(options_html)}
                </select>
            </div>
            <div id="shot-meta-badge" style="font-size: 12px; color: #38bdf8; background: rgba(56,189,248,0.12); padding: 5px 12px; border-radius: 6px; border: 1px solid rgba(56,189,248,0.3); font-family: monospace;">
                {initial_meta}
            </div>
        </div>
        '''

        shots_json_str = json.dumps(shots_cache)
        script_injection = f'''
        <script type="text/javascript">
            window.ALL_SHOTS_DATA = {shots_json_str};

            function onSelectShot(idxStr) {{
                var shot = window.ALL_SHOTS_DATA[idxStr];
                if (!shot) return;

                var gd = document.getElementsByClassName('plotly-graph-div')[0];
                if (!gd) return;

                Plotly.newPlot(gd, shot.data, shot.layout, {{responsive: true}}).then(function() {{
                    if (shot.frames && shot.frames.length > 0) {{
                        Plotly.addFrames(gd, shot.frames);
                    }}
                }});

                var badge = document.getElementById('shot-meta-badge');
                if (badge && shot.meta) {{
                    badge.innerText = shot.meta;
                }}
            }}
        </script>
        '''

        if '<body>' in base_html:
            final_html = base_html.replace('<body>', f'<body>\n{toolbar_html}', 1)
        else:
            final_html = toolbar_html + base_html

        if '</body>' in final_html:
            final_html = final_html.replace('</body>', f'{script_injection}\n</body>', 1)
        else:
            final_html += script_injection

        with open(output_html_path, 'w', encoding='utf-8') as f:
            f.write(final_html)

        print(f"Saved interactive multi-shot 3D visualizer to {output_html_path} (Active Shot #{shot_idx})", flush=True)


def run_hybrid_pipeline(shot_choice=None):
    print("=" * 70, flush=True)
    print("INRANGE COMPETITION: HYBRID PIPELINE EXECUTION", flush=True)
    print("=" * 70, flush=True)

    # Support command line shot choice: python src/hybrid.py --shot 5 or python src/hybrid.py 5
    if shot_choice is None:
        import sys
        for i, arg in enumerate(sys.argv):
            if arg in ('--shot', '-s') and i + 1 < len(sys.argv):
                shot_choice = sys.argv[i + 1]
                break
            elif i > 0 and (arg.isdigit() or len(arg) > 8) and not arg.startswith('-'):
                shot_choice = arg
                break
        if shot_choice is None:
            shot_choice = 0

    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, '..')) if os.path.basename(current_dir) == 'src' else current_dir

    train_path = os.path.join(project_root, 'data', 'train.csv')
    test_path = os.path.join(project_root, 'data', 'test.csv')
    results_dir = os.path.join(project_root, 'results')
    visualizer_dir = os.path.join(project_root, 'visualizer')
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(visualizer_dir, exist_ok=True)

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Train data not found at {train_path}")

    train_df = pd.read_csv(train_path)
    print(f"Loaded {len(train_df)} training shots.", flush=True)

    hybrid_pipe = HybridTrajectoryPipeline(random_state=42)
    hybrid_pipe.fit_and_evaluate(train_df)

    oof_df = pd.concat([train_df[['track_id']], hybrid_pipe.oof_predictions], axis=1)
    oof_path = os.path.join(results_dir, 'hybrid_oof.csv')
    oof_df.to_csv(oof_path, index=False)
    print(f"\nSaved Hybrid Out-Of-Fold predictions to {oof_path}", flush=True)

    if os.path.exists(test_path):
        test_df = pd.read_csv(test_path)
        print(f"\nLoaded {len(test_df)} test shots.", flush=True)
        test_sub = hybrid_pipe.predict(test_df)

        sub_path = os.path.join(results_dir, 'hybrid_predictions.csv')
        test_sub.to_csv(sub_path, index=False)
        print(f"Saved Hybrid test predictions to {sub_path} (Shape: {test_sub.shape})", flush=True)

        # Merge test inputs with model predictions for visualization!
        viz_df = pd.merge(test_df, test_sub, on='track_id')
        is_pred = True
    else:
        viz_df = train_df
        is_pred = False

    visualizer_anim_path = os.path.join(visualizer_dir, 'hybrid_flight_animation.html')
    hybrid_pipe.create_interactive_trajectory_animation(shot_choice, visualizer_anim_path, df=viz_df, is_prediction=is_pred)

    print("\nHybrid model pipeline completed successfully!", flush=True)


if __name__ == '__main__':
    run_hybrid_pipeline()
