"""
src/trajectory_visualizer.py

Plotly-based interactive 3D flight and radar comparison visualizer.
Generates `visualizer/hybrid_flight_animation.html` with:
- Pure aerodynamic trajectory (cyan) vs gate-fitted noisy trajectory (orange).
- Radar checkpoint offset error vectors (red dotted lines).
- Stellenbosch Kikuyu turf multi-hop bounce and rollout.
- Full time-scrubbing, animation controls, and multi-shot dropdown selector.
"""

import os
import json
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
import plotly.graph_objects as go
from physics import GolfBallPhysicsSimulator


class TrajectoryPlotlyVisualizer:
    """
    Generates interactive 3D trajectory visualizers comparing aerodynamic flight against radar checkpoints.
    """

    def __init__(self, simulator: GolfBallPhysicsSimulator = None):
        self.simulator = simulator or GolfBallPhysicsSimulator()

    def build_shot_figure(self, sample_row: pd.Series, shot_num: int = None, is_prediction: bool = True) -> go.Figure:
        cd_scale, cl_scale, cs_obs, est_spin = self.simulator.estimate_aero_coefficients(sample_row)
        if 'launch_spin_rate' in sample_row and pd.notnull(sample_row['launch_spin_rate']):
            est_spin = float(sample_row['launch_spin_rate'])

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

        # 1. Pure Aerodynamic Path
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

        # 2. Checkpoint-Constrained Spline Path (Visualizing Sensor Noise)
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

        # 3. Radar Gate Offset Error Vectors
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

        full_pos = np.vstack([phys_pos, phys_bpos[1:]])
        full_t = np.concatenate([phys_t, phys_t[-1] + phys_bt[1:]])
        n_flight = len(phys_pos)

        fig = go.Figure()

        # Trace 0: Ideal Aerodynamic Flight Path
        fig.add_trace(go.Scatter3d(
            x=phys_pos[:, 0], y=phys_pos[:, 1], z=phys_pos[:, 2],
            mode='lines',
            line=dict(color='#00f0ff', width=5),
            name='Ideal Aerodynamic Path (Pure Physics - True Flight)'
        ))

        # Trace 1: Ideal Bounce & Rollout
        fig.add_trace(go.Scatter3d(
            x=phys_bpos[:, 0], y=phys_bpos[:, 1], z=phys_bpos[:, 2],
            mode='lines',
            line=dict(color='rgba(0, 240, 255, 0.60)', width=3, dash='dot'),
            name='Ideal Turf Rollout (Stellenbosch Turf)'
        ))

        # Trace 2: Radar-Constrained Path
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

        # Trace 4: Measurement Error Vectors
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

    def create_interactive_trajectory_animation(self, sample_row, output_html_path: str, df: pd.DataFrame = None, is_prediction: bool = True):
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

        fig = self.build_shot_figure(row, shot_num=shot_idx, is_prediction=is_prediction)
        os.makedirs(os.path.dirname(output_html_path), exist_ok=True)

        if df is None or len(df) == 0:
            fig.write_html(output_html_path)
            print(f"Saved interactive 3D trajectory animation to {output_html_path}", flush=True)
            return

        # Pre-compute selectable shots
        n_selectable = min(10, len(df))
        selectable_indices = list(range(n_selectable))
        if shot_idx not in selectable_indices:
            selectable_indices.append(shot_idx)
            selectable_indices.sort()

        shots_cache = {}
        options_html = []

        for idx in selectable_indices:
            row_i = df.iloc[idx]
            fig_i = self.build_shot_figure(row_i, shot_num=idx, is_prediction=is_prediction)
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
