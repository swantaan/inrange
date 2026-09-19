"""
src/build_3d_simulator.py

SpaceX-Style Minimalist 3D Golf Ball Flight Simulator:
- 3D Holographic Meter Tags & Vertical Laser Beams for 15m, 30m, 45m, 60m checkpoints
- 3D Apex Elevation Tag with vertical ground drop-line (▲ APEX XX.Xm)
- Interactive timeline scrubber takes you to the EXACT simulation time and automatically PAUSES
- Realistic hand-painted stylized cartoon turf
- Tesla/SpaceX minimalist HUD
- State-machine Play button (PLAY -> PAUSE / RESUME -> REPLAY with smooth reset to tee without auto-playing)
- Keyboard shortcut: Spacebar or 'R'
"""

import os
import json
import base64
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

def get_base64_texture(file_path):
    if os.path.exists(file_path):
        with open(file_path, "rb") as image_file:
            encoded = base64.b64encode(image_file.read()).decode('utf-8')
            return f"data:image/png;base64,{encoded}"
    return ""

def process_shot_data(row, is_train=True):
    x0, y0, z0 = float(row['launch_x']), float(row['launch_y']), float(row['launch_z'])
    t_pts = [0.0, float(row['cp1_t']), float(row['cp2_t']), float(row['cp3_t']), float(row['cp4_t'])]
    x_pts = [x0, float(row['cp1_x']), float(row['cp2_x']), float(row['cp3_x']), float(row['cp4_x'])]
    y_pts = [y0, float(row['cp1_y']), float(row['cp2_y']), float(row['cp3_y']), float(row['cp4_y'])]
    z_pts = [z0, float(row['cp1_z']), float(row['cp2_z']), float(row['cp3_z']), float(row['cp4_z'])]

    t_apex = float(row['apex_t'])
    t_land = float(row['landing_t'])
    spin_rpm_0 = float(row.get('launch_spin_rate', 4800))
    
    pts_dict = {
        0.0: (x_pts[0], y_pts[0], z_pts[0]),
        float(row['cp1_t']): (x_pts[1], y_pts[1], z_pts[1]),
        float(row['cp2_t']): (x_pts[2], y_pts[2], z_pts[2]),
        float(row['cp3_t']): (x_pts[3], y_pts[3], z_pts[3]),
        float(row['cp4_t']): (x_pts[4], y_pts[4], z_pts[4]),
        t_apex: (float(row['apex_x']), float(row['apex_y']), float(row['apex_z'])),
        t_land: (float(row['landing_x']), float(row['landing_y']), float(row['landing_z']))
    }
    
    sorted_times = sorted(pts_dict.keys())
    filtered_times = [sorted_times[0]]
    for st in sorted_times[1:]:
        if st - filtered_times[-1] > 0.03:
            filtered_times.append(st)
    
    t_arr = np.array(filtered_times)
    x_arr = np.array([pts_dict[t][0] for t in t_arr])
    y_arr = np.array([pts_dict[t][1] for t in t_arr])
    z_arr = np.array([pts_dict[t][2] for t in t_arr])
    
    # Calculate shot launch azimuth angle so we can rotate the entire flight to be perfectly straight at 0° (+X axis)
    # Use initial velocity direction or early trajectory vector
    v0_x = float(row['launch_vx'])
    v0_y = float(row['launch_vy'])
    theta_0 = np.arctan2(v0_y, v0_x)
    cos_t = np.cos(theta_0)
    sin_t = np.sin(theta_0)
    
    def to_straight_frame(x_raw, y_raw):
        dx = x_raw - x0
        dy = y_raw - y0
        x_str = dx * cos_t + dy * sin_t
        y_str = -dx * sin_t + dy * cos_t
        return x_str, y_str

    cs_x = CubicSpline(t_arr, x_arr, bc_type='natural')
    cs_y = CubicSpline(t_arr, y_arr, bc_type='natural')
    cs_z = CubicSpline(t_arr, z_arr, bc_type='natural')
    
    t_eval = np.linspace(0.0, t_land, 160)
    xs_raw = cs_x(t_eval)
    ys_raw = cs_y(t_eval)
    
    # Transform spline to straight 0-degree frame starting from (0, 0, 0)
    xs_str, ys_str = to_straight_frame(xs_raw, ys_raw)
    
    zs_raw = cs_z(t_eval)
    zs_rel = np.maximum(zs_raw - z0, 0.0)
    
    full_pts = []
    tau_spin = 15.0
    
    for i in range(len(t_eval)):
        t = float(t_eval[i])
        x = float(xs_str[i])
        y = float(ys_str[i])
        z_rel = float(zs_rel[i])
        
        if i < len(t_eval) - 1:
            dt = t_eval[i+1] - t
            vx = (xs_str[i+1] - x) / dt
            vy = (ys_str[i+1] - y) / dt
            vz = (zs_rel[i+1] - z_rel) / dt
        else:
            dt = t - t_eval[i-1]
            vx = (x - xs_str[i-1]) / dt
            vy = (y - ys_str[i-1]) / dt
            vz = (z_rel - zs_rel[i-1]) / dt
            
        current_spin_rpm = spin_rpm_0 * np.exp(-t / tau_spin)
        dist_from_tee = np.sqrt(x**2 + y**2)
        
        full_pts.append({
            't': round(t, 3),
            'x': round(x, 3),
            'y': round(y, 3),
            'z': round(z_rel, 3),
            'vx': round(float(vx), 2),
            'vy': round(float(vy), 2),
            'vz': round(float(vz), 2),
            'dist': round(float(dist_from_tee), 1),
            'spin_rpm': round(float(current_spin_rpm), 0),
            'is_flying': True
        })
            
    v_land_x = full_pts[-1]['vx']
    v_land_y = full_pts[-1]['vy']
    v_land_z = full_pts[-1]['vz']
    
    bounce_pts = []
    curr_p = np.array([xs_str[-1], ys_str[-1], 0.0], dtype=float)
    
    v_xy = np.sqrt(v_land_x**2 + v_land_y**2)
    descent_angle = np.degrees(np.arctan2(abs(v_land_z), v_xy + 1e-6))
    
    rest_z = 0.32 if descent_angle > 45 else 0.38
    tangent_restitution = 0.45
    
    curr_v = np.array([v_land_x * tangent_restitution, v_land_y * tangent_restitution, abs(v_land_z) * rest_z], dtype=float)
    b_dt = 0.015
    sim_t = t_land
    ball_radius = 0.02135
    current_spin = full_pts[-1]['spin_rpm']
    
    bounce_count = 0
    max_bounces = 2
    
    while bounce_count < max_bounces and (curr_v[2] > 0.15 or curr_p[2] > 0.005):
        curr_v[2] -= 9.81 * b_dt
        curr_p += curr_v * b_dt
        sim_t += b_dt
        
        # Continuous aerodynamic & rotational spin decay while in bounce
        current_spin *= np.exp(-b_dt / 1.8)
        
        if curr_p[2] <= 0.0 and curr_v[2] < 0:
            curr_p[2] = 0.0
            curr_v[2] = abs(curr_v[2]) * 0.28
            curr_v[0] *= 0.65
            curr_v[1] *= 0.65
            bounce_count += 1
            current_spin *= 0.65
            
        dist_from_tee = np.sqrt(curr_p[0]**2 + curr_p[1]**2)
        bounce_pts.append({
            't': round(float(sim_t), 3),
            'x': round(float(curr_p[0]), 3),
            'y': round(float(curr_p[1]), 3),
            'z': round(float(max(curr_p[2], 0.0)), 3),
            'vx': round(float(curr_v[0]), 2),
            'vy': round(float(curr_v[1]), 2),
            'vz': round(float(curr_v[2]), 2),
            'dist': round(float(dist_from_tee), 1),
            'spin_rpm': round(float(current_spin), 0),
            'is_flying': False,
            'is_rolling': False
        })
        
    curr_p[2] = 0.0
    curr_v[2] = 0.0
    
    mu_roll = 0.35
    v_roll = np.sqrt(curr_v[0]**2 + curr_v[1]**2)
    u_dir = np.array([curr_v[0], curr_v[1], 0.0]) / (v_roll + 1e-8)
    
    # Smooth continuous transition from bounce spin to ground roll
    start_roll_spin = current_spin
    roll_step = 0
    
    while v_roll > 0.01 and sim_t < t_land + 5.0:
        v_roll -= mu_roll * 9.81 * b_dt
        if v_roll < 0:
            v_roll = 0.0
        curr_p += u_dir * v_roll * b_dt
        curr_p[2] = 0.0
        sim_t += b_dt
        roll_step += 1
        
        # Physical pure-rolling spin (v / R)
        natural_roll_spin = (v_roll / ball_radius) * (60.0 / (2.0 * np.pi))
        # Smoothly blend from impact spin into pure rolling spin
        blend = min(1.0, roll_step / 12.0)
        roll_spin = (1.0 - blend) * start_roll_spin + blend * natural_roll_spin
        
        dist_from_tee = np.sqrt(curr_p[0]**2 + curr_p[1]**2)
        
        bounce_pts.append({
            't': round(float(sim_t), 3),
            'x': round(float(curr_p[0]), 3),
            'y': round(float(curr_p[1]), 3),
            'z': 0.0,
            'vx': round(float(u_dir[0] * v_roll), 2),
            'vy': round(float(u_dir[1] * v_roll), 2),
            'vz': 0.0,
            'dist': round(float(dist_from_tee), 1),
            'spin_rpm': round(float(roll_spin), 0),
            'is_flying': False,
            'is_rolling': True
        })
        
    final_p = bounce_pts[-1]
    bounce_pts.append({
        't': round(final_p['t'] + 0.05, 3),
        'x': final_p['x'],
        'y': final_p['y'],
        'z': 0.0,
        'vx': 0.0,
        'vy': 0.0,
        'vz': 0.0,
        'dist': final_p['dist'],
        'spin_rpm': 0.0,
        'is_flying': False,
        'is_rolling': True
    })
            
    total_carry = np.sqrt(xs_str[-1]**2 + ys_str[-1]**2)
    total_distance = np.sqrt(bounce_pts[-1]['x']**2 + bounce_pts[-1]['y']**2)
    
    v0 = np.sqrt(row['launch_vx']**2 + row['launch_vy']**2 + row['launch_vz']**2)
    launch_angle = np.degrees(np.arctan2(row['launch_vz'], np.sqrt(row['launch_vx']**2 + row['launch_vy']**2)))
    
    if v0 > 65 and launch_angle < 16:
        club = "1W Driver"
    elif v0 > 55:
        club = "3-Wood"
    elif v0 > 45:
        club = "5-Iron"
    elif v0 > 35:
        club = "7-Iron"
    else:
        club = "Pitching Wedge"
        
    # Transform checkpoints and apex
    cp1_sx, cp1_sy = to_straight_frame(float(row['cp1_x']), float(row['cp1_y']))
    cp2_sx, cp2_sy = to_straight_frame(float(row['cp2_x']), float(row['cp2_y']))
    cp3_sx, cp3_sy = to_straight_frame(float(row['cp3_x']), float(row['cp3_y']))
    cp4_sx, cp4_sy = to_straight_frame(float(row['cp4_x']), float(row['cp4_y']))
    apex_sx, apex_sy = to_straight_frame(float(row['apex_x']), float(row['apex_y']))
    land_sx, land_sy = to_straight_frame(float(row['landing_x']), float(row['landing_y']))
        
    return {
        'track_id': str(row['track_id']),
        'club': club,
        'launch_pos': {'x': 0.0, 'y': 0.0, 'z': 0.0},
        'launch_speed_ms': round(float(v0), 1),
        'launch_speed_kmh': round(float(v0 * 3.6), 1),
        'launch_angle_deg': round(float(launch_angle), 1),
        'launch_spin_rpm': round(spin_rpm_0, 0),
        'apex_t': round(t_apex, 2),
        'apex_x': round(float(apex_sx), 2),
        'apex_y': round(float(apex_sy), 2),
        'apex_z': round(float(row['apex_z'] - z0), 2),
        'landing_t': round(t_land, 2),
        'landing_x': round(float(land_sx), 2),
        'landing_y': round(float(land_sy), 2),
        'landing_z': 0.0,
        'carry_distance_m': round(float(total_carry), 1),
        'total_distance_m': round(float(total_distance), 1),
        'checkpoints': [
            {'name': '15m', 'tag': '15m', 't': round(float(row['cp1_t']), 3), 'dist': 15.0, 'x': round(cp1_sx, 2), 'y': round(cp1_sy, 2), 'z': round(float(row['cp1_z'] - z0), 2)},
            {'name': '30m', 'tag': '30m', 't': round(float(row['cp2_t']), 3), 'dist': 30.0, 'x': round(cp2_sx, 2), 'y': round(cp2_sy, 2), 'z': round(float(row['cp2_z'] - z0), 2)},
            {'name': '45m', 'tag': '45m', 't': round(float(row['cp3_t']), 3), 'dist': 45.0, 'x': round(cp3_sx, 2), 'y': round(cp3_sy, 2), 'z': round(float(row['cp3_z'] - z0), 2)},
            {'name': '60m', 'tag': '60m (NET)', 't': round(float(row['cp4_t']), 3), 'dist': 60.0, 'x': round(cp4_sx, 2), 'y': round(cp4_sy, 2), 'z': round(float(row['cp4_z'] - z0), 2)},
        ],
        'flight_points': full_pts,
        'bounce_points': bounce_pts
    }

def build_3d_simulator():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    train_path = os.path.join(project_root, 'data', 'train.csv')
    vis_dir = os.path.join(project_root, 'visualizer')
    tex_dir = os.path.join(vis_dir, 'textures')
    
    tex_map = {
        'turf': get_base64_texture(os.path.join(tex_dir, 'stylized_turf.png')),
        'ball': get_base64_texture(os.path.join(tex_dir, 'ball_radar_pattern.png')),
    }
    
    train_df = pd.read_csv(train_path)
    shots_data = []
    
    indices = np.linspace(0, len(train_df) - 1, 40, dtype=int)
    for idx in indices:
        row = train_df.iloc[idx]
        shot_info = process_shot_data(row, is_train=True)
        if shot_info:
            shots_data.append(shot_info)
            
    print(f"Extracted {len(shots_data)} precision shots with 3D checkpoint & Apex tags.")
    
    html_content = generate_elon_minimalist_html(shots_data, tex_map)
    
    output_html = os.path.join(vis_dir, 'golf_simulator_3d.html')
    with open(output_html, 'w', encoding='utf-8') as f:
        f.write(html_content)
        
    index_html = os.path.join(vis_dir, 'index.html')
    with open(index_html, 'w', encoding='utf-8') as f:
        f.write(html_content)
        
    print(f"Successfully generated Visualizer at:\n  - {output_html}\n  - {index_html}")

def generate_elon_minimalist_html(shots_data, textures):
    shots_json = json.dumps(shots_data)
    tex_json = json.dumps(textures)
    
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Inrange Flight Telemetry - SpaceX Style 3D Simulator</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Chivo+Mono:ital,wght@0,600;0,700;0,800;1,700&family=Outfit:wght@400;500;600;700;800;900&display=swap" rel="stylesheet">
    
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
    
    <style>
        :root {{
            --bg-titanium: rgba(10, 14, 18, 0.94);
            --border-crisp: 1px solid rgba(255, 255, 255, 0.15);
            --text-main: #f8fafc;
            --text-sub: #71717a;
            --font-main: 'Outfit', -apple-system, sans-serif;
            --font-mono: 'Chivo Mono', monospace;
        }}

        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
            user-select: none;
            -webkit-user-select: none;
        }}

        body {{
            font-family: var(--font-main);
            background: #000000;
            overflow: hidden;
            width: 100vw;
            height: 100vh;
            color: var(--text-main);
        }}

        #canvas-container {{
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            z-index: 1;
        }}

        .telemetry-overlay {{
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            z-index: 10;
            pointer-events: none;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            padding: 24px;
        }}

        .interactive {{
            pointer-events: auto;
        }}

        /* Top Center Telemetry Dashboard Bar */
        .top-dashboard-container {{
            display: flex;
            justify-content: center;
            width: 100%;
        }}

        .spacex-dashboard {{
            background: rgba(10, 14, 18, 0.92);
            border: 1px solid rgba(255, 255, 255, 0.18);
            border-radius: 0px;
            display: flex;
            align-items: center;
            box-shadow: 0 16px 40px rgba(0, 0, 0, 0.7);
            backdrop-filter: blur(16px);
        }}

        .dash-cell {{
            padding: 10px 22px;
            border-right: 1px solid rgba(255, 255, 255, 0.08);
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            min-width: 95px;
        }}

        .dash-cell:last-child {{
            border-right: none;
        }}

        .dash-cell.hero-cell {{
            padding: 10px 32px;
            min-width: 170px;
            background: rgba(255, 255, 255, 0.03);
            border-right: 1px solid rgba(255, 255, 255, 0.12);
        }}

        .hero-value {{
            font-family: var(--font-mono);
            font-size: 52px; /* 3x as big as normal stat numbers (17px * 3 ≈ 51px) */
            font-weight: 800;
            color: #ffffff;
            line-height: 1.0;
            letter-spacing: -1.5px;
        }}

        .hero-sub {{
            display: flex;
            align-items: center;
            gap: 5px;
            margin-top: 3px;
        }}

        .hero-unit {{
            font-family: var(--font-mono);
            font-size: 11px;
            font-weight: 700;
            color: #a1a1aa;
        }}

        .cell-label {{
            font-size: 8px;
            font-weight: 700;
            color: #71717a;
            text-transform: uppercase;
            letter-spacing: 1.5px;
            margin-bottom: 2px;
        }}

        .cell-value {{
            font-family: var(--font-mono);
            font-size: 17px;
            font-weight: 700;
            color: #ffffff;
            letter-spacing: -0.5px;
        }}

        .cell-unit {{
            font-size: 10px;
            color: #a1a1aa;
            font-weight: 600;
            margin-left: 3px;
        }}

        /* Bottom Controls Bar */
        .bottom-bar {{
            display: flex;
            justify-content: space-between;
            align-items: flex-end;
            width: 100%;
            gap: 16px;
        }}

        .shot-select-panel {{
            background: var(--bg-titanium);
            border: var(--border-crisp);
            border-radius: 0px;
            padding: 10px 14px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
            backdrop-filter: blur(20px);
            width: 320px;
        }}

        .panel-header {{
            font-size: 8px;
            font-weight: 700;
            color: #a1a1aa;
            text-transform: uppercase;
            letter-spacing: 2px;
            margin-bottom: 6px;
        }}

        .spacex-dropdown {{
            width: 100%;
            background: #080c10;
            border: 1px solid rgba(255, 255, 255, 0.22);
            border-radius: 0px;
            padding: 8px 10px;
            font-family: var(--font-main);
            font-size: 13px;
            font-weight: 700;
            color: #ffffff;
            outline: none;
            cursor: pointer;
            transition: border-color 0.2s;
        }}

        .spacex-dropdown:hover {{
            border-color: #ffffff;
        }}

        .center-control-deck {{
            background: var(--bg-titanium);
            border: var(--border-crisp);
            border-radius: 0px;
            padding: 8px 18px;
            display: flex;
            align-items: center;
            gap: 14px;
            box-shadow: 0 12px 35px rgba(0, 0, 0, 0.6);
            backdrop-filter: blur(20px);
        }}

        .tesla-btn {{
            background: #0c1015;
            border: 1px solid rgba(255, 255, 255, 0.22);
            border-radius: 0px;
            padding: 9px 18px;
            color: #ffffff;
            font-family: var(--font-main);
            font-size: 12px;
            font-weight: 800;
            letter-spacing: 1.5px;
            text-transform: uppercase;
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            outline: none;
            transition: all 0.15s cubic-bezier(0.16, 1, 0.3, 1);
            min-width: 90px;
        }}

        .tesla-btn:hover {{
            background: #181f28;
            border-color: #ffffff;
        }}

        .tesla-btn.primary-action {{
            background: #ffffff;
            color: #000000;
            border-color: #ffffff;
            font-weight: 900;
        }}

        .tesla-btn.primary-action:hover {{
            background: #e2e8f0;
            box-shadow: 0 0 16px rgba(255, 255, 255, 0.4);
        }}

        .tesla-btn.slowmo.active {{
            background: #ffffff;
            color: #000000;
            border-color: #ffffff;
        }}

        .scrubber-zone {{
            display: flex;
            flex-direction: column;
            width: 250px;
            gap: 4px;
        }}

        .time-telemetry {{
            display: flex;
            justify-content: space-between;
            font-family: var(--font-mono);
            font-size: 11px;
            font-weight: 700;
            color: var(--text-sub);
        }}

        .tesla-slider {{
            -webkit-appearance: none;
            width: 100%;
            height: 4px;
            border-radius: 0px;
            background: #27272a;
            outline: none;
            cursor: pointer;
        }}

        .tesla-slider::-webkit-slider-thumb {{
            -webkit-appearance: none;
            width: 12px;
            height: 12px;
            border-radius: 0px;
            background: #ffffff;
            border: 1px solid #71717a;
            cursor: pointer;
            box-shadow: 0 0 6px rgba(255, 255, 255, 0.5);
        }}
    </style>
</head>
<body>
    <div id="canvas-container"></div>

    <div class="telemetry-overlay">
        <!-- Top Minimalist Telemetry Bar -->
        <div class="top-dashboard-container">
            <div class="spacex-dashboard interactive">
                <div class="dash-cell hero-cell">
                    <span class="hero-value" id="val-dist">0.0</span>
                    <div class="hero-sub">
                        <span class="cell-label">DISTANCE</span>
                        <span class="hero-unit">METERS</span>
                    </div>
                </div>
                <div class="dash-cell">
                    <span class="cell-label">SPEED</span>
                    <span class="cell-value" id="val-speed">0.0<span class="cell-unit">km/h</span></span>
                </div>
                <div class="dash-cell">
                    <span class="cell-label">SPIN</span>
                    <span class="cell-value" id="val-spin">0<span class="cell-unit">rpm</span></span>
                </div>
            </div>
        </div>

        <!-- Bottom Controls -->
        <div class="bottom-bar">
            <!-- Shot Selector -->
            <div class="shot-select-panel interactive">
                <div class="panel-header">SELECT TRAJECTORY</div>
                <select id="shot-dropdown" class="spacex-dropdown"></select>
            </div>

            <!-- Play / Slow-Mo / Camera / Timeline Controls -->
            <div class="center-control-deck interactive">
                <button class="tesla-btn primary-action" id="btn-action">PLAY</button>
                <button class="tesla-btn slowmo" id="btn-slowmo">SLOW-MO</button>
                <button class="tesla-btn" id="btn-cam-mode">CAM: CHASE</button>

                <div class="scrubber-zone">
                    <div class="time-telemetry">
                        <span id="time-curr">0.00s</span>
                        <span id="time-total">4.85s</span>
                    </div>
                    <input type="range" id="time-slider" class="tesla-slider" min="0" max="100" value="0" step="0.1">
                </div>
            </div>
        </div>
    </div>

    <script>
        const SHOTS = {shots_json};
        const TEXTURES_DATA = {tex_json};

        let currentShotIndex = 0;
        let currentShot = SHOTS[0];

        // --- 3D Scene ---
        let scene, camera, renderer, controls;
        let ballMesh;
        let fullTrajectoryMeshGroup, shadowTrailLine, shadowGeometry, shadowPositions;
        let markerObjects = []; // 3D Checkpoint & Apex objects
        
        let playState = 'READY'; 
        let isSlowmo = false;
        let cameraMode = 'CHASE'; // 'CHASE' (top-angled following ball) or 'ORIGIN' (stays at tee, tracks ball downrange)
        let fullTrajectory = [];
        let simProgress = 0.0;

        let ballQuaternion = new THREE.Quaternion();
        const BALL_RADIUS = 0.12; // High-visibility scaled golf ball

        let isResettingCamera = false;
        let userInteracting = false;
        let snapBackTimeout = null;
        let userZoomFactor = 1.0; // Dynamic zoom multiplier (0.2x close-up to 4.0x wide-angle)

        // Pre-builds the entire 3D predetermined flight path so the trajectory is always visible from tee-off
        function buildPredeterminedTrajectory() {{
            while (fullTrajectoryMeshGroup.children.length > 0) {{
                const obj = fullTrajectoryMeshGroup.children[0];
                if (obj.geometry) obj.geometry.dispose();
                fullTrajectoryMeshGroup.remove(obj);
            }}

            if (!fullTrajectory || fullTrajectory.length < 2) return;

            const points = [];
            for (let i = 0; i < fullTrajectory.length; i++) {{
                points.push(new THREE.Vector3(
                    fullTrajectory[i].x,
                    fullTrajectory[i].z + BALL_RADIUS,
                    fullTrajectory[i].y
                ));
            }}

            // Smooth continuous 3D Spline Tube for the full predetermined trajectory
            const curve = new THREE.CatmullRomCurve3(points);
            const tubeGeo = new THREE.TubeGeometry(curve, Math.max(points.length * 2, 60), 0.09, 8, false);
            
            // Outer Vibrant Electric Blue Ribbon (semi-transparent)
            const tubeMat = new THREE.MeshBasicMaterial({{
                color: 0x0088ff,
                transparent: true,
                opacity: 0.55,
                depthWrite: false
            }});
            const tubeMesh = new THREE.Mesh(tubeGeo, tubeMat);
            fullTrajectoryMeshGroup.add(tubeMesh);

            // Inner Neon Cyan/White Core (subtle semi-transparent glow)
            const coreGeo = new THREE.TubeGeometry(curve, Math.max(points.length * 2, 60), 0.035, 6, false);
            const coreMat = new THREE.MeshBasicMaterial({{
                color: 0xe0f2fe,
                transparent: true,
                opacity: 0.70,
                depthWrite: false
            }});
            const coreMesh = new THREE.Mesh(coreGeo, coreMat);
            fullTrajectoryMeshGroup.add(coreMesh);

            // Full ground shadow projection along the entire path
            for (let i = 0; i < fullTrajectory.length; i++) {{
                shadowPositions[i * 3] = fullTrajectory[i].x;
                shadowPositions[i * 3 + 1] = 0.012;
                shadowPositions[i * 3 + 2] = fullTrajectory[i].y;
            }}
            shadowGeometry.setDrawRange(0, fullTrajectory.length);
            shadowGeometry.attributes.position.needsUpdate = true;
        }}

        function loadTexture(b64, repeatX=1, repeatY=1) {{
            const loader = new THREE.TextureLoader();
            const tex = loader.load(b64);
            tex.wrapS = THREE.RepeatWrapping;
            tex.wrapT = THREE.RepeatWrapping;
            tex.repeat.set(repeatX, repeatY);
            return tex;
        }}

        // Helper to generate crisp high-contrast 3D Badge Sprite with solid pill backdrop
        function create3DTextBadge(text, textColor='#ffffff', bgColor='rgba(15, 23, 42, 0.92)', scaleW=2.2, scaleH=1.0) {{
            const canvas = document.createElement('canvas');
            canvas.width = 512;
            canvas.height = 256;
            const ctx = canvas.getContext('2d');

            ctx.clearRect(0, 0, canvas.width, canvas.height);

            // Draw rounded pill badge backdrop for crystal-clear readability
            const padX = 24, padY = 24;
            const w = canvas.width - padX * 2;
            const h = canvas.height - padY * 2;
            const radius = 32;

            ctx.beginPath();
            ctx.moveTo(padX + radius, padY);
            ctx.lineTo(padX + w - radius, padY);
            ctx.quadraticCurveTo(padX + w, padY, padX + w, padY + radius);
            ctx.lineTo(padX + w, padY + h - radius);
            ctx.quadraticCurveTo(padX + w, padY + h, padX + w - radius, padY + h);
            ctx.lineTo(padX + radius, padY + h);
            ctx.quadraticCurveTo(padX, padY + h, padX, padY + h - radius);
            ctx.lineTo(padX, padY + radius);
            ctx.quadraticCurveTo(padX, padY, padX + radius, padY);
            ctx.closePath();

            ctx.fillStyle = bgColor;
            ctx.fill();
            ctx.lineWidth = 8;
            ctx.strokeStyle = textColor;
            ctx.stroke();

            // Text
            ctx.shadowColor = 'rgba(0, 0, 0, 0.9)';
            ctx.shadowBlur = 10;
            ctx.font = 'bold 84px "Outfit", "Chivo Mono", sans-serif';
            ctx.fillStyle = textColor;
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillText(text, canvas.width / 2, canvas.height / 2);

            const texture = new THREE.CanvasTexture(canvas);
            texture.minFilter = THREE.LinearFilter;
            const spriteMat = new THREE.SpriteMaterial({{ map: texture, transparent: true, depthTest: false }});
            const sprite = new THREE.Sprite(spriteMat);
            sprite.scale.set(scaleW, scaleH, 1.0);
            return sprite;
        }}

        function initScene() {{
            const container = document.getElementById('canvas-container');
            scene = new THREE.Scene();
            
            // Vibrant Luminous Azure Sky & Atmosphere
            scene.background = new THREE.Color(0x38bdf8); // Radiant clear vibrant sky blue
            scene.fog = new THREE.FogExp2(0x7dd3fc, 0.00028);

            camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.1, 4000);
            camera.position.set(-22, 18, 0);

            renderer = new THREE.WebGLRenderer({{ antialias: true, alpha: false }});
            renderer.setSize(window.innerWidth, window.innerHeight);
            renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
            renderer.shadowMap.enabled = true;
            renderer.shadowMap.type = THREE.PCFSoftShadowMap;
            renderer.toneMapping = THREE.ACESFilmicToneMapping;
            renderer.toneMappingExposure = 1.25;
            container.appendChild(renderer.domElement);

            controls = new THREE.OrbitControls(camera, renderer.domElement);
            controls.enableDamping = true;
            controls.dampingFactor = 0.08;
            controls.maxPolarAngle = Math.PI / 2 - 0.005;

            renderer.domElement.addEventListener('pointerdown', () => {{
                userInteracting = true;
                if (snapBackTimeout) clearTimeout(snapBackTimeout);
            }});

            window.addEventListener('pointerup', () => {{
                if (userInteracting) {{
                    snapBackTimeout = setTimeout(() => {{
                        userInteracting = false;
                    }}, 250);
                }}
            }});

            // Enable interactive zoom scaling via mouse wheel & trackpad
            renderer.domElement.addEventListener('wheel', (e) => {{
                e.preventDefault();
                const zoomDelta = e.deltaY * 0.0012;
                userZoomFactor = Math.max(0.15, Math.min(4.5, userZoomFactor + zoomDelta));
            }}, {{ passive: false }});

            const sunLight = new THREE.DirectionalLight(0xffffff, 1.7);
            sunLight.position.set(150, 320, 140);
            sunLight.castShadow = true;
            sunLight.shadow.mapSize.width = 2048;
            sunLight.shadow.mapSize.height = 2048;
            sunLight.shadow.camera.near = 10;
            sunLight.shadow.camera.far = 750;
            sunLight.shadow.camera.left = -220;
            sunLight.shadow.camera.right = 220;
            sunLight.shadow.camera.top = 220;
            sunLight.shadow.camera.bottom = -220;
            scene.add(sunLight);

            const hemiLight = new THREE.HemisphereLight(0xffffff, 0x15803d, 0.95);
            scene.add(hemiLight);

            buildStylizedTurfFloor();

            const ballGeo = new THREE.SphereGeometry(BALL_RADIUS, 32, 32);
            const ballTex = loadTexture(TEXTURES_DATA.ball, 1, 1);
            const ballMat = new THREE.MeshStandardMaterial({{
                map: ballTex,
                roughness: 0.20,
                metalness: 0.02,
                emissive: 0xffffff,
                emissiveIntensity: 0.25
            }});
            ballMesh = new THREE.Mesh(ballGeo, ballMat);
            ballMesh.castShadow = true;
            scene.add(ballMesh);

            // High-Visibility 3D Glowing Blue Flight Ribbon & Ground Shadow
            fullTrajectoryMeshGroup = new THREE.Group();
            scene.add(fullTrajectoryMeshGroup);

            // Ground Shadow Projection Line
            const shadowPointsMax = 400;
            shadowPositions = new Float32Array(shadowPointsMax * 3);
            shadowGeometry = new THREE.BufferGeometry();
            shadowGeometry.setAttribute('position', new THREE.BufferAttribute(shadowPositions, 3));
            const shadowMat = new THREE.LineBasicMaterial({{
                color: 0x064e3b,
                transparent: true,
                opacity: 0.40
            }});
            shadowTrailLine = new THREE.Line(shadowGeometry, shadowMat);
            scene.add(shadowTrailLine);

            window.addEventListener('resize', onWindowResize);
        }}

        function buildStylizedTurfFloor() {{
            const turfGeo = new THREE.PlaneGeometry(1400, 1400, 1, 1);
            const turfMat = new THREE.MeshStandardMaterial({{
                color: 0x15803d, // Vibrant rich lush emerald golf green
                roughness: 0.85,
                metalness: 0.03
            }});
            const turfMesh = new THREE.Mesh(turfGeo, turfMat);
            turfMesh.rotation.x = -Math.PI / 2;
            turfMesh.position.y = 0.0;
            turfMesh.receiveShadow = true;
            scene.add(turfMesh);

            // 1. Sleek Crimson Centerline Running Straight Down the Driving Range (0° along +X)
            const redCenterGeo = new THREE.PlaneGeometry(380, 0.09);
            const redCenterMat = new THREE.MeshBasicMaterial({{
                color: 0xff1744, // Vivid laser crimson
                side: THREE.DoubleSide
            }});
            const redCenterMesh = new THREE.Mesh(redCenterGeo, redCenterMat);
            redCenterMesh.rotation.x = -Math.PI / 2;
            redCenterMesh.position.set(165, 0.015, 0);
            scene.add(redCenterMesh);

            // Translucent Guidance Extension Dash Line
            const centerDashedGeo = new THREE.BufferGeometry().setFromPoints([
                new THREE.Vector3(-30, 0.02, 0),
                new THREE.Vector3(380, 0.02, 0)
            ]);
            const centerDashedMat = new THREE.LineDashedMaterial({{
                color: 0xffffff,
                dashSize: 2.0,
                gapSize: 1.5,
                transparent: true,
                opacity: 0.65
            }});
            const centerDashedLine = new THREE.Line(centerDashedGeo, centerDashedMat);
            centerDashedLine.computeLineDistances();
            scene.add(centerDashedLine);

            // 2. High-Precision Range Distance Markers on Turf (50m, 100m, 150m, 200m, 250m, 300m)
            const arcDistances = [50, 100, 150, 200, 250, 300];
            arcDistances.forEach(dist => {{
                // Transverse Ground Strip across the centerline
                const stripGeo = new THREE.PlaneGeometry(0.20, 11);
                const stripMat = new THREE.MeshBasicMaterial({{ color: 0xffffff, transparent: true, opacity: 0.75 }});
                const stripMesh = new THREE.Mesh(stripGeo, stripMat);
                stripMesh.rotation.x = -Math.PI / 2;
                stripMesh.position.set(dist, 0.016, 0);
                scene.add(stripMesh);

                // Distance ring arc
                const arcPoints = [];
                for (let a = -Math.PI / 5; a <= Math.PI / 5; a += 0.03) {{
                    arcPoints.push(new THREE.Vector3(dist * Math.cos(a), 0.012, dist * Math.sin(a)));
                }}
                const arcGeo = new THREE.BufferGeometry().setFromPoints(arcPoints);
                const arcMat = new THREE.LineBasicMaterial({{
                    color: 0xffffff,
                    transparent: true,
                    opacity: 0.35
                }});
                const arcLine = new THREE.Line(arcGeo, arcMat);
                scene.add(arcLine);

                // Yardage badge on the floor
                const badge = create3DTextBadge(`${{dist}}m`, '#ffffff', 'rgba(15, 23, 42, 0.90)', 1.9, 0.95);
                badge.position.set(dist, 0.22, -4.2);
                scene.add(badge);
            }});
        }}

        function clear3DMarkers() {{
            markerObjects.forEach(obj => scene.remove(obj));
            markerObjects = [];
        }}

        // Spawns clean 15m, 30m, 45m, 60m checkpoints and Apex 3D tags
        function spawn3DTelemetryMarkers() {{
            clear3DMarkers();

            // 1. Checkpoint Markers (15m, 30m, 45m, 60m)
            currentShot.checkpoints.forEach((cp, idx) => {{
                const isNet = (cp.dist === 60.0);
                const beaconHex = isNet ? 0xff1744 : 0x0099ff; // Vibrant electric cobalt blue for beacons, vivid red for net
                const badgeTextColor = isNet ? '#ffffff' : '#00e5ff';
                const badgeBg = isNet ? 'rgba(220, 38, 38, 0.95)' : 'rgba(10, 15, 26, 0.92)';

                // Checkpoint Dot Sphere along the flight path
                const sphereGeo = new THREE.SphereGeometry(isNet ? 0.22 : 0.16, 20, 20);
                const sphereMat = new THREE.MeshBasicMaterial({{
                    color: beaconHex
                }});
                const sphere = new THREE.Mesh(sphereGeo, sphereMat);
                sphere.position.set(cp.x, cp.z + BALL_RADIUS, cp.y);
                scene.add(sphere);
                markerObjects.push(sphere);

                // 3D Floating Meter Tag Sprite Displayed ON THE SIDE of the dot indication
                const sprite = create3DTextBadge(cp.tag, badgeTextColor, badgeBg, isNet ? 2.5 : 2.0, 0.95);
                sprite.position.set(cp.x, cp.z + BALL_RADIUS + 0.25, cp.y + 2.0);
                scene.add(sprite);
                markerObjects.push(sprite);
            }});

            // 2. Apex 3D Elevation Tag Marker (Vibrant Electric Gold #f59e0b)
            const apexX = currentShot.apex_x;
            const apexY = currentShot.apex_z + BALL_RADIUS; // Height
            const apexZ = currentShot.apex_y;

            // Apex Golden Marker Diamond
            const apexGeo = new THREE.OctahedronGeometry(0.26);
            const apexMat = new THREE.MeshBasicMaterial({{ color: 0xffb800 }});
            const apexMesh = new THREE.Mesh(apexGeo, apexMat);
            apexMesh.position.set(apexX, apexY, apexZ);
            scene.add(apexMesh);
            markerObjects.push(apexMesh);

            // Apex 3D Tag Sprite Displayed on the side of the Apex marker dot
            const apexTagText = `▲ ${{currentShot.apex_z}}m`;
            const apexSprite = create3DTextBadge(apexTagText, '#ffc107', 'rgba(10, 15, 26, 0.92)', 2.2, 0.95);
            apexSprite.position.set(apexX, apexY + 0.35, apexZ + 2.2);
            scene.add(apexSprite);
            markerObjects.push(apexSprite);
        }}

        function loadShot(index) {{
            currentShotIndex = index;
            currentShot = SHOTS[index];
            
            playState = 'READY';
            simProgress = 0.0;
            isResettingCamera = false;
            ballQuaternion.identity();

            fullTrajectory = [...currentShot.flight_points, ...currentShot.bounce_points];

            const startP = currentShot.launch_pos;
            ballMesh.position.set(startP.x, BALL_RADIUS, startP.y);

            document.getElementById('val-dist').innerText = `0.0`;
            document.getElementById('val-speed').innerHTML = `0.0<span class="cell-unit">km/h</span>`;
            document.getElementById('val-spin').innerHTML = `0<span class="cell-unit">rpm</span>`;
            
            const totalDuration = fullTrajectory[fullTrajectory.length - 1].t;
            document.getElementById('time-total').innerText = totalDuration.toFixed(2) + 's';
            document.getElementById('time-curr').innerText = '0.00s';
            document.getElementById('time-slider').value = 0;
            document.getElementById('btn-action').innerText = 'PLAY';

            // Build full predetermined 3D trajectory path tube immediately upon load
            buildPredeterminedTrajectory();

            // Spawn 3D Checkpoints (15m, 30m, 45m, 60m) & Apex Tag
            spawn3DTelemetryMarkers();

            if (cameraMode === 'CHASE') {{
                camera.position.set(-18, 14, 0);
                controls.target.set(startP.x, BALL_RADIUS, startP.y);
            }} else {{
                camera.position.set(-14, 4.5, 0);
                controls.target.set(startP.x + 20, 3, startP.y);
            }}
        }}

        function handleActionButton() {{
            const btn = document.getElementById('btn-action');

            if (playState === 'READY') {{
                playState = 'RUNNING';
                btn.innerText = 'PAUSE';
            }} 
            else if (playState === 'RUNNING') {{
                playState = 'PAUSED';
                btn.innerText = 'RESUME';
            }} 
            else if (playState === 'PAUSED') {{
                playState = 'RUNNING';
                btn.innerText = 'PAUSE';
            }} 
            else if (playState === 'FINISHED') {{
                smoothResetToTee();
            }}
        }}

        function smoothResetToTee() {{
            playState = 'READY';
            simProgress = 0.0;
            isResettingCamera = true;
            ballQuaternion.identity();

            const startP = currentShot.launch_pos;
            ballMesh.position.set(startP.x, BALL_RADIUS, startP.y);

            document.getElementById('val-dist').innerText = `0.0`;
            document.getElementById('val-speed').innerHTML = `0.0<span class="cell-unit">km/h</span>`;
            document.getElementById('val-spin').innerHTML = `0<span class="cell-unit">rpm</span>`;
            document.getElementById('time-curr').innerText = '0.00s';
            document.getElementById('time-slider').value = 0;
            document.getElementById('btn-action').innerText = 'PLAY';
        }}

        function seekAndPause(progress) {{
            if (!fullTrajectory || fullTrajectory.length === 0) return;

            simProgress = Math.max(0.0, Math.min(1.0, progress));

            const actionBtn = document.getElementById('btn-action');

            if (simProgress >= 1.0) {{
                playState = 'FINISHED';
                actionBtn.innerText = 'REPLAY';
            }} else if (simProgress <= 0.0) {{
                playState = 'READY';
                actionBtn.innerText = 'PLAY';
            }} else {{
                playState = 'PAUSED';
                actionBtn.innerText = 'RESUME';
            }}

            renderExactFrame(simProgress);
        }}

        function renderExactFrame(progress) {{
            const maxIdx = fullTrajectory.length - 1;
            const exactIdx = progress * maxIdx;
            const currIdx = Math.min(Math.floor(exactIdx), maxIdx);
            const nextIdx = Math.min(currIdx + 1, maxIdx);
            const alpha = exactIdx - currIdx;

            const p1 = fullTrajectory[currIdx];
            const p2 = fullTrajectory[nextIdx];

            const ballX = p1.x + (p2.x - p1.x) * alpha;
            const ballY = (p1.z + (p2.z - p1.z) * alpha) + BALL_RADIUS;
            const ballZ = p1.y + (p2.y - p1.y) * alpha;

            ballMesh.position.set(ballX, ballY, ballZ);

            const curVx = p1.vx + (p2.vx - p1.vx) * alpha;
            const curVz = p1.vy + (p2.vy - p1.vy) * alpha;
            const horizSpeed = Math.sqrt(curVx * curVx + curVz * curVz) + 1e-6;

            const curDist = p1.dist + (p2.dist - p1.dist) * alpha;
            document.getElementById('val-dist').innerText = `${{curDist.toFixed(1)}}`;

            const isEnd = (progress >= 1.0 || currIdx >= maxIdx - 1);
            let liveSpinRPM = isEnd ? 0.0 : (p1.spin_rpm + (p2.spin_rpm - p1.spin_rpm) * alpha);
            let liveSpeedKmh = isEnd ? 0.0 : Math.sqrt(curVx**2 + curVz**2 + (p1.vz + (p2.vz - p1.vz)*alpha)**2) * 3.6;

            document.getElementById('val-spin').innerHTML = `${{Math.round(liveSpinRPM)}}<span class="cell-unit">rpm</span>`;
            document.getElementById('val-speed').innerHTML = `${{liveSpeedKmh.toFixed(1)}}<span class="cell-unit">km/h</span>`;

            const currT = p1.t + (p2.t - p1.t) * alpha;
            document.getElementById('time-curr').innerText = currT.toFixed(2) + 's';
            document.getElementById('time-slider').value = (progress * 100).toFixed(1);

            updateCinematicCamera(ballX, ballY, ballZ, currT);
        }}

        const animClock = new THREE.Clock();

        function animate() {{
            requestAnimationFrame(animate);

            const rawDelta = animClock.getDelta();
            const dt = Math.min(rawDelta, 0.05) * (isSlowmo ? 0.25 : 1.0);

            if (fullTrajectory.length > 0) {{
                const startP = currentShot.launch_pos;
                const totalDuration = fullTrajectory[fullTrajectory.length - 1].t;

                if (playState === 'READY') {{
                    ballMesh.position.set(startP.x, BALL_RADIUS, startP.y);
                    
                    if (isResettingCamera) {{
                        const targetCam = (cameraMode === 'CHASE') ? new THREE.Vector3(-18, 14, 0) : new THREE.Vector3(-14, 4.5, 0);
                        const targetPivot = (cameraMode === 'CHASE') ? new THREE.Vector3(startP.x, BALL_RADIUS, startP.y) : new THREE.Vector3(startP.x + 20, 3, startP.y);
                        camera.position.lerp(targetCam, 0.08);
                        controls.target.lerp(targetPivot, 0.12);
                        if (camera.position.distanceTo(targetCam) < 0.1) {{
                            isResettingCamera = false;
                        }}
                    }}
                }}
                else if (playState === 'RUNNING') {{
                    const maxIdx = fullTrajectory.length - 1;
                    const exactIdx = simProgress * maxIdx;
                    const currIdx = Math.min(Math.floor(exactIdx), maxIdx);
                    const nextIdx = Math.min(currIdx + 1, maxIdx);
                    const alpha = exactIdx - currIdx;

                    const p1 = fullTrajectory[currIdx];
                    const p2 = fullTrajectory[nextIdx];

                    const ballX = p1.x + (p2.x - p1.x) * alpha;
                    const ballY = (p1.z + (p2.z - p1.z) * alpha) + BALL_RADIUS;
                    const ballZ = p1.y + (p2.y - p1.y) * alpha;

                    ballMesh.position.set(ballX, ballY, ballZ);

                    const curVx = p1.vx + (p2.vx - p1.vx) * alpha;
                    const curVz = p1.vy + (p2.vy - p1.vy) * alpha;
                    const horizSpeed = Math.sqrt(curVx * curVx + curVz * curVz) + 1e-6;

                    const curDist = p1.dist + (p2.dist - p1.dist) * alpha;
                    document.getElementById('val-dist').innerText = `${{curDist.toFixed(1)}}`;

                    const spinAxisX = -curVz / horizSpeed;
                    const spinAxisY = 0;
                    const spinAxisZ = curVx / horizSpeed;
                    const spinAxis = new THREE.Vector3(spinAxisX, spinAxisY, spinAxisZ);

                    const isEnd = (simProgress >= 1.0 || currIdx >= maxIdx - 1);
                    let liveSpinRPM = isEnd ? 0.0 : (p1.spin_rpm + (p2.spin_rpm - p1.spin_rpm) * alpha);
                    let liveSpeedKmh = isEnd ? 0.0 : Math.sqrt(curVx**2 + curVz**2 + (p1.vz + (p2.vz - p1.vz)*alpha)**2) * 3.6;

                    document.getElementById('val-spin').innerHTML = `${{Math.round(liveSpinRPM)}}<span class="cell-unit">rpm</span>`;
                    document.getElementById('val-speed').innerHTML = `${{liveSpeedKmh.toFixed(1)}}<span class="cell-unit">km/h</span>`;

                    const isFlying = p1.is_flying;
                    const isRolling = p1.is_rolling;

                    let dTheta = 0.0;
                    if (liveSpinRPM > 0.1 || horizSpeed > 0.02) {{
                        if (isFlying) {{
                            // True aerodynamic backspin while airborne
                            const radPerSec = liveSpinRPM * (2.0 * Math.PI / 60.0);
                            dTheta = -radPerSec * dt * 0.08;
                        }} else {{
                            // Physical forward overspin roll upon ground contact
                            dTheta = +(horizSpeed / BALL_RADIUS) * dt;
                        }}
                        const deltaQuat = new THREE.Quaternion().setFromAxisAngle(spinAxis, dTheta);
                        ballQuaternion.multiplyQuaternions(deltaQuat, ballQuaternion);
                        ballMesh.quaternion.copy(ballQuaternion);
                    }}

                    const currT = p1.t + (p2.t - p1.t) * alpha;
                    document.getElementById('time-curr').innerText = currT.toFixed(2) + 's';
                    document.getElementById('time-slider').value = (simProgress * 100).toFixed(1);

                    updateCinematicCamera(ballX, ballY, ballZ, currT);

                    const progressStep = dt / Math.max(totalDuration, 1.0);
                    simProgress += progressStep;

                    if (simProgress >= 1.0) {{
                        simProgress = 1.0;
                        playState = 'FINISHED';
                        document.getElementById('btn-action').innerText = 'REPLAY';
                    }}
                }}
            }}

            controls.update();
            renderer.render(scene, camera);
        }}

        // Dual Camera System:
        // 1. CHASE CAM: Top-angled elevated chase camera with ball locked dead-center in viewport (zoom scaled)
        // 2. ORIGIN CAM: Stays at the origin/tee box and tracks the ball downrange (zoom scaled)
        function updateCinematicCamera(bx, by, bz, currT) {{
            const targetLookAt = new THREE.Vector3(bx, by, bz);

            if (!userInteracting) {{
                if (cameraMode === 'CHASE') {{
                    const camDistBehind = 18.0 * userZoomFactor;
                    const camHeightAbove = 14.0 * userZoomFactor;
                    const targetCamPos = new THREE.Vector3(bx - camDistBehind, by + camHeightAbove, bz);

                    // Ball dead-center at all times
                    controls.target.copy(targetLookAt);
                    camera.position.lerp(targetCamPos, 0.18);
                }} else if (cameraMode === 'ORIGIN') {{
                    // Camera fixed at tee box origin (distance scaled by zoom)
                    const originCamPos = new THREE.Vector3(-14.0 * userZoomFactor, 4.5 * userZoomFactor, 0.0);
                    camera.position.lerp(originCamPos, 0.15);
                    
                    // Smoothly track/pan the ball as it flies downrange
                    controls.target.lerp(targetLookAt, 0.22);
                }}
            }}
        }}

        function toggleCameraMode() {{
            cameraMode = (cameraMode === 'CHASE') ? 'ORIGIN' : 'CHASE';
            const camBtn = document.getElementById('btn-cam-mode');
            camBtn.innerText = `CAM: ${{cameraMode}}`;
            camBtn.classList.toggle('active', cameraMode === 'ORIGIN');

            // Instantly transition target orientation
            if (playState === 'READY') {{
                const startP = currentShot.launch_pos;
                if (cameraMode === 'CHASE') {{
                    camera.position.set(-18 * userZoomFactor, 14 * userZoomFactor, 0);
                    controls.target.set(startP.x, BALL_RADIUS, startP.y);
                }} else {{
                    camera.position.set(-14 * userZoomFactor, 4.5 * userZoomFactor, 0);
                    controls.target.set(startP.x + 20, 3, startP.y);
                }}
            }}
        }}

        function onWindowResize() {{
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, window.innerHeight);
        }}

        window.addEventListener('DOMContentLoaded', () => {{
            initScene();

            const select = document.getElementById('shot-dropdown');
            SHOTS.forEach((s, idx) => {{
                const opt = document.createElement('option');
                opt.value = idx;
                opt.innerText = `Shot #${{idx + 1}}: ${{s.club}} (${{s.total_distance_m}}m - ${{s.launch_spin_rpm}} RPM)`;
                select.appendChild(opt);
            }});

            select.addEventListener('change', (e) => {{
                loadShot(parseInt(e.target.value));
            }});

            const actionBtn = document.getElementById('btn-action');
            actionBtn.addEventListener('click', handleActionButton);

            const slowmoBtn = document.getElementById('btn-slowmo');
            slowmoBtn.addEventListener('click', () => {{
                isSlowmo = !isSlowmo;
                slowmoBtn.classList.toggle('active', isSlowmo);
                slowmoBtn.innerText = isSlowmo ? '0.25x SPEED' : 'SLOW-MO';
            }});

            const camBtn = document.getElementById('btn-cam-mode');
            camBtn.addEventListener('click', toggleCameraMode);

            const slider = document.getElementById('time-slider');
            slider.addEventListener('input', (e) => {{
                const targetProgress = parseFloat(e.target.value) / 100.0;
                seekAndPause(targetProgress);
            }});

            window.addEventListener('keydown', (e) => {{
                if (e.code === 'Space' || e.key.toLowerCase() === 'r') {{
                    e.preventDefault();
                    handleActionButton();
                }} else if (e.key.toLowerCase() === 'c') {{
                    e.preventDefault();
                    toggleCameraMode();
                }} else if (e.key === '=' || e.key === '+') {{
                    userZoomFactor = Math.max(0.15, userZoomFactor - 0.15);
                }} else if (e.key === '-' || e.key === '_') {{
                    userZoomFactor = Math.min(4.5, userZoomFactor + 0.15);
                }}
            }});

            loadShot(0);
            animate();
        }});
    </script>
</body>
</html>
"""
    return html

if __name__ == '__main__':
    build_3d_simulator()
