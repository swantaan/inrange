"""
scratch/generate_visualizer_gui.py

Builds the executive, TrackMan-style 3D radar visualizer HTML.
- Zero neon colors: uses refined dark slate, pearl white, amber gold, brick red, and forest emerald.
- Multi-shot browser: lets user select and inspect ANY shot (1,050 shots: 491 train + 559 test).
- Dynamic PCHIP spline interpolation running in-browser with exact checkpoint matching.
- Live 60 FPS animation with scrubbable timeline, playback speed, and camera presets.
"""

import os
import json
import numpy as np
import pandas as pd

def build_visualizer_html(project_root, output_path):
    train_path = os.path.join(project_root, 'data', 'train.csv')
    test_path = os.path.join(project_root, 'data', 'test.csv')
    sub_path = os.path.join(project_root, 'results', 'hybrid_predictions.csv')

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    sub_df = pd.read_csv(sub_path) if os.path.exists(sub_path) else None

    if sub_df is not None:
        test_full = pd.merge(test_df, sub_df, on='track_id')
    else:
        test_full = test_df.copy()

    train_df['split'] = 'Train'
    test_full['split'] = 'Test (Pred)'

    combined = pd.concat([train_df, test_full], ignore_index=True)

    shots_data = []
    for idx, row in combined.iterrows():
        vx, vy, vz = float(row['launch_vx']), float(row['launch_vy']), float(row['launch_vz'])
        v0 = np.sqrt(vx**2 + vy**2 + vz**2)
        launch_angle = np.degrees(np.arctan2(vz, np.sqrt(vx**2 + vy**2)))
        launch_dir = np.degrees(np.arctan2(vy, vx))
        carry = float(row['landing_x'])

        # Club inference category
        if carry >= 210:
            club = "Driver / 3-Wood"
            cat = "driver"
        elif carry >= 160:
            club = "Long Iron (3-5i)"
            cat = "long_iron"
        elif carry >= 120:
            club = "Mid Iron (6-8i)"
            cat = "mid_iron"
        elif carry >= 80:
            club = "Short Iron / 9i"
            cat = "short_iron"
        else:
            club = "Wedge / Pitching"
            cat = "wedge"

        shot_item = {
            "idx": int(idx),
            "track_id": str(row['track_id']),
            "split": str(row['split']),
            "club": club,
            "cat": cat,
            "v0": round(v0, 2),
            "vx": round(vx, 2),
            "vy": round(vy, 2),
            "vz": round(vz, 2),
            "launch_angle": round(launch_angle, 1),
            "launch_dir": round(launch_dir, 1),
            "spin": int(round(float(row['launch_spin_rate']))),
            "lx": round(float(row['launch_x']), 2),
            "ly": round(float(row['launch_y']), 2),
            "lz": round(float(row['launch_z']), 2),
            "cp1_t": round(float(row['cp1_t']), 3),
            "cp1_x": round(float(row['cp1_x']), 2),
            "cp1_y": round(float(row['cp1_y']), 2),
            "cp1_z": round(float(row['cp1_z']), 2),
            "cp2_t": round(float(row['cp2_t']), 3),
            "cp2_x": round(float(row['cp2_x']), 2),
            "cp2_y": round(float(row['cp2_y']), 2),
            "cp2_z": round(float(row['cp2_z']), 2),
            "cp3_t": round(float(row['cp3_t']), 3),
            "cp3_x": round(float(row['cp3_x']), 2),
            "cp3_y": round(float(row['cp3_y']), 2),
            "cp3_z": round(float(row['cp3_z']), 2),
            "cp4_t": round(float(row['cp4_t']), 3),
            "cp4_x": round(float(row['cp4_x']), 2),
            "cp4_y": round(float(row['cp4_y']), 2),
            "cp4_z": round(float(row['cp4_z']), 2),
            "apex_t": round(float(row['apex_t']), 2),
            "apex_x": round(float(row['apex_x']), 2),
            "apex_y": round(float(row['apex_y']), 2),
            "apex_z": round(float(row['apex_z']), 2),
            "landing_t": round(float(row['landing_t']), 2),
            "landing_x": round(float(row['landing_x']), 2),
            "landing_y": round(float(row['landing_y']), 2),
            "landing_z": round(float(row['landing_z']), 2)
        }
        shots_data.append(shot_item)

    shots_json = json.dumps(shots_data)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Inrange 3D Trajectory Visualizer | TrackMan-Style Radar Pro</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
  <style>
    :root {{
      --bg-darkest: #0a0e17;
      --bg-surface: #111827;
      --bg-card: #182234;
      --bg-card-hover: #1f2d44;
      --border-subtle: #25334a;
      --border-accent: #3b557d;
      --text-main: #f1f5f9;
      --text-muted: #94a3b8;
      --text-dim: #64748b;
      
      /* Executive TrackMan Color System - NO NEON */
      --color-ball: #ffffff;
      --color-flight: #e2e8f0;
      --color-cp: #d97706;      /* Refined warm amber */
      --color-net: #b91c1c;     /* Muted brick red */
      --color-apex: #7c3aed;    /* Subdued royal violet */
      --color-land: #059669;    /* Deep forest emerald */
      --color-roll: #b45309;    /* Muted earth ochre */
      --color-primary: #3b82f6; /* Classic deep sapphire */
    }}

    * {{
      box-sizing: border-box;
      margin: 0;
      padding: 0;
    }}

    body {{
      background-color: var(--bg-darkest);
      color: var(--text-main);
      font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
      overflow-x: hidden;
      display: flex;
      flex-direction: column;
      height: 100vh;
    }}

    /* Header & Navigation Bar */
    header {{
      background: var(--bg-surface);
      border-bottom: 1px solid var(--border-subtle);
      padding: 10px 20px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      flex-shrink: 0;
    }}

    .brand-section {{
      display: flex;
      align-items: center;
      gap: 12px;
    }}

    .brand-logo {{
      display: flex;
      align-items: center;
      justify-content: center;
      width: 34px;
      height: 34px;
      background: linear-gradient(135deg, #1e293b, #0f172a);
      border: 1px solid var(--border-accent);
      border-radius: 8px;
      color: #38bdf8;
      font-weight: 700;
      font-size: 16px;
    }}

    .brand-title {{
      font-size: 15px;
      font-weight: 700;
      letter-spacing: 0.5px;
      color: var(--text-main);
    }}

    .brand-sub {{
      font-size: 11px;
      color: var(--text-dim);
      font-weight: 400;
      display: block;
    }}

    /* Shot Selection Bar */
    .controls-center {{
      display: flex;
      align-items: center;
      gap: 10px;
      flex: 1;
      max-width: 820px;
    }}

    .shot-nav-btn {{
      background: var(--bg-card);
      border: 1px solid var(--border-subtle);
      color: var(--text-main);
      padding: 7px 12px;
      border-radius: 6px;
      font-size: 13px;
      font-weight: 500;
      cursor: pointer;
      display: flex;
      align-items: center;
      gap: 4px;
      transition: all 0.15s ease;
      white-space: nowrap;
    }}

    .shot-nav-btn:hover {{
      background: var(--bg-card-hover);
      border-color: var(--border-accent);
      color: #ffffff;
    }}

    .shot-dropdown-wrap {{
      flex: 1;
      position: relative;
    }}

    select.shot-select {{
      width: 100%;
      background: var(--bg-card);
      border: 1px solid var(--border-subtle);
      color: var(--text-main);
      padding: 7px 12px;
      border-radius: 6px;
      font-size: 13px;
      font-family: inherit;
      cursor: pointer;
      outline: none;
      transition: border-color 0.15s ease;
    }}

    select.shot-select:focus {{
      border-color: var(--color-primary);
    }}

    .quick-jump-wrap {{
      display: flex;
      align-items: center;
      gap: 6px;
    }}

    input.jump-input {{
      width: 80px;
      background: var(--bg-card);
      border: 1px solid var(--border-subtle);
      color: var(--text-main);
      padding: 7px 10px;
      border-radius: 6px;
      font-size: 13px;
      font-family: 'JetBrains Mono', monospace;
      text-align: center;
      outline: none;
    }}

    input.jump-input:focus {{
      border-color: var(--color-primary);
    }}

    /* Filter Chips */
    .filter-chips {{
      display: flex;
      gap: 6px;
      margin-left: 4px;
    }}

    .chip-btn {{
      background: transparent;
      border: 1px solid var(--border-subtle);
      color: var(--text-muted);
      padding: 5px 9px;
      border-radius: 12px;
      font-size: 11px;
      font-weight: 500;
      cursor: pointer;
      transition: all 0.15s ease;
    }}

    .chip-btn:hover {{
      color: var(--text-main);
      border-color: var(--text-dim);
    }}

    .chip-btn.active {{
      background: #1e3a8a;
      border-color: #3b82f6;
      color: #ffffff;
    }}

    /* Telemetry HUD Cards Grid */
    .telemetry-bar {{
      background: var(--bg-surface);
      border-bottom: 1px solid var(--border-subtle);
      padding: 10px 20px;
      display: grid;
      grid-template-columns: repeat(8, 1fr);
      gap: 12px;
      flex-shrink: 0;
    }}

    .metric-card {{
      background: var(--bg-card);
      border: 1px solid var(--border-subtle);
      border-radius: 8px;
      padding: 8px 12px;
      display: flex;
      flex-direction: column;
      justify-content: center;
      position: relative;
      overflow: hidden;
    }}

    .metric-card::before {{
      content: '';
      position: absolute;
      top: 0;
      left: 0;
      width: 3px;
      height: 100%;
      background: var(--border-accent);
    }}

    .metric-card.accent-speed::before {{ background: #38bdf8; }}
    .metric-card.accent-spin::before {{ background: #d97706; }}
    .metric-card.accent-carry::before {{ background: #059669; }}
    .metric-card.accent-apex::before {{ background: #7c3aed; }}
    .metric-card.accent-time::before {{ background: #64748b; }}

    .metric-label {{
      font-size: 10px;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.6px;
      color: var(--text-muted);
      margin-bottom: 3px;
    }}

    .metric-val-row {{
      display: flex;
      align-items: baseline;
      gap: 4px;
    }}

    .metric-value {{
      font-size: 19px;
      font-weight: 700;
      font-family: 'JetBrains Mono', monospace;
      color: var(--text-main);
    }}

    .metric-unit {{
      font-size: 11px;
      color: var(--text-dim);
      font-weight: 500;
    }}

    .metric-sub {{
      font-size: 11px;
      color: var(--text-dim);
      margin-top: 2px;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }}

    /* Main Viewport Workspace */
    .viewport-container {{
      flex: 1;
      position: relative;
      background: #080d15;
      display: flex;
      flex-direction: column;
      overflow: hidden;
    }}

    #plot3d {{
      width: 100%;
      flex: 1;
      min-height: 400px;
    }}

    /* Floating View Controls */
    .floating-camera-bar {{
      position: absolute;
      top: 16px;
      right: 20px;
      display: flex;
      gap: 8px;
      z-index: 10;
      background: rgba(17, 24, 39, 0.85);
      backdrop-filter: blur(8px);
      border: 1px solid var(--border-subtle);
      border-radius: 8px;
      padding: 6px;
    }}

    .cam-btn {{
      background: transparent;
      border: none;
      color: var(--text-muted);
      padding: 6px 10px;
      border-radius: 5px;
      font-size: 12px;
      font-weight: 500;
      cursor: pointer;
      display: flex;
      align-items: center;
      gap: 5px;
      transition: all 0.15s ease;
    }}

    .cam-btn:hover {{
      background: var(--bg-card);
      color: var(--text-main);
    }}

    .cam-btn.active {{
      background: #1e293b;
      color: #38bdf8;
      border: 1px solid #334155;
    }}

    /* Floating Checkpoint Legend Overlay */
    .floating-legend {{
      position: absolute;
      top: 16px;
      left: 20px;
      z-index: 10;
      background: rgba(17, 24, 39, 0.88);
      backdrop-filter: blur(8px);
      border: 1px solid var(--border-subtle);
      border-radius: 8px;
      padding: 10px 14px;
      font-size: 12px;
      display: flex;
      flex-direction: column;
      gap: 6px;
      pointer-events: none;
    }}

    .legend-item {{
      display: flex;
      align-items: center;
      gap: 8px;
      color: var(--text-muted);
    }}

    .legend-dot {{
      width: 9px;
      height: 9px;
      border-radius: 50%;
    }}

    /* Bottom Playback & Scrubber Controls */
    .playback-bar {{
      background: var(--bg-surface);
      border-top: 1px solid var(--border-subtle);
      padding: 12px 24px;
      display: flex;
      align-items: center;
      gap: 16px;
      z-index: 10;
      flex-shrink: 0;
    }}

    .play-btn {{
      background: #2563eb;
      border: none;
      color: #ffffff;
      width: 36px;
      height: 36px;
      border-radius: 50%;
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 14px;
      transition: all 0.15s ease;
      flex-shrink: 0;
    }}

    .play-btn:hover {{
      background: #1d4ed8;
      transform: scale(1.05);
    }}

    .timeline-container {{
      flex: 1;
      display: flex;
      align-items: center;
      gap: 12px;
    }}

    .timeline-slider {{
      flex: 1;
      -webkit-appearance: none;
      height: 6px;
      border-radius: 3px;
      background: #25334a;
      outline: none;
      cursor: pointer;
    }}

    .timeline-slider::-webkit-slider-thumb {{
      -webkit-appearance: none;
      appearance: none;
      width: 14px;
      height: 14px;
      border-radius: 50%;
      background: #38bdf8;
      border: 2px solid #0f172a;
      cursor: pointer;
      box-shadow: 0 0 8px rgba(56, 189, 248, 0.4);
    }}

    .timeline-slider::-moz-range-thumb {{
      width: 14px;
      height: 14px;
      border-radius: 50%;
      background: #38bdf8;
      border: 2px solid #0f172a;
      cursor: pointer;
    }}

    .time-status {{
      font-family: 'JetBrains Mono', monospace;
      font-size: 13px;
      color: var(--text-main);
      white-space: nowrap;
      min-width: 130px;
    }}

    .speed-select {{
      background: var(--bg-card);
      border: 1px solid var(--border-subtle);
      color: var(--text-main);
      padding: 5px 8px;
      border-radius: 6px;
      font-size: 12px;
      outline: none;
      cursor: pointer;
    }}

    @media (max-width: 1100px) {{
      .telemetry-bar {{
        grid-template-columns: repeat(4, 1fr);
      }}
    }}
  </style>
</head>
<body>

  <!-- Top Header Navigation -->
  <header>
    <div class="brand-section">
      <div class="brand-logo">IN</div>
      <div>
        <span class="brand-title">INRANGE RADAR 3D PRO</span>
        <span class="brand-sub">Atmospheric Ballistics &amp; Radar Gate Verification</span>
      </div>
    </div>

    <!-- Center Shot Selection Controls -->
    <div class="controls-center">
      <button class="shot-nav-btn" id="btnPrev" title="Previous Shot (Left Arrow)">◀ Prev</button>
      
      <div class="shot-dropdown-wrap">
        <select class="shot-select" id="shotSelect"></select>
      </div>

      <button class="shot-nav-btn" id="btnNext" title="Next Shot (Right Arrow)">Next</button>
      <button class="shot-nav-btn" id="btnRandom" title="Pick Random Shot">Random</button>

      <div class="quick-jump-wrap">
        <input class="jump-input" id="jumpInput" type="number" min="0" placeholder="# Shot" title="Type index 0-1049 and hit Enter">
        <button class="shot-nav-btn" id="btnJump">Go</button>
      </div>
    </div>

    <!-- Filter Chips -->
    <div class="filter-chips">
      <button class="chip-btn active" data-filter="all">All (1050)</button>
      <button class="chip-btn" data-filter="driver">Drivers</button>
      <button class="chip-btn" data-filter="irons">Irons</button>
      <button class="chip-btn" data-filter="wedges">Wedges</button>
      <button class="chip-btn" data-filter="train">Train (491)</button>
      <button class="chip-btn" data-filter="test">Test (559)</button>
    </div>
  </header>

  <!-- Telemetry HUD Grid (TrackMan Pro Specs) -->
  <div class="telemetry-bar">
    <div class="metric-card accent-speed">
      <span class="metric-label">Ball Speed</span>
      <div class="metric-val-row">
        <span class="metric-value" id="hudSpeed">--</span>
        <span class="metric-unit">m/s</span>
      </div>
      <span class="metric-sub" id="hudSpeedMph">-- mph</span>
    </div>

    <div class="metric-card">
      <span class="metric-label">Launch Angle</span>
      <div class="metric-val-row">
        <span class="metric-value" id="hudLaunchAngle">--</span>
        <span class="metric-unit">deg</span>
      </div>
      <span class="metric-sub">Vertical Loft</span>
    </div>

    <div class="metric-card">
      <span class="metric-label">Launch Direction</span>
      <div class="metric-val-row">
        <span class="metric-value" id="hudLaunchDir">--</span>
        <span class="metric-unit">deg</span>
      </div>
      <span class="metric-sub" id="hudLaunchDirText">--</span>
    </div>

    <div class="metric-card accent-spin">
      <span class="metric-label">Spin Rate</span>
      <div class="metric-val-row">
        <span class="metric-value" id="hudSpin">--</span>
        <span class="metric-unit">RPM</span>
      </div>
      <span class="metric-sub">Supercritical Wake</span>
    </div>

    <div class="metric-card accent-carry">
      <span class="metric-label">Carry Distance</span>
      <div class="metric-val-row">
        <span class="metric-value" id="hudCarry">--</span>
        <span class="metric-unit">m</span>
      </div>
      <span class="metric-sub" id="hudCarryYards">-- yds</span>
    </div>

    <div class="metric-card accent-apex">
      <span class="metric-label">Apex Height</span>
      <div class="metric-val-row">
        <span class="metric-value" id="hudApexZ">--</span>
        <span class="metric-unit">m</span>
      </div>
      <span class="metric-sub" id="hudApexSub">-- ft @ --s</span>
    </div>

    <div class="metric-card accent-time">
      <span class="metric-label">Flight Duration</span>
      <div class="metric-val-row">
        <span class="metric-value" id="hudHangTime">--</span>
        <span class="metric-unit">s</span>
      </div>
      <span class="metric-sub">Hang Time</span>
    </div>

    <div class="metric-card">
      <span class="metric-label">Total Distance</span>
      <div class="metric-val-row">
        <span class="metric-value" id="hudTotal">--</span>
        <span class="metric-unit">m</span>
      </div>
      <span class="metric-sub" id="hudRolloutSub">+--m Rollout</span>
    </div>
  </div>

  <!-- 3D Plot Viewport -->
  <div class="viewport-container">
    <!-- Camera View Controls -->
    <div class="floating-camera-bar">
      <button class="cam-btn active" id="camTee">Tee View</button>
      <button class="cam-btn" id="camSide">Side Arc</button>
      <button class="cam-btn" id="camTop">Top Down</button>
      <button class="cam-btn" id="camFollow">Follow Ball</button>
      <button class="cam-btn" id="camReset">Reset</button>
    </div>

    <!-- Exact Checkpoints Legend -->
    <div class="floating-legend">
      <div class="legend-item"><div class="legend-dot" style="background: #e2e8f0;"></div> Trajectory (Exact Gate Spline)</div>
      <div class="legend-item"><div class="legend-dot" style="background: #d97706;"></div> Radar Gates (CP1 15m, CP2 30m, CP3 45m)</div>
      <div class="legend-item"><div class="legend-dot" style="background: #b91c1c;"></div> CP4 (60m Net Barrier)</div>
      <div class="legend-item"><div class="legend-dot" style="background: #7c3aed;"></div> Apex Peak Height</div>
      <div class="legend-item"><div class="legend-dot" style="background: #059669;"></div> Landing Impact Ground</div>
      <div class="legend-item"><div class="legend-dot" style="background: #b45309;"></div> Bounce &amp; Decelerating Rollout</div>
    </div>

    <div id="plot3d"></div>
  </div>

  <!-- Bottom Playback Toolbar -->
  <div class="playback-bar">
    <button class="play-btn" id="playBtn" title="Play/Pause Flight (Spacebar)">▶</button>
    
    <div class="timeline-container">
      <input type="range" class="timeline-slider" id="timelineSlider" min="0" max="1000" value="0">
      <span class="time-status" id="timeStatus">t = 0.00s | 0.0m</span>
    </div>

    <select class="speed-select" id="speedSelect">
      <option value="0.5">0.5x Speed</option>
      <option value="1.0" selected>1.0x Normal</option>
      <option value="1.5">1.5x Fast</option>
      <option value="2.0">2.0x Turbo</option>
    </select>
  </div>

  <script>
    // Embedded Shots Dataset (1,050 shots)
    const ALL_SHOTS = {shots_json};

    let currentShotIdx = 0;
    let currentFilter = 'all';
    let trajectoryPoints = [];
    let rolloutPoints = [];
    let fullTrajectory = []; // array of {{t, x, y, z, isFlight}}
    let maxTime = 1.0;
    let flightTime = 1.0;
    
    // Animation state
    let isPlaying = false;
    let animProgress = 0; // 0.0 to 1.0
    let playSpeed = 1.0;
    let lastAnimTime = 0;
    let followBallActive = false;

    // Camera viewpoints
    const CAMERAS = {{
      tee: {{ eye: {{ x: -0.6, y: -1.9, z: 0.65 }}, center: {{ x: 0.4, y: 0.0, z: 0.1 }} }},
      side: {{ eye: {{ x: 0.2, y: -2.3, z: 0.25 }}, center: {{ x: 0.4, y: 0.0, z: 0.15 }} }},
      top: {{ eye: {{ x: 0.3, y: 0.0, z: 2.5 }}, center: {{ x: 0.4, y: 0.0, z: 0.0 }} }},
      reset: {{ eye: {{ x: 1.4, y: 1.4, z: 0.8 }}, center: {{ x: 0.0, y: 0.0, z: 0.0 }} }}
    }};

    // Monotonic Cubic Hermite (PCHIP) Interpolator in 3D
    function computePchipTrajectory(shot) {{
      // Anchor Control Points (chronological order)
      const rawPoints = [
        {{ t: 0.0, x: shot.lx, y: shot.ly, z: shot.lz }},
        {{ t: shot.cp1_t, x: shot.cp1_x, y: shot.cp1_y, z: shot.cp1_z }},
        {{ t: shot.cp2_t, x: shot.cp2_x, y: shot.cp2_y, z: shot.cp2_z }},
        {{ t: shot.cp3_t, x: shot.cp3_x, y: shot.cp3_y, z: shot.cp3_z }},
        {{ t: shot.cp4_t, x: shot.cp4_x, y: shot.cp4_y, z: shot.cp4_z }},
        {{ t: shot.apex_t, x: shot.apex_x, y: shot.apex_y, z: shot.apex_z }},
        {{ t: shot.landing_t, x: shot.landing_x, y: shot.landing_y, z: shot.landing_z }}
      ];

      // Sort strictly by time t (handles wedges where apex_t < cp4_t)
      rawPoints.sort((a, b) => a.t - b.t);

      const n = rawPoints.length;
      const t = rawPoints.map(p => p.t);
      const x = rawPoints.map(p => p.x);
      const y = rawPoints.map(p => p.y);
      const z = rawPoints.map(p => p.z);

      // PCHIP Slopes calculation
      function getSlopes(h, m) {{
        const d = new Float64Array(n);
        const kEnd = n - 1;
        // Internal slopes
        for (let i = 1; i < kEnd; i++) {{
          if (m[i - 1] * m[i] > 0) {{
            const w1 = 2 * h[i] + h[i - 1];
            const w2 = h[i] + 2 * h[i - 1];
            d[i] = (w1 + w2) / (w1 / m[i - 1] + w2 / m[i]);
          }} else {{
            d[i] = 0.0;
          }}
        }}
        // End points (shape-preserving)
        d[0] = ((2 * h[0] + h[1]) * m[0] - h[0] * m[1]) / (h[0] + h[1]);
        if (d[0] * m[0] <= 0) d[0] = 0;
        else if (m[0] * m[1] < 0 && Math.abs(d[0]) > 3 * Math.abs(m[0])) d[0] = 3 * m[0];

        d[kEnd] = ((2 * h[kEnd - 1] + h[kEnd - 2]) * m[kEnd - 1] - h[kEnd - 1] * m[kEnd - 2]) / (h[kEnd - 1] + h[kEnd - 2]);
        if (d[kEnd] * m[kEnd - 1] <= 0) d[kEnd] = 0;
        else if (m[kEnd - 1] * m[kEnd - 2] < 0 && Math.abs(d[kEnd]) > 3 * Math.abs(m[kEnd - 1])) d[kEnd] = 3 * m[kEnd - 1];

        return d;
      }}

      const h = new Float64Array(n - 1);
      const mx = new Float64Array(n - 1);
      const my = new Float64Array(n - 1);
      const mz = new Float64Array(n - 1);

      for (let i = 0; i < n - 1; i++) {{
        h[i] = t[i + 1] - t[i];
        mx[i] = (x[i + 1] - x[i]) / h[i];
        my[i] = (y[i + 1] - y[i]) / h[i];
        mz[i] = (z[i + 1] - z[i]) / h[i];
      }}

      const dx = getSlopes(h, mx);
      const dy = getSlopes(h, my);
      const dz = getSlopes(h, mz);

      function evalPchip(timeVal) {{
        if (timeVal <= t[0]) return [x[0], y[0], z[0]];
        if (timeVal >= t[n - 1]) return [x[n - 1], y[n - 1], z[n - 1]];

        let idx = 0;
        for (let i = 0; i < n - 1; i++) {{
          if (timeVal >= t[i] && timeVal <= t[i + 1]) {{
            idx = i;
            break;
          }}
        }}

        const dt = timeVal - t[idx];
        const hi = h[idx];
        const s = dt / hi;
        const s2 = s * s;
        const s3 = s2 * s;

        const h00 = 2 * s3 - 3 * s2 + 1;
        const h10 = s3 - 2 * s2 + s;
        const h01 = -2 * s3 + 3 * s2;
        const h11 = s3 - s2;

        const evalX = h00 * x[idx] + h10 * hi * dx[idx] + h01 * x[idx + 1] + h11 * hi * dx[idx + 1];
        const evalY = h00 * y[idx] + h10 * hi * dy[idx] + h01 * y[idx + 1] + h11 * hi * dy[idx + 1];
        const evalZ = h00 * z[idx] + h10 * hi * dz[idx] + h01 * z[idx + 1] + h11 * hi * dz[idx + 1];

        return [evalX, evalY, Math.max(shot.lz, evalZ)];
      }}

      // Sample 120 smooth points along flight path
      const flightSamples = [];
      const numSamples = 120;
      const tEnd = shot.landing_t;
      for (let i = 0; i <= numSamples; i++) {{
        const curT = (i / numSamples) * tEnd;
        const pt = evalPchip(curT);
        flightSamples.push({{ t: curT, x: pt[0], y: pt[1], z: pt[2], isFlight: true }});
      }}

      // Simulate Ground Impact Bounce and Rollout
      const lastPt = flightSamples[flightSamples.length - 1];
      const prevPt = flightSamples[flightSamples.length - 2];
      const dtLand = lastPt.t - prevPt.t;
      let vx = (lastPt.x - prevPt.x) / dtLand;
      let vy = (lastPt.y - prevPt.y) / dtLand;
      let vz = -Math.abs((lastPt.z - prevPt.z) / dtLand);

      let rx = lastPt.x;
      let ry = lastPt.y;
      let rz = shot.lz;
      let rT = tEnd;
      const groundZ = shot.lz;

      const rolloutSamples = [];
      let bounceCount = 0;
      const dtStep = 0.02;

      // 1-2 small realistic hops followed by turf friction roll
      while (rT - tEnd < 3.5 && Math.hypot(vx, vy) > 0.3) {{
        rT += dtStep;
        rx += vx * dtStep;
        ry += vy * dtStep;
        rz += vz * dtStep - 0.5 * 9.81 * dtStep * dtStep;
        vz -= 9.81 * dtStep;

        if (rz <= groundZ) {{
          rz = groundZ;
          bounceCount++;
          if (bounceCount <= 2 && Math.abs(vz) > 1.0) {{
            vz = -vz * 0.32; // restitution
            vx *= 0.65;
            vy *= 0.65;
          }} else {{
            vz = 0.0;
            const speedH = Math.hypot(vx, vy);
            const decel = 0.28 * 9.81 * dtStep;
            if (speedH <= decel) {{
              vx = 0;
              vy = 0;
            }} else {{
              vx -= (vx / speedH) * decel;
              vy -= (vy / speedH) * decel;
            }}
          }}
        }}

        rolloutSamples.push({{ t: rT, x: rx, y: ry, z: rz, isFlight: false }});
      }}

      return {{
        flight: flightSamples,
        rollout: rolloutSamples,
        total: flightSamples.concat(rolloutSamples),
        tFlight: tEnd,
        tTotal: rolloutSamples.length ? rolloutSamples[rolloutSamples.length - 1].t : tEnd
      }};
    }}

    // Populate Shot Selection Dropdown
    function populateDropdown() {{
      const select = document.getElementById('shotSelect');
      select.innerHTML = '';

      ALL_SHOTS.forEach((s, idx) => {{
        // Check filter
        let show = true;
        if (currentFilter === 'driver' && s.cat !== 'driver') show = false;
        else if (currentFilter === 'irons' && s.cat !== 'long_iron' && s.cat !== 'mid_iron' && s.cat !== 'short_iron') show = false;
        else if (currentFilter === 'wedges' && s.cat !== 'wedge') show = false;
        else if (currentFilter === 'train' && s.split !== 'Train') show = false;
        else if (currentFilter === 'test' && !s.split.includes('Test')) show = false;

        if (show) {{
          const opt = document.createElement('option');
          opt.value = idx;
          opt.textContent = `Shot #${{idx}} [${{s.split}}]: ${{s.club}} — Carry: ${{s.landing_x}}m | Speed: ${{s.v0}} m/s | Apex: ${{s.apex_z}}m`;
          select.appendChild(opt);
        }}
      }});

      select.value = currentShotIdx;
    }}

    // Update HUD Telemetry Cards
    function updateHUD(shot, trajData) {{
      document.getElementById('hudSpeed').textContent = shot.v0.toFixed(1);
      document.getElementById('hudSpeedMph').textContent = `${{(shot.v0 * 2.23694).toFixed(1)}} mph`;
      
      document.getElementById('hudLaunchAngle').textContent = shot.launch_angle.toFixed(1);
      
      const dirVal = shot.launch_dir;
      document.getElementById('hudLaunchDir').textContent = (dirVal > 0 ? '+' : '') + dirVal.toFixed(1);
      document.getElementById('hudLaunchDirText').textContent = Math.abs(dirVal) < 1.0 ? 'Straight' : (dirVal > 0 ? 'Push / Fade' : 'Pull / Draw');

      document.getElementById('hudSpin').textContent = shot.spin.toLocaleString();
      
      document.getElementById('hudCarry').textContent = shot.landing_x.toFixed(1);
      document.getElementById('hudCarryYards').textContent = `${{(shot.landing_x * 1.09361).toFixed(1)}} yds`;

      document.getElementById('hudApexZ').textContent = shot.apex_z.toFixed(1);
      document.getElementById('hudApexSub').textContent = `${{(shot.apex_z * 3.28084).toFixed(0)}} ft @ ${{shot.apex_t.toFixed(1)}}s`;

      document.getElementById('hudHangTime').textContent = shot.landing_t.toFixed(2);

      const totalDist = trajData.rollout.length ? trajData.rollout[trajData.rollout.length - 1].x : shot.landing_x;
      const rollDist = totalDist - shot.landing_x;
      document.getElementById('hudTotal').textContent = totalDist.toFixed(1);
      document.getElementById('hudRolloutSub').textContent = `+${{rollDist.toFixed(1)}}m Rollout`;
    }}

    // Render 3D Plotly Canvas
    function render3DScene(shot, trajData) {{
      const f = trajData.flight;
      const r = trajData.rollout;

      // Radar Fairway Turf Grid
      const maxX = Math.max(260, shot.landing_x + 30);
      
      const traces = [
        // 0. Constrained Flight Trajectory (Passing 100% through all checkpoints)
        {{
          type: 'scatter3d',
          mode: 'lines',
          x: f.map(p => p.x),
          y: f.map(p => p.y),
          z: f.map(p => p.z),
          line: {{ color: '#f1f5f9', width: 4.5 }},
          name: 'Flight Path (Exact Radar Spline)'
        }},

        // 1. Rollout Phase (Warm Earth Ochre dashed)
        {{
          type: 'scatter3d',
          mode: 'lines',
          x: r.map(p => p.x),
          y: r.map(p => p.y),
          z: r.map(p => p.z),
          line: {{ color: '#b45309', width: 3, dash: 'dot' }},
          name: 'Bounce & Rollout'
        }},

        // 2. Launch Tee
        {{
          type: 'scatter3d',
          mode: 'markers+text',
          x: [shot.lx], y: [shot.ly], z: [shot.lz],
          marker: {{ size: 7, color: '#38bdf8', symbol: 'diamond' }},
          text: ['Tee'],
          textposition: 'top center',
          textfont: {{ color: '#94a3b8', size: 10 }},
          name: 'Launch Tee'
        }},

        // 3. Radar Checkpoints (CP1, CP2, CP3) - Warm Amber
        {{
          type: 'scatter3d',
          mode: 'markers+text',
          x: [shot.cp1_x, shot.cp2_x, shot.cp3_x],
          y: [shot.cp1_y, shot.cp2_y, shot.cp3_y],
          z: [shot.cp1_z, shot.cp2_z, shot.cp3_z],
          marker: {{ size: 7, color: '#d97706', symbol: 'circle' }},
          text: ['CP1 (15m)', 'CP2 (30m)', 'CP3 (45m)'],
          textposition: 'top center',
          textfont: {{ color: '#d97706', size: 10 }},
          name: 'Radar Gates (15m, 30m, 45m)'
        }},

        // 4. CP4 (60m Net Barrier) - Brick Crimson
        {{
          type: 'scatter3d',
          mode: 'markers+text',
          x: [shot.cp4_x], y: [shot.cp4_y], z: [shot.cp4_z],
          marker: {{ size: 9, color: '#b91c1c', symbol: 'square' }},
          text: ['CP4 (60m Net)'],
          textposition: 'top center',
          textfont: {{ color: '#ef4444', size: 11 }},
          name: 'Urban Net Barrier (60m)'
        }},

        // 5. Apex Peak Height - Royal Violet
        {{
          type: 'scatter3d',
          mode: 'markers+text',
          x: [shot.apex_x], y: [shot.apex_y], z: [shot.apex_z],
          marker: {{ size: 8, color: '#7c3aed', symbol: 'cross' }},
          text: [`Apex (${{shot.apex_z.toFixed(1)}}m)`],
          textposition: 'top center',
          textfont: {{ color: '#a78bfa', size: 10 }},
          name: 'Trajectory Apex'
        }},

        // 6. Landing Impact Ground - Forest Emerald
        {{
          type: 'scatter3d',
          mode: 'markers+text',
          x: [shot.landing_x], y: [shot.landing_y], z: [shot.landing_z],
          marker: {{ size: 8, color: '#059669', symbol: 'diamond' }},
          text: [`Landing (${{shot.landing_x.toFixed(1)}}m)`],
          textposition: 'bottom center',
          textfont: {{ color: '#34d399', size: 10 }},
          name: 'Ground Impact'
        }},

        // 7. Resting Ball
        {{
          type: 'scatter3d',
          mode: 'markers',
          x: [r.length ? r[r.length - 1].x : shot.landing_x],
          y: [r.length ? r[r.length - 1].y : shot.landing_y],
          z: [r.length ? r[r.length - 1].z : shot.landing_z],
          marker: {{ size: 6, color: '#b45309', symbol: 'circle' }},
          name: 'Rest Position'
        }},

        // 8. Animated Golf Ball (Index 8 for real-time updates)
        {{
          type: 'scatter3d',
          mode: 'markers',
          x: [f[0].x], y: [f[0].y], z: [f[0].z],
          marker: {{ size: 9, color: '#ffffff', symbol: 'circle', line: {{ color: '#38bdf8', width: 2 }} }},
          name: 'Live Golf Ball'
        }},

        // 9. Live Motion Trail (Index 9)
        {{
          type: 'scatter3d',
          mode: 'lines',
          x: [f[0].x], y: [f[0].y], z: [f[0].z],
          line: {{ color: '#ffffff', width: 5 }},
          name: 'Active Trail'
        }}
      ];

      const layout = {{
        title: false,
        paper_bgcolor: '#080d15',
        plot_bgcolor: '#080d15',
        margin: {{ l: 0, r: 0, t: 0, b: 0 }},
        showlegend: false,
        scene: {{
          xaxis: {{
            title: 'Downfield X (m)',
            color: '#64748b',
            gridcolor: '#1a2436',
            zerolinecolor: '#25334a',
            range: [-5, maxX],
            backgroundcolor: '#080d15'
          }},
          yaxis: {{
            title: 'Lateral Y (m)',
            color: '#64748b',
            gridcolor: '#1a2436',
            zerolinecolor: '#25334a',
            range: [-25, 25],
            backgroundcolor: '#080d15'
          }},
          zaxis: {{
            title: 'Elevation Z (m)',
            color: '#64748b',
            gridcolor: '#1a2436',
            zerolinecolor: '#25334a',
            range: [0, Math.max(35, shot.apex_z * 1.3)],
            backgroundcolor: '#080d15'
          }},
          camera: CAMERAS.tee,
          aspectratio: {{ x: 2.2, y: 0.8, z: 0.65 }}
        }}
      }};

      const config = {{
        responsive: true,
        displayModeBar: false
      }};

      Plotly.react('plot3d', traces, layout, config);
    }}

    // Load and Display Selected Shot
    function loadShot(idx) {{
      currentShotIdx = idx;
      const shot = ALL_SHOTS[idx];
      if (!shot) return;

      document.getElementById('shotSelect').value = idx;
      document.getElementById('jumpInput').value = idx;

      const trajData = computePchipTrajectory(shot);
      trajectoryPoints = trajData.flight;
      rolloutPoints = trajData.rollout;
      fullTrajectory = trajData.total;
      flightTime = trajData.tFlight;
      maxTime = trajData.tTotal;

      updateHUD(shot, trajData);
      render3DScene(shot, trajData);

      // Reset playback to start
      pausePlayback();
      setTimelineProgress(0);
    }}

    // Update Live Ball & Trail Position
    function updateBallPosition(prog) {{
      if (!fullTrajectory.length) return;

      const targetT = prog * maxTime;
      let curIdx = 0;
      for (let i = 0; i < fullTrajectory.length; i++) {{
        if (fullTrajectory[i].t <= targetT) curIdx = i;
        else break;
      }}

      const curPt = fullTrajectory[curIdx];
      const isFlying = curPt.isFlight;

      // Update Traces 8 & 9 directly with restyle for 60fps performance
      const updateData = {{
        x: [[curPt.x]],
        y: [[curPt.y]],
        z: [[curPt.z]],
        'marker.color': [isFlying ? '#ffffff' : '#b45309']
      }};

      const trailUpdate = {{
        x: [fullTrajectory.slice(0, curIdx + 1).map(p => p.x)],
        y: [fullTrajectory.slice(0, curIdx + 1).map(p => p.y)],
        z: [fullTrajectory.slice(0, curIdx + 1).map(p => p.z)],
        'line.color': [isFlying ? '#e2e8f0' : '#b45309']
      }};

      Plotly.restyle('plot3d', updateData, [8]);
      Plotly.restyle('plot3d', trailUpdate, [9]);

      // Dynamic Chase Cam if enabled
      if (followBallActive) {{
        const eyeX = (curPt.x - 25) / 100;
        const eyeY = (curPt.y - 10) / 100;
        const eyeZ = Math.max(0.2, (curPt.z + 10) / 100);
        Plotly.relayout('plot3d', {{
          'scene.camera.eye': {{ x: eyeX, y: eyeY, z: eyeZ }},
          'scene.camera.center': {{ x: curPt.x / 100, y: curPt.y / 100, z: curPt.z / 100 }}
        }});
      }}

      // Status text
      const statusText = isFlying
        ? `Flight: t = ${{curPt.t.toFixed(2)}}s | X: ${{curPt.x.toFixed(1)}}m | Z: ${{curPt.z.toFixed(1)}}m`
        : `Rollout: t = ${{curPt.t.toFixed(2)}}s | X: ${{curPt.x.toFixed(1)}}m | Z: ${{curPt.z.toFixed(1)}}m`;
      document.getElementById('timeStatus').textContent = statusText;
    }}

    function setTimelineProgress(prog) {{
      prog = Math.max(0, Math.min(1, prog));
      animProgress = prog;
      document.getElementById('timelineSlider').value = Math.round(prog * 1000);
      updateBallPosition(prog);
    }}

    // Playback Engine Loop
    function animLoop(timestamp) {{
      if (!isPlaying) return;

      if (!lastAnimTime) lastAnimTime = timestamp;
      const deltaSec = (timestamp - lastAnimTime) / 1000.0;
      lastAnimTime = timestamp;

      const duration = maxTime / playSpeed;
      animProgress += deltaSec / duration;

      if (animProgress >= 1.0) {{
        animProgress = 1.0;
        setTimelineProgress(1.0);
        pausePlayback();
        return;
      }}

      setTimelineProgress(animProgress);
      requestAnimationFrame(animLoop);
    }}

    function startPlayback() {{
      if (animProgress >= 1.0) animProgress = 0.0;
      isPlaying = true;
      lastAnimTime = 0;
      document.getElementById('playBtn').textContent = '⏸';
      requestAnimationFrame(animLoop);
    }}

    function pausePlayback() {{
      isPlaying = false;
      document.getElementById('playBtn').textContent = '▶';
    }}

    function togglePlayback() {{
      if (isPlaying) pausePlayback();
      else startPlayback();
    }}

    // Event Listeners Setup
    document.addEventListener('DOMContentLoaded', () => {{
      populateDropdown();
      loadShot(0);

      // Shot navigation
      document.getElementById('shotSelect').addEventListener('change', (e) => {{
        loadShot(parseInt(e.target.value));
      }});

      document.getElementById('btnPrev').addEventListener('click', () => {{
        const next = (currentShotIdx - 1 + ALL_SHOTS.length) % ALL_SHOTS.length;
        loadShot(next);
      }});

      document.getElementById('btnNext').addEventListener('click', () => {{
        const next = (currentShotIdx + 1) % ALL_SHOTS.length;
        loadShot(next);
      }});

      document.getElementById('btnRandom').addEventListener('click', () => {{
        const rand = Math.floor(Math.random() * ALL_SHOTS.length);
        loadShot(rand);
      }});

      document.getElementById('btnJump').addEventListener('click', () => {{
        const val = parseInt(document.getElementById('jumpInput').value);
        if (!isNaN(val) && val >= 0 && val < ALL_SHOTS.length) loadShot(val);
      }});

      document.getElementById('jumpInput').addEventListener('keydown', (e) => {{
        if (e.key === 'Enter') {{
          const val = parseInt(e.target.value);
          if (!isNaN(val) && val >= 0 && val < ALL_SHOTS.length) loadShot(val);
        }}
      }});

      // Keyboard shortcuts: Left/Right arrow for shots, Space for play
      document.addEventListener('keydown', (e) => {{
        if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;
        if (e.code === 'Space') {{
          e.preventDefault();
          togglePlayback();
        }} else if (e.code === 'ArrowLeft') {{
          e.preventDefault();
          const next = (currentShotIdx - 1 + ALL_SHOTS.length) % ALL_SHOTS.length;
          loadShot(next);
        }} else if (e.code === 'ArrowRight') {{
          e.preventDefault();
          const next = (currentShotIdx + 1) % ALL_SHOTS.length;
          loadShot(next);
        }}
      }});

      // Filter chips
      document.querySelectorAll('.chip-btn').forEach(btn => {{
        btn.addEventListener('click', () => {{
          document.querySelectorAll('.chip-btn').forEach(b => b.classList.remove('active'));
          btn.classList.add('active');
          currentFilter = btn.getAttribute('data-filter');
          populateDropdown();
        }});
      }});

      // Playback bar
      document.getElementById('playBtn').addEventListener('click', togglePlayback);

      document.getElementById('timelineSlider').addEventListener('input', (e) => {{
        pausePlayback();
        setTimelineProgress(parseInt(e.target.value) / 1000);
      }});

      document.getElementById('speedSelect').addEventListener('change', (e) => {{
        playSpeed = parseFloat(e.target.value);
      }});

      // Camera view presets
      function setCam(viewKey, btnId) {{
        followBallActive = false;
        document.querySelectorAll('.cam-btn').forEach(b => b.classList.remove('active'));
        document.getElementById(btnId).classList.add('active');
        Plotly.relayout('plot3d', {{ 'scene.camera': CAMERAS[viewKey] }});
      }}

      document.getElementById('camTee').addEventListener('click', () => setCam('tee', 'camTee'));
      document.getElementById('camSide').addEventListener('click', () => setCam('side', 'camSide'));
      document.getElementById('camTop').addEventListener('click', () => setCam('top', 'camTop'));
      document.getElementById('camReset').addEventListener('click', () => setCam('reset', 'camReset'));

      document.getElementById('camFollow').addEventListener('click', () => {{
        document.querySelectorAll('.cam-btn').forEach(b => b.classList.remove('active'));
        document.getElementById('camFollow').classList.add('active');
        followBallActive = true;
      }});
    }});
  </script>
</body>
</html>
"""

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"Successfully generated Visualizer GUI to {output_path} (Size: {len(html_content)/1024:.1f} KB)")


if __name__ == '__main__':
    project_root = r'c:\Users\csjak\OneDrive\Desktop\inrange'
    output_html = os.path.join(project_root, 'visualizer', 'hybrid_flight_animation.html')
    build_visualizer_html(project_root, output_html)
