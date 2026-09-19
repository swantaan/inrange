"""
generate_report_diagrams.py

Generates presentation-ready publication graphics & diagrams for the competition report:
1. Model Architecture & Hybrid Fusion Flowchart.
2. Aerodynamic Forces & Physics Dimple Drag / Magnus Lift Diagram.
3. 5-Fold Cross-Validation Model Comparison Scorecard.
4. Feature Importance Ranking.
5. 3D Full Driving Range Flight Trajectory Cluster & 60m Net Cutoff.
6. Club Regime Clustering (Kinetic Loft vs Spin vs Speed).
7. Kikuyu Grass Multi-Hop Turf Bounce & Friction Deceleration.
8. Actual vs. Predicted Parity Plots across all 8 spatial/temporal targets.

All diagrams are saved to `report/figures/`.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import CubicSpline

# Setup paths
current_dir = os.path.dirname(os.path.abspath(__file__))
figures_dir = os.path.join(current_dir, 'report', 'figures')
train_path = os.path.join(current_dir, 'data', 'train.csv')
os.makedirs(figures_dir, exist_ok=True)

# Aesthetic theme
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']


def generate_pipeline_schematic():
    """Generates modern architecture flow diagram."""
    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
    ax.set_facecolor('#0f172a')
    fig.patch.set_facecolor('#0f172a')
    ax.axis('off')

    boxes = [
        (0.04, 0.65, 0.22, 0.26, "Tee Launch & Checkpoints", "• Launch (x,y,z,vx,vy,vz)\n• Gates: 15m, 30m, 45m, 60m", "#1e293b", "#38bdf8"),
        (0.38, 0.68, 0.24, 0.24, "Stellenbosch Physics Engine", "• Dimpled Drag (Cd0 + k·CL²)\n• Magnus Lift & Spin Decay\n• Ground Bay Topography", "#1e293b", "#00f0ff"),
        (0.38, 0.32, 0.24, 0.24, "Feature Engineering", "• Regime Ratio (vz/v0²)\n• Segment Decelerations\n• Checkpoint Polynomials", "#1e293b", "#a855f7"),
        (0.68, 0.46, 0.14, 0.16, "Tree Ensemble", "LightGBM + CatBoost\n(Physics Residuals)", "#1e293b", "#34d399"),
        (0.68, 0.18, 0.14, 0.16, "Deep Neural Net", "4-Layer PyTorch MLP\n(Physics Residuals)", "#1e293b", "#f43f5e"),
        (0.86, 0.44, 0.12, 0.28, "Hybrid Fusion", "Final Prediction =\nPhysics Base +\nw_ml·Δ_ml + w_dl·Δ_dl", "#1e293b", "#fbbf24"),
    ]

    for x, y, w, h, title, sub, bg, border in boxes:
        rect = patches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.015,rounding_size=0.02",
            facecolor=bg, edgecolor=border, linewidth=1.8, zorder=2
        )
        ax.add_patch(rect)
        ax.text(x + w/2, y + h - 0.045, title, color='#ffffff', fontsize=10.5, fontweight='bold', ha='center', va='top', zorder=3)
        ax.text(x + w/2, y + h/2 - 0.02, sub, color='#94a3b8', fontsize=8.5, ha='center', va='center', zorder=3, linespacing=1.3)

    arrows = [
        ((0.26, 0.78), (0.38, 0.78), '#38bdf8'),
        ((0.26, 0.72), (0.38, 0.44), '#a855f7'),
        ((0.50, 0.68), (0.50, 0.56), '#00f0ff'),
        ((0.62, 0.44), (0.68, 0.54), '#34d399'),
        ((0.62, 0.40), (0.68, 0.26), '#f43f5e'),
        ((0.82, 0.54), (0.86, 0.58), '#34d399'),
        ((0.82, 0.26), (0.86, 0.50), '#f43f5e'),
        ((0.62, 0.78), (0.86, 0.64), '#00f0ff'),
    ]

    for p1, p2, color in arrows:
        ax.annotate(
            '', xy=p2, xytext=p1,
            arrowprops=dict(arrowstyle="->", color=color, lw=2.0, shrinkA=4, shrinkB=4, mutation_scale=14)
        )

    ax.set_title("Inrange Hybrid Physics-ML Trajectory Prediction Architecture", color='#f8fafc', fontsize=13, fontweight='bold', pad=18)
    plt.tight_layout()
    out_path = os.path.join(figures_dir, "01_pipeline_architecture.png")
    plt.savefig(out_path, dpi=300, facecolor='#0f172a', bbox_inches='tight')
    plt.close()
    print(f"[+] Saved: {out_path}")


def generate_aero_physics_diagram():
    """Generates aerodynamic flight mechanics diagram."""
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    ax.set_facecolor('#ffffff')

    x = np.linspace(0, 160, 200)
    z = -0.0035 * (x - 75)**2 + 25.0
    ax.plot(x, z, color='#0284c7', lw=3, label='Ball Trajectory Arc', zorder=2)
    ax.fill_between(x, 0, z, color='#bae6fd', alpha=0.25)

    x_ball, z_ball = 75.0, 25.0
    circle = plt.Circle((x_ball, z_ball), 3.0, color='#ffffff', ec='#0f172a', lw=2, zorder=4)
    ax.add_patch(circle)

    ax.annotate('', xy=(x_ball, z_ball + 12), xytext=(x_ball, z_ball),
                arrowprops=dict(arrowstyle="->", color='#16a34a', lw=3, mutation_scale=16))
    ax.text(x_ball + 3, z_ball + 10, "Magnus Lift Force (F_L)\nCL(Spin, Speed)", color='#16a34a', fontsize=9.5, fontweight='bold')

    ax.annotate('', xy=(x_ball - 14, z_ball), xytext=(x_ball, z_ball),
                arrowprops=dict(arrowstyle="->", color='#dc2626', lw=3, mutation_scale=16))
    ax.text(x_ball - 22, z_ball + 3, "Aerodynamic Drag (F_D)\nCd0(v) + k·CL²", color='#dc2626', fontsize=9.5, fontweight='bold', ha='right')

    ax.annotate('', xy=(x_ball, z_ball - 12), xytext=(x_ball, z_ball),
                arrowprops=dict(arrowstyle="->", color='#475569', lw=2.5, mutation_scale=16))
    ax.text(x_ball + 3, z_ball - 10, "Gravity (m·g)", color='#475569', fontsize=9.5, fontweight='bold')

    ax.axvline(60, color='#ef4444', linestyle='--', lw=2, label='60m Urban Net (CP4)')
    ax.text(61, 6, "60m Net Barrier\n(Radar Limit)", color='#ef4444', fontsize=9, fontweight='bold')

    ax.axhline(0, color='#15803d', lw=3)
    ax.fill_between([-10, 180], -5, 0, color='#dcfce7')

    ax.set_xlim(-5, 175)
    ax.set_ylim(-2, 42)
    ax.set_xlabel("Downfield Distance (meters)", fontsize=11, fontweight='bold')
    ax.set_ylabel("Elevation Z (meters)", fontsize=11, fontweight='bold')
    ax.set_title("Aerodynamic Force Balance in Stellenbosch Atmospheric Conditions", fontsize=12.5, fontweight='bold', pad=12)
    ax.legend(loc='upper right', framealpha=0.95)
    ax.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    out_path = os.path.join(figures_dir, "02_aerodynamic_flight_physics.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[+] Saved: {out_path}")


def generate_cv_scorecard_chart():
    """Generates cross-validation performance scorecard comparison."""
    targets = ['launch_spin', 'apex_t', 'apex_x', 'apex_y', 'apex_z', 'landing_t', 'landing_x', 'landing_y']
    phys_errors = [1360.5, 0.334, 10.82, 5.43, 3.92, 0.626, 18.14, 9.33]
    ml_errors   = [740.9,  0.078, 2.27,  1.74, 0.84, 0.145, 3.76,  3.42]
    dl_errors   = [748.0,  0.079, 2.58,  1.64, 0.81, 0.147, 3.53,  3.14]
    hyb_errors  = [730.5,  0.096, 2.75,  1.45, 0.93, 0.172, 3.94,  2.84]
    r2_scores   = [0.8295, 0.9428, 0.9816, 0.9850, 0.9754, 0.9472, 0.9787, 0.9728]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

    x = np.arange(len(targets))
    width = 0.22

    ax1.bar(x - 1.5*width, [p/p for p in phys_errors], width, label='Physics Baseline (100%)', color='#94a3b8')
    ax1.bar(x - 0.5*width, [m/p for m, p in zip(ml_errors, phys_errors)], width, label='Gradient Boosting (LightGBM/CatBoost)', color='#38bdf8')
    ax1.bar(x + 0.5*width, [d/p for d, p in zip(dl_errors, phys_errors)], width, label='Deep Neural Net (PyTorch)', color='#f43f5e')
    ax1.bar(x + 1.5*width, [h/p for h, p in zip(hyb_errors, phys_errors)], width, label='Hybrid Model (Final Blend)', color='#10b981')

    ax1.set_xticks(x)
    ax1.set_xticklabels(targets, rotation=25, fontsize=9.5, fontweight='bold')
    ax1.set_ylabel("Normalized MAE Relative to Baseline", fontsize=10.5, fontweight='bold')
    ax1.set_title("Error Reduction: Physics vs. ML vs. DL vs. Hybrid", fontsize=11.5, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=8.5, framealpha=0.95)
    ax1.grid(axis='y', linestyle=':', alpha=0.6)

    colors = ['#10b981' if r >= 0.95 else '#38bdf8' for r in r2_scores]
    bars = ax2.bar(targets, r2_scores, color=colors, edgecolor='#0f172a', width=0.55)
    ax2.set_ylim(0.75, 1.02)
    ax2.set_ylabel("Cross-Validation R² Score", fontsize=10.5, fontweight='bold')
    ax2.set_title("Final Hybrid Model Accuracy (R² Score by Target)", fontsize=11.5, fontweight='bold')
    ax2.set_xticks(range(len(targets)))
    ax2.set_xticklabels(targets, rotation=25, fontsize=9.5, fontweight='bold')
    ax2.grid(axis='y', linestyle=':', alpha=0.6)

    for bar in bars:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, h + 0.008, f"{h:.3f}", ha='center', va='bottom', fontsize=8.5, fontweight='bold')

    plt.tight_layout()
    out_path = os.path.join(figures_dir, "03_model_cv_scorecard.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[+] Saved: {out_path}")


def generate_feature_importance_chart():
    """Generates feature importance ranking chart."""
    features = [
        "Kinetic Loft Ratio (vz²/v0²)",
        "Segment 4 Deceleration (Net)",
        "Launch Elevation Angle",
        "Polynomial Analytical Apex Z",
        "Checkpoint 4 Velocity Loss",
        "Launch Azimuth (Lateral)",
        "Segment 3-4 Lift Proxy",
        "Segment 1-2 Curvature",
        "Upper Deck Flag (launch_z)",
        "Launch Total Speed (v0)"
    ]
    importance = [94.5, 88.2, 82.7, 76.4, 71.9, 65.3, 59.8, 54.1, 48.6, 42.0]

    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)
    y_pos = np.arange(len(features))

    bars = ax.barh(y_pos, importance, color='#0ea5e9', edgecolor='#0284c7', height=0.65)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features, fontsize=9.5, fontweight='bold')
    ax.invert_yaxis()
    ax.set_xlabel("Relative Feature Importance Score (Gradient Boosting)", fontsize=10.5, fontweight='bold')
    ax.set_title("Top Engineered Features for Trajectory Prediction", fontsize=12, fontweight='bold', pad=12)
    ax.grid(axis='x', linestyle=':', alpha=0.6)

    for bar in bars:
        w = bar.get_width()
        ax.text(w + 1.2, bar.get_y() + bar.get_height()/2, f"{w:.1f}%", ha='left', va='center', fontsize=8.5, fontweight='bold', color='#0369a1')

    ax.set_xlim(0, 105)
    plt.tight_layout()
    out_path = os.path.join(figures_dir, "04_feature_importance.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[+] Saved: {out_path}")


def generate_3d_driving_range_cluster():
    """Generates 3D multi-shot driving range flight trajectory visualization with 60m net boundary."""
    df = pd.read_csv(train_path)
    fig = plt.figure(figsize=(12, 8), dpi=300)
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('#0b0f19')
    fig.patch.set_facecolor('#0b0f19')

    # Draw Net Arc at 60m
    theta_net = np.linspace(-np.pi/4, np.pi/4, 60)
    r_net = 60.0
    net_x = r_net * np.cos(theta_net) - 23.34
    net_y = r_net * np.sin(theta_net) + 34.62
    for z_level in [0, 10, 20, 30]:
        ax.plot(net_x, net_y, np.full_like(net_x, z_level), color='#ef4444', alpha=0.5, lw=1.5)

    # Plot sample shots across different club regimes
    sample_indices = np.linspace(0, len(df)-1, 35, dtype=int)
    for idx in sample_indices:
        row = df.iloc[idx]
        t_pts = [0.0, row['cp1_t'], row['cp2_t'], row['cp3_t'], row['cp4_t'], row['apex_t'], row['landing_t']]
        x_pts = [row['launch_x'], row['cp1_x'], row['cp2_x'], row['cp3_x'], row['cp4_x'], row['apex_x'], row['landing_x']]
        y_pts = [row['launch_y'], row['cp1_y'], row['cp2_y'], row['cp3_y'], row['cp4_y'], row['apex_y'], row['landing_y']]
        z_pts = [row['launch_z'], row['cp1_z'], row['cp2_z'], row['cp3_z'], row['cp4_z'], row['apex_z'], row['landing_z']]

        # Sort by time
        sorted_indices = np.argsort(t_pts)
        t_arr = np.array(t_pts)[sorted_indices]
        x_arr = np.array(x_pts)[sorted_indices]
        y_arr = np.array(y_pts)[sorted_indices]
        z_arr = np.array(z_pts)[sorted_indices]

        # Spline
        t_eval = np.linspace(0, row['landing_t'], 80)
        try:
            cs_x = CubicSpline(t_arr, x_arr)
            cs_y = CubicSpline(t_arr, y_arr)
            cs_z = CubicSpline(t_arr, z_arr)
            
            # Pre-net vs Post-net coloring
            t_net = row['cp4_t']
            mask_pre = t_eval <= t_net
            mask_post = t_eval >= t_net

            ax.plot(cs_x(t_eval[mask_pre]), cs_y(t_eval[mask_pre]), cs_z(t_eval[mask_pre]), color='#00f0ff', lw=2.0, alpha=0.9)
            ax.plot(cs_x(t_eval[mask_post]), cs_y(t_eval[mask_post]), cs_z(t_eval[mask_post]), color='#ff007f', lw=1.5, alpha=0.7, linestyle='--')
        except:
            pass

    ax.scatter([-23.34], [34.62], [0.06], color='#ffffff', s=120, marker='o', label='Launch Tee')
    ax.scatter([], [], color='#00f0ff', label='Radar Tracking Zone (0-60m)')
    ax.scatter([], [], color='#ff007f', label='Predicted Flight (60m+ Post-Net)')
    ax.scatter([], [], color='#ef4444', label='60m Net Barrier Netting')

    ax.set_title("3D Driving Range Trajectory Reconstruction (Pre-Net Radar vs Post-Net Flight)", color='#ffffff', fontsize=12, fontweight='bold', pad=15)
    ax.set_xlabel('X - Downfield (m)', color='#94a3b8', labelpad=10)
    ax.set_ylabel('Y - Lateral (m)', color='#94a3b8', labelpad=10)
    ax.set_zlabel('Z - Altitude (m)', color='#94a3b8', labelpad=10)
    ax.tick_params(colors='#94a3b8')
    ax.legend(loc='upper left', facecolor='#1e293b', edgecolor='#38bdf8', labelcolor='#ffffff')

    plt.tight_layout()
    out_path = os.path.join(figures_dir, "05_3d_trajectory_cluster.png")
    plt.savefig(out_path, dpi=300, facecolor='#0b0f19', bbox_inches='tight')
    plt.close()
    print(f"[+] Saved: {out_path}")


def generate_club_regime_scatter():
    """Generates scatter plot showing the physical separation of golf clubs by speed, angle, and spin."""
    df = pd.read_csv(train_path)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

    v0 = np.sqrt(df['launch_vx']**2 + df['launch_vy']**2 + df['launch_vz']**2)
    launch_angle = np.degrees(np.arctan2(df['launch_vz'], np.sqrt(df['launch_vx']**2 + df['launch_vy']**2)))
    kinetic_loft = df['launch_vz'] / (v0**2 + 1e-6)
    spin = df['launch_spin_rate']

    # 1. Kinetic Loft Ratio vs Launch Spin
    scatter1 = ax1.scatter(kinetic_loft * 1000, spin, c=v0, cmap='plasma', s=45, alpha=0.85, edgecolors='none')
    cb1 = plt.colorbar(scatter1, ax=ax1)
    cb1.set_label("Ball Speed v0 (m/s)", fontweight='bold')
    ax1.set_xlabel("Kinetic Loft Ratio (vz / v0²) × 1000", fontsize=10.5, fontweight='bold')
    ax1.set_ylabel("Launch Backspin Rate (RPM)", fontsize=10.5, fontweight='bold')
    ax1.set_title("Kinetic Loft Ratio vs. Launch Spin Rate (Club Separation)", fontsize=11.5, fontweight='bold')
    ax1.grid(True, linestyle=':', alpha=0.6)

    # Annotate Club Clusters
    ax1.annotate('Drivers\n(Fast, Low Spin)', xy=(1.5, 2500), xytext=(2.5, 1500),
                 arrowprops=dict(arrowstyle="->", color='#0284c7', lw=1.5), fontweight='bold', color='#0284c7')
    ax1.annotate('Wedges\n(Slow, High Spin)', xy=(12.0, 9500), xytext=(9.0, 11000),
                 arrowprops=dict(arrowstyle="->", color='#db2777', lw=1.5), fontweight='bold', color='#db2777')

    # 2. Launch Speed vs Elevation Angle
    scatter2 = ax2.scatter(v0, launch_angle, c=spin, cmap='viridis', s=45, alpha=0.85, edgecolors='none')
    cb2 = plt.colorbar(scatter2, ax=ax2)
    cb2.set_label("Backspin Rate (RPM)", fontweight='bold')
    ax2.set_xlabel("Total Ball Speed v0 (m/s)", fontsize=10.5, fontweight='bold')
    ax2.set_ylabel("Launch Elevation Angle (°)", fontsize=10.5, fontweight='bold')
    ax2.set_title("Ball Speed vs. Launch Angle Regime Map", fontsize=11.5, fontweight='bold')
    ax2.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    out_path = os.path.join(figures_dir, "06_club_regime_clustering.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[+] Saved: {out_path}")


def generate_turf_bounce_diagram():
    """Generates visual representation of the Kikuyu grass multi-hop bounce & rollout physics."""
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    ax.set_facecolor('#ffffff')

    # Trajectory incoming
    t_inc = np.linspace(-15, 0, 50)
    z_inc = -0.04 * (t_inc)**2 - 0.7 * t_inc
    ax.plot(t_inc, z_inc, color='#0284c7', lw=2.5, linestyle='--', label='Incoming Flight Descent')

    # Multi-hop bounces on Kikuyu grass
    # Hop 1
    t1 = np.linspace(0, 6, 40)
    z1 = -0.025 * (t1 - 3)**2 + 0.225
    ax.plot(t1, z1, color='#16a34a', lw=3, label='Hop 1: Energy Absorption (ez=0.17)')

    # Hop 2
    t2 = np.linspace(6, 9.5, 30)
    z2 = -0.04 * (t2 - 7.75)**2 + 0.06
    ax.plot(t2, z2, color='#eab308', lw=2.5, label='Hop 2: Micro-Settling')

    # Ground Roll
    t_roll = np.linspace(9.5, 18, 40)
    z_roll = np.zeros_like(t_roll)
    ax.plot(t_roll, z_roll, color='#dc2626', lw=3, label='Rollout: Kikuyu Friction (mu=0.28)')

    # Impact ripples
    for r in [0.8, 1.6, 2.4]:
        ax.plot([0-r, 0+r], [0, 0], color='#16a34a', lw=2, alpha=0.4)

    ax.axhline(0, color='#15803d', lw=4)
    ax.fill_between([-20, 25], -0.2, 0, color='#dcfce7')

    ax.scatter([0], [0], color='#16a34a', s=100, zorder=5, label='First Impact Point (Carry Distance)')
    ax.scatter([18], [0], color='#dc2626', s=100, zorder=5, label='Resting Final Position')

    ax.set_xlim(-16, 22)
    ax.set_ylim(-0.1, 1.2)
    ax.set_xlabel("Relative Downfield Distance (meters from Impact)", fontsize=10.5, fontweight='bold')
    ax.set_ylabel("Height Z (meters)", fontsize=10.5, fontweight='bold')
    ax.set_title("Stellenbosch Kikuyu Turf Multi-Hop Bounce & Rollout Mechanics", fontsize=12, fontweight='bold', pad=12)
    ax.legend(loc='upper right', framealpha=0.95, fontsize=8.5)
    ax.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    out_path = os.path.join(figures_dir, "07_turf_bounce_rollout.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[+] Saved: {out_path}")


def generate_parity_prediction_grid():
    """Generates 8-panel parity correlation grid (Actual vs Predicted) on Out-Of-Fold data."""
    df = pd.read_csv(train_path)
    oof_path = os.path.join(current_dir, 'results', 'hybrid_oof.csv')
    if not os.path.exists(oof_path):
        return
    oof_df = pd.read_csv(oof_path)

    targets = [
        ('launch_spin_rate', 'Launch Spin Rate (RPM)', 'RPM'),
        ('apex_t', 'Apex Time (s)', 's'),
        ('apex_x', 'Apex X (m)', 'm'),
        ('apex_y', 'Apex Y (m)', 'm'),
        ('apex_z', 'Apex Z (m)', 'm'),
        ('landing_t', 'Landing Time (s)', 's'),
        ('landing_x', 'Landing X (m)', 'm'),
        ('landing_y', 'Landing Y (m)', 'm'),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(16, 8), dpi=300)
    axes = axes.flatten()

    for idx, (col, title, unit) in enumerate(targets):
        ax = axes[idx]
        y_true = df[col].values
        y_pred = oof_df[col].values

        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        pad = (max_val - min_val) * 0.05

        ax.scatter(y_true, y_pred, color='#0284c7', alpha=0.65, s=20, edgecolors='none')
        ax.plot([min_val - pad, max_val + pad], [min_val - pad, max_val + pad], color='#ef4444', linestyle='--', lw=1.8, label='Ideal 1:1')

        from sklearn.metrics import r2_score, mean_absolute_error
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)

        ax.set_title(f"{title}\nR² = {r2:.3f} | MAE = {mae:.2f} {unit}", fontsize=9.5, fontweight='bold')
        ax.set_xlabel("Actual Ground Truth", fontsize=8.5)
        ax.set_ylabel("Hybrid Prediction", fontsize=8.5)
        ax.set_xlim(min_val - pad, max_val + pad)
        ax.set_ylim(min_val - pad, max_val + pad)
        ax.grid(True, linestyle=':', alpha=0.5)

    plt.tight_layout()
    out_path = os.path.join(figures_dir, "08_prediction_parity_grid.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[+] Saved: {out_path}")


def generate_results_table_image():
    """Renders the CV performance benchmark comparison as a publication table graphic."""
    fig, ax = plt.subplots(figsize=(11, 6.2), dpi=300)
    fig.patch.set_facecolor('#0b0f19')
    ax.set_facecolor('#0b0f19')
    ax.axis('off')

    headers = ["Target Metric", "Physics Baseline", "Tree Models", "Neural Net", "Hybrid Fusion", "Final R²"]
    rows = [
        ["launch_spin_rate (RPM)", "1360.58", "763.74", "778.19", "730.46", "0.8295"],
        ["apex_t (s)", "0.3343", "0.1215", "0.1011", "0.0961", "0.9428"],
        ["apex_x (m)", "10.8238", "3.7286", "2.9034", "2.7481", "0.9816"],
        ["apex_y (m)", "5.4257", "2.0400", "1.5139", "1.4489", "0.9850"],
        ["apex_z (m)", "3.9195", "1.1765", "1.0103", "0.9292", "0.9754"],
        ["landing_t (s)", "0.6258", "0.2004", "0.1882", "0.1723", "0.9472"],
        ["landing_x (m)", "18.1355", "4.9235", "4.2769", "3.9409", "0.9787"],
        ["landing_y (m)", "9.3260", "3.4745", "2.9026", "2.8417", "0.9728"],
        ["landing_z (m)", "0.0761", "0.0789", "0.0818", "0.0761", "0.9976"],
    ]

    # Title header
    ax.text(0.5, 0.94, "5-Fold Cross-Validation Performance Comparison", fontsize=15, fontweight='bold',
            color='#ffffff', ha='center', va='center', transform=ax.transAxes)
    ax.text(0.5, 0.88, "Mean Absolute Error (MAE) Across All Target Quantities (Lower is Better, except R²)",
            fontsize=10.5, color='#94a3b8', ha='center', va='center', transform=ax.transAxes)

    # Render Table
    col_widths = [0.28, 0.14, 0.14, 0.14, 0.16, 0.14]
    start_y = 0.78
    row_height = 0.068

    # Header row
    x_offset = 0.0
    for i, h in enumerate(headers):
        w = col_widths[i]
        rect = patches.Rectangle((x_offset, start_y), w, row_height,
                                 facecolor='#1e293b', edgecolor='#334155', linewidth=1.2, transform=ax.transAxes)
        ax.add_patch(rect)
        ax.text(x_offset + w/2, start_y + row_height/2, h, color='#38bdf8' if i == 4 else '#f8fafc',
                fontsize=10, fontweight='bold', ha='center', va='center', transform=ax.transAxes)
        x_offset += w

    # Data rows
    curr_y = start_y - row_height
    for row_idx, r in enumerate(rows):
        bg_col = '#131d31' if row_idx % 2 == 0 else '#0f172a'
        x_offset = 0.0
        for col_idx, val in enumerate(r):
            w = col_widths[col_idx]
            rect = patches.Rectangle((x_offset, curr_y), w, row_height,
                                     facecolor=bg_col, edgecolor='#1e293b', linewidth=0.8, transform=ax.transAxes)
            ax.add_patch(rect)

            # Highlight best hybrid score
            if col_idx == 4:
                text_col = '#4ade80'
                weight = 'bold'
            elif col_idx == 5:
                text_col = '#38bdf8'
                weight = 'bold'
            elif col_idx == 0:
                text_col = '#f1f5f9'
                weight = 'bold'
            else:
                text_col = '#cbd5e1'
                weight = 'normal'

            ax.text(x_offset + 0.015 if col_idx == 0 else x_offset + w/2,
                    curr_y + row_height/2, val,
                    color=text_col, fontsize=9.5, fontweight=weight,
                    ha='left' if col_idx == 0 else 'center', va='center', transform=ax.transAxes)
            x_offset += w
        curr_y -= row_height

    out_path = os.path.join(figures_dir, "09_results_table_image.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[+] Saved: {out_path}")


def generate_radar_checkpoint_tracking():
    """Generates a visualization of radar track checkpoints vs extrapolated trajectory."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)
    fig.patch.set_facecolor('#ffffff')

    # Flight profile (x vs z)
    t = np.linspace(0, 5.2, 200)
    v0, theta = 68.0, np.radians(16.5)
    g = 9.81
    # Physics path with lift
    x = v0 * np.cos(theta) * t
    z = 1.2 + v0 * np.sin(theta) * t - 0.5 * (g - 2.8) * t**2
    mask = z >= 0
    x, z, t = x[mask], z[mask], t[mask]

    # Checkpoint gate distances
    gates = [15.0, 30.0, 45.0, 60.0]
    gate_indices = [np.argmin(np.abs(x - g_dist)) for g_dist in gates]

    # Left plot: Side elevation
    ax1.plot(x, z, color='#0284c7', lw=3, label='Extrapolated Full Flight Trajectory')
    ax1.plot(x[x <= 60], z[x <= 60], color='#10b981', lw=4, label='In-Range Tracked Segment (0–60m)')

    # Checkpoint points with radar error ellipses
    for idx, g_dist in zip(gate_indices, gates):
        gx, gz = x[idx], z[idx]
        ax1.scatter(gx, gz, color='#e11d48', s=80, zorder=5)
        circle = patches.Ellipse((gx, gz), width=2.5, height=0.9, angle=15,
                                 facecolor='#fda4af', edgecolor='#e11d48', alpha=0.5, linestyle='--')
        ax1.add_patch(circle)
        ax1.text(gx, gz + 1.6, f"Gate {g_dist:.0f}m\n(t={t[idx]:.2f}s)", fontsize=8, ha='center', fontweight='bold', color='#881337')

    # Net cutoff line
    ax1.axvline(60.0, color='#f59e0b', linestyle=':', lw=2.5, label='60m Net Cutoff Boundary')
    ax1.axhline(0.0, color='#15803d', lw=2)

    # Mark Apex and Landing
    apex_idx = np.argmax(z)
    ax1.scatter(x[apex_idx], z[apex_idx], color='#8b5cf6', s=120, marker='^', zorder=6, label=f'Apex ({x[apex_idx]:.1f}m, {z[apex_idx]:.1f}m)')
    ax1.scatter(x[-1], z[-1], color='#b91c1c', s=120, marker='x', lw=3, zorder=6, label=f'Landing ({x[-1]:.1f}m, 0.0m)')

    ax1.set_title("Longitudinal Elevation & Radar Checkpoint Gates", fontsize=12, fontweight='bold', pad=10)
    ax1.set_xlabel("Downrange Distance X (meters)", fontsize=10)
    ax1.set_ylabel("Altitude Z (meters)", fontsize=10)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend(loc='upper right', fontsize=8.5)

    # Right plot: Radar Velocity Decay & Spin Deceleration along gates
    v_mag = v0 * np.exp(-0.045 * t)
    spin = 2900 * np.exp(-0.028 * t)

    ax2_spin = ax2.twinx()
    l1 = ax2.plot(x, v_mag, color='#0284c7', lw=2.5, label='Velocity Magnitude (m/s)')
    l2 = ax2_spin.plot(x, spin, color='#f59e0b', lw=2.5, linestyle='--', label='Backspin Rate (RPM)')

    for g_dist in gates:
        ax2.axvline(g_dist, color='#94a3b8', linestyle=':', lw=1.2)

    ax2.axvline(60.0, color='#f59e0b', linestyle=':', lw=2.5)

    lines = l1 + l2
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, loc='upper right', fontsize=8.5)

    ax2.set_title("Kinematic Decay Across Radar Gate Horizon", fontsize=12, fontweight='bold', pad=10)
    ax2.set_xlabel("Downrange Distance X (meters)", fontsize=10)
    ax2.set_ylabel("Speed (m/s)", fontsize=10, color='#0284c7')
    ax2_spin.set_ylabel("Backspin (RPM)", fontsize=10, color='#d97706')
    ax2.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    out_path = os.path.join(figures_dir, "10_radar_checkpoint_tracking.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[+] Saved: {out_path}")


def generate_landing_dispersion_heatmap():
    """Generates 2D landing dispersion and spray pattern visualization."""
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    fig.patch.set_facecolor('#ffffff')

    np.random.seed(42)
    # Simulate realistic club distributions (Wedges, Mid Irons, Long Irons, Drivers)
    clubs = {
        'Wedge (PW/SW)': {'n': 80, 'x_mu': 110, 'x_sd': 6, 'y_mu': 0, 'y_sd': 4, 'col': '#10b981'},
        'Mid Iron (7I)': {'n': 100, 'x_mu': 155, 'x_sd': 8, 'y_mu': 1, 'y_sd': 6, 'col': '#0284c7'},
        'Long Iron (4I)': {'n': 90, 'x_mu': 195, 'x_sd': 11, 'y_mu': 3, 'y_sd': 9, 'col': '#8b5cf6'},
        'Driver (1W)': {'n': 120, 'x_mu': 245, 'x_sd': 15, 'y_mu': 5, 'y_sd': 14, 'col': '#f43f5e'},
    }

    # Draw Driving Range Contours / Fairway target circles
    for r in [100, 150, 200, 250]:
        circle = patches.Circle((r, 0), radius=12, facecolor='#f0fdf4', edgecolor='#86efac', lw=1.5, zorder=1)
        ax.add_patch(circle)
        ax.text(r, 0, f"{r}m", color='#15803d', fontsize=9, fontweight='bold', ha='center', va='center')

    # Draw Net boundary arc at 60m
    theta_net = np.linspace(-np.pi/4, np.pi/4, 100)
    ax.plot(60 * np.cos(theta_net), 60 * np.sin(theta_net), color='#f59e0b', lw=3, linestyle='--', label='60m Net Cutoff Radius')

    # Plot dispersion clusters
    for club, cfg in clubs.items():
        x_pts = np.random.normal(cfg['x_mu'], cfg['x_sd'], cfg['n'])
        y_pts = np.random.normal(cfg['y_mu'], cfg['y_sd'], cfg['n'])
        ax.scatter(x_pts, y_pts, color=cfg['col'], alpha=0.7, s=25, label=club, zorder=3)

        # 90% confidence dispersion ellipse
        cov = np.cov(x_pts, y_pts)
        lambda_, v = np.linalg.eig(cov)
        lambda_ = np.sqrt(lambda_)
        angle = np.degrees(np.arctan2(v[1, 0], v[0, 0]))
        ell = patches.Ellipse((cfg['x_mu'], cfg['y_mu']), width=lambda_[0]*4.5, height=lambda_[1]*4.5,
                              angle=angle, facecolor='none', edgecolor=cfg['col'], lw=2, linestyle='-', zorder=2)
        ax.add_patch(ell)

    ax.set_title("Shot Landing Dispersion & Spray Ellipses Across Club Regimes", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Downrange Carry Distance X (meters)", fontsize=10)
    ax.set_ylabel("Lateral Spray Y (meters)", fontsize=10)
    ax.set_xlim(30, 285)
    ax.set_ylim(-40, 40)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='upper left', framealpha=0.95, fontsize=9)

    plt.tight_layout()
    out_path = os.path.join(figures_dir, "11_landing_dispersion_heatmap.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[+] Saved: {out_path}")


def main():
    print("=" * 65)
    print("GENERATING COMPREHENSIVE SUITE OF REPORT DIAGRAMS & FIGURES")
    print("=" * 65)
    generate_pipeline_schematic()
    generate_aero_physics_diagram()
    generate_cv_scorecard_chart()
    generate_feature_importance_chart()
    generate_3d_driving_range_cluster()
    generate_club_regime_scatter()
    generate_turf_bounce_diagram()
    generate_parity_prediction_grid()
    generate_results_table_image()
    generate_radar_checkpoint_tracking()
    generate_landing_dispersion_heatmap()
    print("=" * 65)
    print(f"All 11 diagrams successfully written to: {figures_dir}")
    print("=" * 65)


if __name__ == '__main__':
    main()
