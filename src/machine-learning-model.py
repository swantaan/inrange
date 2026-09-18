"""
machine-learning-model.py

Machine learning pipeline for the Inrange Golf Competition.
Performs:
  - Physics-informed feature engineering (kinematics, Magnus lift proxy, drag deceleration,
    segment curvatures, polynomial trajectory extrapolation).
  - 5-Fold Cross-Validation across all 9 competition targets.
  - Multi-model ensemble fusing LightGBM, XGBoost, and CatBoost regressors.
  - Out-of-fold validation reporting and feature importance visualization.
  - Test set prediction export adhering to submission format.

Author: Inrange Competition Participant
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor


TARGET_COLUMNS = [
    'launch_spin_rate',
    'apex_t', 'apex_x', 'apex_y', 'apex_z',
    'landing_t', 'landing_x', 'landing_y', 'landing_z'
]


def extract_ml_features(df):
    """
    Extracts rich physics-informed kinematic and trajectory features from
    launch conditions and the 4 checkpoint observations.
    """
    feats = pd.DataFrame(index=df.index)

    # 1. Raw Launch Kinematics
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

    # Shot-Regime / Club Inference Prior: Regime = vz0 / vtot0^2
    # Delineates high-speed low-spin drivers from low-speed high-spin wedges
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


class MLTrajectoryModel:
    """
    Trains and blends LightGBM, XGBoost, and CatBoost models across 5 folds.
    """

    def __init__(self, n_splits=5, random_state=42):
        self.n_splits = n_splits
        self.random_state = random_state
        self.models = {} # target -> list of fold models
        self.oof_predictions = None
        self.feature_names = None

    def train_and_evaluate(self, X, y_df):
        self.feature_names = list(X.columns)
        self.oof_predictions = pd.DataFrame(index=y_df.index)

        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        scores = {}

        print("=" * 65, flush=True)
        print(f"TRAINING GRADIENT BOOSTING ENSEMBLE ({self.n_splits}-FOLD CV)", flush=True)
        print("=" * 65, flush=True)
        print(f"{'Target':<18} | {'MAE':<10} | {'RMSE':<10} | {'R2 Score':<10}", flush=True)
        print("-" * 65, flush=True)

        for target in TARGET_COLUMNS:
            y = y_df[target].values
            oof_pred = np.zeros(len(y))
            target_models = []

            for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
                X_tr, y_tr = X.iloc[train_idx], y[train_idx]
                X_va, y_va = X.iloc[val_idx], y[val_idx]

                # Model 1: LightGBM
                m_lgb = lgb.LGBMRegressor(
                    n_estimators=180,
                    learning_rate=0.04,
                    num_leaves=24,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    random_state=self.random_state + fold,
                    verbose=-1
                )
                m_lgb.fit(X_tr, y_tr)

                # Model 2: CatBoost
                m_cat = CatBoostRegressor(
                    iterations=200,
                    learning_rate=0.04,
                    depth=5,
                    random_seed=self.random_state + fold,
                    verbose=0
                )
                m_cat.fit(X_tr, y_tr)

                # Model 3: XGBoost
                m_xgb = xgb.XGBRegressor(
                    n_estimators=180,
                    learning_rate=0.04,
                    max_depth=4,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    random_state=self.random_state + fold,
                    verbosity=0
                )
                m_xgb.fit(X_tr, y_tr)

                pred_va = (
                    0.40 * m_lgb.predict(X_va) +
                    0.35 * m_cat.predict(X_va) +
                    0.25 * m_xgb.predict(X_va)
                )

                oof_pred[val_idx] = pred_va
                target_models.append((m_lgb, m_cat, m_xgb))

            self.models[target] = target_models
            self.oof_predictions[target] = oof_pred

            mae = mean_absolute_error(y, oof_pred)
            rmse = np.sqrt(mean_squared_error(y, oof_pred))
            r2 = r2_score(y, oof_pred)
            scores[target] = {'MAE': mae, 'RMSE': rmse, 'R2': r2}

            print(f"{target:<18} | {mae:<10.4f} | {rmse:<10.4f} | {r2:<10.4f}", flush=True)

        return scores

    def predict(self, X_test):
        preds = pd.DataFrame(index=X_test.index)

        for target in TARGET_COLUMNS:
            target_models = self.models[target]
            fold_preds = np.zeros(len(X_test))

            for m_lgb, m_cat, m_xgb in target_models:
                p_lgb = m_lgb.predict(X_test)
                p_cat = m_cat.predict(X_test)
                p_xgb = m_xgb.predict(X_test)
                fold_preds += (0.40 * p_lgb + 0.35 * p_cat + 0.25 * p_xgb)

            preds[target] = fold_preds / len(target_models)

        return preds

    def plot_feature_importance(self, save_path=None):
        """
        Plots the average feature importance for key targets.
        """
        if 'launch_spin_rate' not in self.models:
            return

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        sample_targets = ['launch_spin_rate', 'apex_z', 'landing_x']

        for ax_idx, tgt in enumerate(sample_targets):
            importances = np.zeros(len(self.feature_names))
            for m_lgb, _, _ in self.models[tgt]:
                importances += m_lgb.feature_importances_
            importances /= len(self.models[tgt])

            top_indices = np.argsort(importances)[-12:]
            top_feats = [self.feature_names[i] for i in top_indices]
            top_vals = importances[top_indices]

            axes[ax_idx].barh(range(len(top_vals)), top_vals, color='#00d2be', edgecolor='#008b8b')
            axes[ax_idx].set_yticks(range(len(top_vals)))
            axes[ax_idx].set_yticklabels(top_feats, fontsize=9)
            axes[ax_idx].set_title(f"Top Features: {tgt}", fontsize=12, fontweight='bold')
            axes[ax_idx].set_xlabel("Average Split Importance")
            axes[ax_idx].grid(axis='x', linestyle=':', alpha=0.6)

        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=200, bbox_inches='tight')
            print(f"Saved feature importance plot to {save_path}", flush=True)
        plt.close()


def run_ml_pipeline():
    print("=" * 65, flush=True)
    print("INRANGE COMPETITION: MACHINE LEARNING PIPELINE", flush=True)
    print("=" * 65, flush=True)

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

    print("\nExtracting physics-informed features from training data...", flush=True)
    t0 = time.time()
    X_train = extract_ml_features(train_df)
    y_train = train_df[TARGET_COLUMNS]
    print(f"Extracted {X_train.shape[1]} features in {time.time() - t0:.2f}s.", flush=True)

    # 1. Train and evaluate with 5-fold cross-validation
    ml_model = MLTrajectoryModel(n_splits=5, random_state=42)
    scores = ml_model.train_and_evaluate(X_train, y_train)

    # Save OOF predictions
    oof_df = pd.concat([train_df[['track_id']], ml_model.oof_predictions], axis=1)
    oof_path = os.path.join(results_dir, 'ml_oof.csv')
    oof_df.to_csv(oof_path, index=False)
    print(f"\nSaved Out-Of-Fold predictions to {oof_path}", flush=True)

    # 2. Predict on Test Set
    if os.path.exists(test_path):
        test_df = pd.read_csv(test_path)
        print(f"\nLoaded {len(test_df)} test shots. Extracting features...", flush=True)
        X_test = extract_ml_features(test_df)
        test_preds = ml_model.predict(X_test)

        test_sub = pd.concat([test_df[['track_id']], test_preds], axis=1)
        sub_path = os.path.join(results_dir, 'ml_predictions.csv')
        test_sub.to_csv(sub_path, index=False)
        print(f"Saved ML test predictions to {sub_path} (Shape: {test_sub.shape})", flush=True)

    print("\nMachine Learning pipeline completed successfully!", flush=True)


if __name__ == '__main__':
    run_ml_pipeline()
