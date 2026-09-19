"""
src/hybrid.py

Main pipeline orchestrator for the Inrange Golf Competition.
Combines:
  1. Flight physics model for Stellenbosch conditions (`src/physics.py`).
  2. Tree-based machine learning models (`src/models.py`).
  3. Deep learning neural network (`src/models.py`).
  4. Best weighting to combine all models.
  5. Interactive 3D visualizers (`src/trajectory_visualizer.py` and `src/build_3d_simulator.py`).
"""

import os
import sys
import time
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from physics import GolfBallPhysicsSimulator, RHO_STELLENBOSCH, PRESSURE_STELLENBOSCH
from features import extract_ml_features, TARGET_COLUMNS
from models import MLTrajectoryModel, DeepTrajectoryNetwork
from trajectory_visualizer import TrajectoryPlotlyVisualizer


class HybridTrajectoryPipeline:
    """
    Main pipeline combining physics baseline, gradient boosting residuals, and deep neural network residuals.
    """

    def __init__(self, random_state=42):
        self.random_state = random_state
        self.simulator = GolfBallPhysicsSimulator()
        self.ml_model = MLTrajectoryModel(n_splits=5, random_state=random_state)
        self.dl_model = DeepTrajectoryNetwork(hidden_layer_sizes=(256, 256, 128, 64), n_splits=5, random_state=random_state)
        self.plotly_visualizer = TrajectoryPlotlyVisualizer(self.simulator)
        self.optimal_weights = {}  # target -> (w_ml, w_dl)
        self.oof_predictions = None

    def fit_and_evaluate(self, train_df: pd.DataFrame):
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

    def predict(self, test_df: pd.DataFrame) -> pd.DataFrame:
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

    def create_interactive_trajectory_animation(self, sample_row, output_html_path: str, df: pd.DataFrame = None, is_prediction: bool = True):
        self.plotly_visualizer.create_interactive_trajectory_animation(sample_row, output_html_path, df=df, is_prediction=is_prediction)


def run_hybrid_pipeline(shot_choice=None):
    print("=" * 70, flush=True)
    print("INRANGE COMPETITION: HYBRID PIPELINE EXECUTION", flush=True)
    print("=" * 70, flush=True)

    if shot_choice is None:
        for i, arg in enumerate(sys.argv):
            if arg in ('--shot', '-s') and i + 1 < len(sys.argv):
                shot_choice = sys.argv[i + 1]
                break
            elif i > 0 and (arg.isdigit() or len(arg) > 8) and not arg.startswith('-'):
                shot_choice = arg
                break
        if shot_choice is None:
            shot_choice = 0

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
