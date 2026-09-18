"""
deep-learning-model.py

Neural network model for the Inrange Golf Competition.
It uses a multi-layer neural network to learn patterns in ball flight
and predict apex, landing, and spin rate using 5-fold cross-validation.
"""

import os
import sys
import time
import importlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Dynamic import of feature extraction from machine-learning-model.py
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

ml_module = importlib.import_module('machine-learning-model')
extract_ml_features = ml_module.extract_ml_features
TARGET_COLUMNS = ml_module.TARGET_COLUMNS


class DeepTrajectoryNetwork:
    """
    Neural network model to predict flight trajectory targets.
    """

    def __init__(self, hidden_layer_sizes=(256, 256, 128, 64), n_splits=5, random_state=42):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.n_splits = n_splits
        self.random_state = random_state
        self.models = []
        self.scalers_X = []
        self.scalers_y = []
        self.loss_curves = []
        self.oof_predictions = None

    def train_and_evaluate(self, X_df, y_df):
        X = X_df.values
        y = y_df.values
        self.oof_predictions = pd.DataFrame(np.zeros((len(X), len(TARGET_COLUMNS))), index=y_df.index, columns=TARGET_COLUMNS)

        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        scores = {}

        print("=" * 65, flush=True)
        print(f"TRAINING DEEP NEURAL NETWORK ARCHITECTURE ({self.n_splits}-FOLD CV)", flush=True)
        print("=" * 65, flush=True)
        print(f"Architecture: Input({X.shape[1]}) -> " + " -> ".join([str(h) for h in self.hidden_layer_sizes]) + f" -> Output({len(TARGET_COLUMNS)})", flush=True)
        print("-" * 65, flush=True)

        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            t_fold_start = time.time()
            X_tr, y_tr = X[train_idx], y[train_idx]
            X_va, y_va = X[val_idx], y[val_idx]

            # Fit independent scalers per fold to avoid leakage
            s_X = StandardScaler()
            s_y = StandardScaler()

            X_tr_scaled = s_X.fit_transform(X_tr)
            y_tr_scaled = s_y.fit_transform(y_tr)
            X_va_scaled = s_X.transform(X_va)

            mlp = MLPRegressor(
                hidden_layer_sizes=self.hidden_layer_sizes,
                activation='relu',
                solver='adam',
                alpha=1e-4,
                batch_size=32,
                learning_rate='adaptive',
                learning_rate_init=0.002,
                max_iter=300,
                early_stopping=True,
                validation_fraction=0.12,
                n_iter_no_change=25,
                random_state=self.random_state + fold
            )

            mlp.fit(X_tr_scaled, y_tr_scaled)

            # Predict on validation fold
            val_pred_scaled = mlp.predict(X_va_scaled)
            val_pred = s_y.inverse_transform(val_pred_scaled)

            self.oof_predictions.iloc[val_idx] = val_pred
            self.models.append(mlp)
            self.scalers_X.append(s_X)
            self.scalers_y.append(s_y)
            self.loss_curves.append(mlp.loss_curve_)

            fold_time = time.time() - t_fold_start
            print(f"Fold {fold + 1}/{self.n_splits} completed in {fold_time:.2f}s (Iterations: {mlp.n_iter_}, Final Loss: {mlp.loss_:.5f})", flush=True)

        print("\n" + "-" * 65, flush=True)
        print("DEEP LEARNING OUT-OF-FOLD (OOF) EVALUATION METRICS", flush=True)
        print("-" * 65, flush=True)
        print(f"{'Target':<18} | {'MAE':<10} | {'RMSE':<10} | {'R2 Score':<10}", flush=True)
        print("-" * 65, flush=True)

        for target in TARGET_COLUMNS:
            y_true = y_df[target].values
            y_pred = self.oof_predictions[target].values

            mae = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            r2 = r2_score(y_true, y_pred)
            scores[target] = {'MAE': mae, 'RMSE': rmse, 'R2': r2}

            print(f"{target:<18} | {mae:<10.4f} | {rmse:<10.4f} | {r2:<10.4f}", flush=True)

        return scores

    def predict(self, X_df):
        X = X_df.values
        accumulated_preds = np.zeros((len(X), len(TARGET_COLUMNS)))

        for mlp, s_X, s_y in zip(self.models, self.scalers_X, self.scalers_y):
            X_scaled = s_X.transform(X)
            pred_scaled = mlp.predict(X_scaled)
            pred = s_y.inverse_transform(pred_scaled)
            accumulated_preds += pred

        avg_preds = accumulated_preds / len(self.models)
        return pd.DataFrame(avg_preds, index=X_df.index, columns=TARGET_COLUMNS)

    def plot_training_loss(self, save_path=None):
        """
        Plots the training loss curve across all 5 folds.
        """
        plt.figure(figsize=(10, 6))
        for fold_idx, curve in enumerate(self.loss_curves):
            plt.plot(curve, label=f'Fold {fold_idx + 1}', lw=2, alpha=0.85)

        plt.title("Deep Neural Network Training Loss Convergence", fontsize=14, fontweight='bold')
        plt.xlabel("Iteration / Epoch", fontsize=11)
        plt.ylabel("Loss (MSE on Standardized Targets)", fontsize=11)
        plt.yscale('log')
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.legend(loc='upper right', frameon=True)
        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=200, bbox_inches='tight')
            print(f"Saved DL training loss curve to {save_path}", flush=True)
        plt.close()


def run_dl_pipeline():
    print("=" * 65, flush=True)
    print("INRANGE COMPETITION: DEEP LEARNING MODEL PIPELINE", flush=True)
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

    print("\nExtracting features for deep learning...", flush=True)
    t0 = time.time()
    X_train = extract_ml_features(train_df)
    y_train = train_df[TARGET_COLUMNS]
    print(f"Extracted {X_train.shape[1]} features in {time.time() - t0:.2f}s.", flush=True)

    # 1. Train Deep Neural Network with 5-fold CV
    dl_model = DeepTrajectoryNetwork(hidden_layer_sizes=(256, 256, 128, 64), n_splits=5, random_state=42)
    scores = dl_model.train_and_evaluate(X_train, y_train)

    # Save OOF predictions
    oof_df = pd.concat([train_df[['track_id']], dl_model.oof_predictions], axis=1)
    oof_path = os.path.join(results_dir, 'dl_oof.csv')
    oof_df.to_csv(oof_path, index=False)
    print(f"\nSaved Deep Learning OOF predictions to {oof_path}", flush=True)

    # 2. Predict on Test Data
    if os.path.exists(test_path):
        test_df = pd.read_csv(test_path)
        print(f"\nLoaded {len(test_df)} test shots. Generating DL predictions...", flush=True)
        X_test = extract_ml_features(test_df)
        test_preds = dl_model.predict(X_test)

        test_sub = pd.concat([test_df[['track_id']], test_preds], axis=1)
        sub_path = os.path.join(results_dir, 'dl_predictions.csv')
        test_sub.to_csv(sub_path, index=False)
        print(f"Saved DL test predictions to {sub_path} (Shape: {test_sub.shape})", flush=True)

    print("\nDeep Learning pipeline completed successfully!", flush=True)


if __name__ == '__main__':
    run_dl_pipeline()
