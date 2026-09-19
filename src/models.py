"""
src/models.py

Model training architectures for the Inrange Golf Trajectory competition:
1. MLTrajectoryModel: 5-fold cross-validated ensemble of LightGBM, CatBoost, and XGBoost.
2. DeepTrajectoryNetwork: 5-fold cross-validated deep neural network with standardized I/O.
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

from features import TARGET_COLUMNS


class MLTrajectoryModel:
    """
    Gradient boosting ensemble (LightGBM + CatBoost + XGBoost) trained with 5-fold cross-validation.
    """

    def __init__(self, n_splits=5, random_state=42):
        self.n_splits = n_splits
        self.random_state = random_state
        self.models = {}  # target -> list of fold models
        self.oof_predictions = None
        self.feature_names = None

    def train_and_evaluate(self, X: pd.DataFrame, y_df: pd.DataFrame):
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

    def predict(self, X_test: pd.DataFrame) -> pd.DataFrame:
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


class DeepTrajectoryNetwork:
    """
    Multi-layer neural network architecture trained with 5-fold cross-validation.
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

    def train_and_evaluate(self, X_df: pd.DataFrame, y_df: pd.DataFrame):
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

    def predict(self, X_df: pd.DataFrame) -> pd.DataFrame:
        X = X_df.values
        accumulated_preds = np.zeros((len(X), len(TARGET_COLUMNS)))

        for mlp, s_X, s_y in zip(self.models, self.scalers_X, self.scalers_y):
            X_scaled = s_X.transform(X)
            pred_scaled = mlp.predict(X_scaled)
            pred = s_y.inverse_transform(pred_scaled)
            accumulated_preds += pred

        avg_preds = accumulated_preds / len(self.models)
        return pd.DataFrame(avg_preds, index=X_df.index, columns=TARGET_COLUMNS)
