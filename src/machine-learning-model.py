"""
src/machine-learning-model.py

Machine learning baseline entrypoint (backwards-compatible wrapper around `src/models.py` and `src/features.py`).
"""

import os
import sys
import time
import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from features import extract_ml_features, TARGET_COLUMNS
from models import MLTrajectoryModel


def run_ml_pipeline():
    print("=" * 65, flush=True)
    print("INRANGE COMPETITION: MACHINE LEARNING PIPELINE", flush=True)
    print("=" * 65, flush=True)

    project_root = os.path.abspath(os.path.join(current_dir, '..')) if os.path.basename(current_dir) == 'src' else current_dir

    train_path = os.path.join(project_root, 'data', 'train.csv')
    test_path = os.path.join(project_root, 'data', 'test.csv')
    results_dir = os.path.join(project_root, 'results')
    os.makedirs(results_dir, exist_ok=True)

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
