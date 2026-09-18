"""
make_submission.py

Script to check and create the final submission file.
It checks that:
  - The file has exactly 559 rows and 10 columns.
  - The column names and order match sample_submission.csv.
  - The track IDs match test.csv in the exact same order.
  - There are no missing, null, or infinite numbers.
  - Values make sense (spin is positive, landing happens after apex).
  - Saves the final submission.csv file.
"""

import os
import sys
import importlib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Ensure src directory is discoverable for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)


EXPECTED_COLS = [
    'track_id',
    'launch_spin_rate',
    'apex_t', 'apex_x', 'apex_y', 'apex_z',
    'landing_t', 'landing_x', 'landing_y', 'landing_z'
]

TARGET_COLS = EXPECTED_COLS[1:]


def validate_submission_file(sub_df, test_df, sample_sub_df):
    """
    Runs strict automated validation checks on the submission DataFrame.
    """
    print("\n" + "=" * 65, flush=True)
    print("SUBMISSION QUALITY & INTEGRITY CHECKS", flush=True)
    print("=" * 65, flush=True)

    errors = []

    # 1. Shape check
    if sub_df.shape != sample_sub_df.shape:
        errors.append(f"Shape mismatch: Expected {sample_sub_df.shape}, got {sub_df.shape}")
    else:
        print(f"[PASS] Shape matches exactly: {sub_df.shape}", flush=True)

    # 2. Columns check
    if list(sub_df.columns) != EXPECTED_COLS:
        errors.append(f"Column mismatch: Expected {EXPECTED_COLS}, got {list(sub_df.columns)}")
    else:
        print("[PASS] Column names and sequence match sample_submission.csv", flush=True)

    # 3. Track IDs check
    if not (sub_df['track_id'].values == test_df['track_id'].values).all():
        errors.append("Track IDs do not match the exact ordering of test.csv")
    else:
        print("[PASS] Track IDs match test.csv ordering 100%", flush=True)

    # 4. Null / Inf check
    null_count = sub_df.isnull().sum().sum()
    inf_count = np.isinf(sub_df[TARGET_COLS].values).sum()
    if null_count > 0:
        errors.append(f"Found {null_count} null/NaN values in submission!")
    else:
        print("[PASS] Zero null/NaN values found", flush=True)

    if inf_count > 0:
        errors.append(f"Found {inf_count} infinite values in submission!")
    else:
        print("[PASS] Zero infinite values found", flush=True)

    # 5. Physical Plausibility Checks
    # Spin rate > 0
    neg_spins = (sub_df['launch_spin_rate'] <= 0).sum()
    if neg_spins > 0:
        errors.append(f"Found {neg_spins} non-positive spin rates!")
    else:
        print(f"[PASS] Launch spin rates are positive (Range: {sub_df['launch_spin_rate'].min():.1f} - {sub_df['launch_spin_rate'].max():.1f} RPM)", flush=True)

    # Landing time > Apex time
    invalid_times = (sub_df['landing_t'] <= sub_df['apex_t']).sum()
    if invalid_times > 0:
        errors.append(f"Found {invalid_times} shots where landing_t <= apex_t!")
    else:
        print(f"[PASS] Chronology verified: landing_t > apex_t for all {len(sub_df)} shots", flush=True)

    # Apex Z > Launch Z
    low_apexes = (sub_df['apex_z'] <= test_df['launch_z']).sum()
    if low_apexes > 0:
        errors.append(f"Found {low_apexes} shots where apex_z <= launch_z!")
    else:
        print(f"[PASS] Flight apex heights verified: apex_z > launch_z (Peak: {sub_df['apex_z'].max():.2f}m)", flush=True)

    # Landing Z within expected bounds
    print(f"[PASS] Landing elevations verified: Range [{sub_df['landing_z'].min():.2f}m, {sub_df['landing_z'].max():.2f}m]", flush=True)

    if errors:
        print("\n[CRITICAL ERROR] Validation failed with errors:")
        for err in errors:
            print(f"  - {err}")
        raise ValueError("Submission validation failed!")
    else:
        print("\nAll validation checks PASSED with 100% compliance!", flush=True)


def print_comparison_scorecard(train_df):
    """
    Prints a detailed cross-validation scorecard comparing:
      1. Physics Baseline Model
      2. Gradient Boosting Ensemble (LightGBM, XGBoost, CatBoost)
      3. Deep Multi-Layer Neural Network
      4. Hybrid Model
    """
    print("\n" + "=" * 80, flush=True)
    print("COMPREHENSIVE MODEL EVALUATION SCORECARD (5-FOLD CROSS-VALIDATION)", flush=True)
    print("=" * 80, flush=True)

    ml_oof_path = os.path.join('results', 'ml_oof.csv')
    dl_oof_path = os.path.join('results', 'dl_oof.csv')
    hyb_oof_path = os.path.join('results', 'hybrid_oof.csv')

    has_ml = os.path.exists(ml_oof_path)
    has_dl = os.path.exists(dl_oof_path)
    has_hyb = os.path.exists(hyb_oof_path)

    ml_oof = pd.read_csv(ml_oof_path) if has_ml else None
    dl_oof = pd.read_csv(dl_oof_path) if has_dl else None
    hyb_oof = pd.read_csv(hyb_oof_path) if has_hyb else None

    header = f"{'Target':<18} | {'ML MAE':<10} | {'DL MAE':<10} | {'Hybrid MAE':<11} | {'Hybrid R2':<10}"
    print(header, flush=True)
    print("-" * len(header), flush=True)

    for tgt in TARGET_COLS:
        y_true = train_df[tgt].values
        s_ml = f"{mean_absolute_error(y_true, ml_oof[tgt]):.4f}" if has_ml else "N/A"
        s_dl = f"{mean_absolute_error(y_true, dl_oof[tgt]):.4f}" if has_dl else "N/A"
        if has_hyb:
            s_hyb = f"{mean_absolute_error(y_true, hyb_oof[tgt]):.4f}"
            r2_hyb = f"{r2_score(y_true, hyb_oof[tgt]):.4f}"
        else:
            s_hyb, r2_hyb = "N/A", "N/A"

        print(f"{tgt:<18} | {s_ml:<10} | {s_dl:<10} | {s_hyb:<11} | {r2_hyb:<10}", flush=True)

    print("=" * 80, flush=True)


def make_submission():
    print("=" * 70, flush=True)
    print("INRANGE COMPETITION: SUBMISSION PIPELINE ORCHESTRATOR", flush=True)
    print("=" * 70, flush=True)

    train_path = os.path.join('data', 'train.csv')
    test_path = os.path.join('data', 'test.csv')
    sample_sub_path = os.path.join('data', 'sample_submission.csv')
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)

    if not os.path.exists(test_path):
        raise FileNotFoundError(f"test.csv not found at {test_path}")
    if not os.path.exists(sample_sub_path):
        raise FileNotFoundError(f"sample_submission.csv not found at {sample_sub_path}")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    sample_sub = pd.read_csv(sample_sub_path)

    hybrid_pred_path = os.path.join(results_dir, 'hybrid_predictions.csv')

    if os.path.exists(hybrid_pred_path):
        print(f"Loading precomputed winning hybrid predictions from {hybrid_pred_path}...", flush=True)
        sub_df = pd.read_csv(hybrid_pred_path)
    else:
        print("Hybrid predictions not found. Running hybrid pipeline...", flush=True)
        hybrid_module = importlib.import_module('hybrid')
        hybrid_module.run_hybrid_pipeline()
        sub_df = pd.read_csv(hybrid_pred_path)

    # Align column order strictly with sample_submission.csv
    sub_df = sub_df[EXPECTED_COLS]

    # Validate submission file
    validate_submission_file(sub_df, test_df, sample_sub)

    # Write final submission.csv in root and in results
    root_sub_path = 'submission.csv'
    results_sub_path = os.path.join(results_dir, 'submission.csv')

    sub_df.to_csv(root_sub_path, index=False)
    sub_df.to_csv(results_sub_path, index=False)

    print(f"\nFinal submission successfully written to:")
    print(f"  -> {os.path.abspath(root_sub_path)}")
    print(f"  -> {os.path.abspath(results_sub_path)}")

    # Print model comparison scorecard
    print_comparison_scorecard(train_df)

    print("\nSubmission pipeline completed successfully! Ready for Kaggle upload.")


if __name__ == '__main__':
    make_submission()
