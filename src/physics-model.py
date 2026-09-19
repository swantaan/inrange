"""
src/physics-model.py

Physics baseline entrypoint (backwards-compatible wrapper around `src/physics.py`).
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

from physics import (
    ELEVATION_STELLENBOSCH,
    TEMP_STELLENBOSCH_C,
    TEMP_STELLENBOSCH_K,
    P_SEA_LEVEL,
    R_SPECIFIC_AIR,
    G_CONST,
    PRESSURE_STELLENBOSCH,
    RHO_STELLENBOSCH,
    MASS,
    RADIUS,
    DIAMETER,
    AREA,
    AERO_FACTOR,
    DYNAMIC_VISCOSITY_AIR,
    SPIN_DECAY_TAU,
    K_INDUCED_DRAG,
    InstantaneousAeroNeuralNet,
    GolfBallPhysicsSimulator
)


def run_physics_pipeline():
    print("=" * 65, flush=True)
    print("INRANGE COMPETITION: QUINTAVALLA DIMPLED PHYSICS PIPELINE", flush=True)
    print("=" * 65, flush=True)
    print(f"Atmospheric parameters (Stellenbosch, 136m):", flush=True)
    print(f"  * Barometric Pressure : {PRESSURE_STELLENBOSCH:.1f} Pa ({PRESSURE_STELLENBOSCH/100:.2f} hPa)", flush=True)
    print(f"  * Air Density (rho)   : {RHO_STELLENBOSCH:.4f} kg/m^3", flush=True)
    print(f"  * Quadratic Induced Drag: Cd = Cd0(v) + {K_INDUCED_DRAG:.2f} * CL^2", flush=True)
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

    simulator = GolfBallPhysicsSimulator()

    t0 = time.time()
    train_preds = simulator.predict_dataset(train_df)
    eval_time = time.time() - t0
    print(f"Physics simulation completed in {eval_time:.2f}s ({eval_time/len(train_df)*1000:.2f} ms/shot).", flush=True)

    targets = ['launch_spin_rate', 'apex_t', 'apex_x', 'apex_y', 'apex_z', 'landing_t', 'landing_x', 'landing_y', 'landing_z']

    print("\n" + "-" * 65, flush=True)
    print("TRAINING SET EVALUATION METRICS (QUINTAVALLA INDUCED DRAG)", flush=True)
    print("-" * 65, flush=True)
    print(f"{'Target':<18} | {'MAE':<10} | {'RMSE':<10} | {'R2 Score':<10}", flush=True)
    print("-" * 65, flush=True)

    for target in targets:
        mae = mean_absolute_error(train_df[target], train_preds[target])
        rmse = np.sqrt(mean_squared_error(train_df[target], train_preds[target]))
        r2 = r2_score(train_df[target], train_preds[target])
        print(f"{target:<18} | {mae:<10.4f} | {rmse:<10.4f} | {r2:<10.4f}", flush=True)

    if os.path.exists(test_path):
        test_df = pd.read_csv(test_path)
        print(f"\nLoaded {len(test_df)} test shots.", flush=True)
        test_preds = simulator.predict_dataset(test_df)

        submission_cols = ['track_id'] + targets
        physics_sub = test_preds[submission_cols]
        out_csv = os.path.join(results_dir, 'physics_predictions.csv')
        physics_sub.to_csv(out_csv, index=False)
        print(f"Saved physics test predictions to {out_csv} (Shape: {physics_sub.shape})", flush=True)

    print("\nPhysics model pipeline completed successfully!", flush=True)


if __name__ == '__main__':
    run_physics_pipeline()
