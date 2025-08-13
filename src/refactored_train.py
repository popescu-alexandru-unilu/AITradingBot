# src/refactored_train.py

import warnings
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_score,
    recall_score, f1_score, confusion_matrix, make_scorer
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier

# --- Suppress Warnings ---
warnings.filterwarnings("ignore", message=".*gpu_hist.*")
warnings.filterwarnings("ignore", message=".*use_label_encoder.*")

# --- Configuration ---
RAW_DATA_FILE = "data/raw/minute_bars.csv"
LABELS_FILE = "data/refactored_swing_labeled_dataset.csv"
MODEL_OUTPUT_FILE = "models/refactored_swing_model.joblib"
RESAMPLE_FREQ = "1h"  # Use lowercase 'h'
FEATURE_WARMUP_PERIOD = 200  # Bars to drop for indicator stabilization

# --- Model & CV Parameters ---
N_SPLITS_OUTER = 6
N_SPLITS_INNER = 4
EMBARGO_HOURS = 24
VALIDATION_SET_SIZE = 0.2  # 20% of train data for validation tail
RANDOM_SEARCH_ITER = 30

# --- Backtest Parameters ---
FEES_BPS = 5  # 5 bps for round-trip cost

# --- Debugging & Feature Engineering Functions ---
def dbg(df, msg):
    """Prints debug info for a DataFrame."""
    print(f"{msg}: {df.shape[0]} rows, {df.shape[1]} cols, "
          f"index[{df.index.min()} -> {df.index.max()}]")

def add_features(df):
    """Adds technical indicators to the DataFrame."""
    df["MA50"] = df["close"].rolling(50).mean()
    df["MA200"] = df["close"].rolling(200).mean()
    
    delta = df["close"].diff()
    up = delta.clip(lower=0).rolling(14).mean()
    dn = -delta.clip(upper=0).rolling(14).mean()
    df["RSI14"] = 100 - 100 / (1 + up / dn)
    
    # Session VWAP (resets daily at UTC midnight) - REFACTORED
    session = df.index.floor("D")
    pv_cum = (df["close"] * df["volume"]).groupby(session).transform("cumsum")
    v_cum  = df["volume"].groupby(session).transform("cumsum")
    df["VWAP"] = pv_cum / v_cum

    df["swing_high20"] = df["high"].rolling(20).max().shift(1)
    df["swing_low20"] = df["low"].rolling(20).min().shift(1)
    df["vol_spike"] = df["volume"] / df["volume"].rolling(20).mean()
    return df

# --- Main Execution ---
def main():
    # 1. Load and Prepare Data
    df_min = pd.read_csv(
        RAW_DATA_FILE, parse_dates=["timestamp"], index_col="timestamp",
        usecols=["timestamp", "open", "high", "low", "close", "volume"]
    ).sort_index()
    
    # Ensure timezone-aware index
    if df_min.index.tz is None:
        df_min.index = df_min.index.tz_localize("UTC")
    dbg(df_min, "Loaded and sorted minute data")

    ohlc = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    df1h = df_min.resample(RESAMPLE_FREQ).agg(ohlc)
    dbg(df1h, "After 1h resample")

    # 2. Feature Engineering
    df1h = add_features(df1h)
    dbg(df1h, "After features (before dropna)")
    
    # 3. Load and Align Labels
    labels_df = pd.read_csv(LABELS_FILE, parse_dates=["timestamp"], index_col="timestamp")
    if labels_df.index.tz is None:
        labels_df.index = labels_df.index.tz_localize("UTC")
    labels_df = labels_df.asfreq(RESAMPLE_FREQ) # Snap to the same hourly grid
    
    data = df1h.join(labels_df["label"], how="inner")
    dbg(data, "After join with labels")

    if data.empty:
        raise ValueError("No rows after joining features and labels. Check for index mismatch (timezone, frequency).")

    # Shift label by 1 to prevent lookahead bias (trade on next bar's close)
    data["label"] = data["label"].shift(-1)
    
    # Final cleanup: warmup period and dropna
    data = data.iloc[FEATURE_WARMUP_PERIOD:]
    data.dropna(inplace=True)
    data = data[data["label"] != 0]
    dbg(data, "Final data after cleanup")

    if data.empty:
        raise ValueError("No rows remaining after cleanup. Check warmup period or data quality.")

    feature_cols = [c for c in data.columns if c not in ["label", "open", "high", "low", "close", "volume"]]
    X = data[feature_cols]
    y = data["label"].map({-1: 0, 1: 1}) # Binary target: 1=TP, 0=SL

    # --- Guardrails ---
    if y.nunique() < 2:
        raise ValueError(f"Target variable has less than 2 unique classes: {y.unique()}")
    
    pos = (y == 1).sum()
    neg = (y == 0).sum()
    if pos == 0 or neg == 0:
        raise ValueError(f"One class is missing in the target. Positives: {pos}, Negatives: {neg}")

    print(f"Training on {len(X)} samples.")
    print(f"Class distribution:\n{y.value_counts(normalize=True)}")

    # 4. Time-Aware CV & Modeling Setup
    scale_pos_weight = neg / pos
    
    # Make n_splits dynamic to avoid errors on small data
    n_splits_outer = min(N_SPLITS_OUTER, len(X) - 1)
    n_splits_inner = min(N_SPLITS_INNER, len(X) - 1)
    
    if n_splits_outer < 2 or n_splits_inner < 2:
        raise ValueError("Not enough data to perform time-series cross-validation.")

    tscv_inner = TimeSeriesSplit(n_splits=n_splits_inner)
    
    # Base models
    xgb = XGBClassifier(eval_metric="logloss", scale_pos_weight=scale_pos_weight, random_state=42, use_label_encoder=False)
    lgbm = LGBMClassifier(random_state=42)
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)

    # Stacking classifier with time-aware inner CV
    stack = StackingClassifier(
        estimators=[('xgb', xgb), ('lgbm', lgbm), ('rf', rf)],
        final_estimator=Pipeline([
            ('scaler', StandardScaler()),
            ('lr', LogisticRegression(class_weight='balanced'))
        ]),
        cv=tscv_inner,
        passthrough=True # Pass original features to final_estimator
    )

    # Hyperparameter search grid
    param_dist = {
        'xgb__n_estimators': [100, 300], 'xgb__learning_rate': [0.05, 0.1], 'xgb__max_depth': [3, 5],
        'lgbm__n_estimators': [100, 300], 'lgbm__learning_rate': [0.05, 0.1], 'lgbm__num_leaves': [20, 40],
        'rf__n_estimators': [100, 300], 'rf__max_depth': [10, 20],
        'final_estimator__lr__C': [0.1, 1, 10]
    }

    # 5. Walk-Forward Validation with Leakage Controls
    outer_cv = TimeSeriesSplit(n_splits=n_splits_outer)
    
    all_metrics = []
    all_probas = []
    all_ys = []

    for i, (train_idx, test_idx) in enumerate(outer_cv.split(X)):
        # Apply embargo by removing a block of data from the end of the training set
        train_idx_embargo = train_idx[:-EMBARGO_HOURS]
        
        if len(train_idx_embargo) == 0:
            print(f"Skipping fold {i+1} due to insufficient data after embargo.")
            continue

        # Split the embargo-adjusted train set into core and validation tail
        val_len = int(len(train_idx_embargo) * VALIDATION_SET_SIZE)
        if val_len == 0:
            print(f"Skipping fold {i+1} due to insufficient data for validation set.")
            continue
            
        train_core_idx = train_idx_embargo[:-val_len]
        val_idx = train_idx_embargo[-val_len:]

        X_train_core, y_train_core = X.iloc[train_core_idx], y.iloc[train_core_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

        print(f"\n--- Outer Fold {i+1}/{N_SPLITS_OUTER} ---")
        print(f"Train: {len(X_train_core)}, Val: {len(X_val)}, Test: {len(X_test)}")

        # a) Tune model on the core training data
        search = RandomizedSearchCV(
            estimator=stack, param_distributions=param_dist, n_iter=RANDOM_SEARCH_ITER,
            cv=tscv_inner, scoring='average_precision', n_jobs=-1, random_state=42, verbose=0
        )
        search.fit(X_train_core, y_train_core)
        
        # b) Calibrate on the validation tail
        calibrated_model = CalibratedClassifierCV(search.best_estimator_, method='isotonic', cv='prefit')
        calibrated_model.fit(X_val, y_val)
        
        # c) Choose threshold on validation tail to maximize F1
        val_proba = calibrated_model.predict_proba(X_val)[:, 1]
        threshs = np.linspace(0.05, 0.95, 100)
        f1s = [f1_score(y_val, (val_proba > t).astype(int)) for t in threshs]
        best_thresh = threshs[np.argmax(f1s)]

        # d) Evaluate on test set with frozen model and threshold
        test_proba = calibrated_model.predict_proba(X_test)[:, 1]
        test_pred = (test_proba > best_thresh).astype(int)
        
        # Store results
        all_probas.extend(test_proba)
        all_ys.extend(y_test)
        
        metrics = {
            'fold': i + 1,
            'threshold': best_thresh,
            'roc_auc': roc_auc_score(y_test, test_proba),
            'pr_auc': average_precision_score(y_test, test_proba),
            'precision': precision_score(y_test, test_pred),
            'recall': recall_score(y_test, test_pred),
            'f1': f1_score(y_test, test_pred)
        }
        all_metrics.append(metrics)
        print(f"Fold {i+1} Metrics: {metrics}")

    # 6. Final Evaluation and Reporting
    print("\n--- Overall Walk-Forward Validation Results ---")
    metrics_df = pd.DataFrame(all_metrics)
    print(metrics_df.round(3))
    print("\nMean Metrics:")
    print(metrics_df.mean().round(3))

    print("\nOverall Confusion Matrix:")
    overall_preds = (np.array(all_probas) > metrics_df['threshold'].mean()).astype(int)
    print(confusion_matrix(all_ys, overall_preds))
    
    # 7. Simple PnL Simulation
    results_df = pd.DataFrame({'y_true': all_ys, 'y_proba': all_probas}, index=X.iloc[np.concatenate([t for _, t in outer_cv.split(X)])].index)
    results_df = results_df.join(df1h[['open', 'close']])
    results_df['pnl'] = np.where(
        (np.array(all_probas) > metrics_df['threshold'].mean()).astype(int) == 1,
        results_df['close'].pct_change().shift(-1), # Trade next bar close
        0
    )
    results_df['pnl_costed'] = results_df['pnl'] - (FEES_BPS / 10000) * (results_df['pnl'] != 0)
    
    results_df['cum_pnl'] = (1 + results_df['pnl_costed']).cumprod()
    print(f"\nFinal Cumulative PnL (with costs): {results_df['cum_pnl'].iloc[-1]:.4f}")

    # 8. Train Final Model and Save
    print("\nTraining final model on all data...")
    final_model = search.best_estimator_
    final_model.fit(X, y)
    joblib.dump(final_model, MODEL_OUTPUT_FILE)
    print(f"✅ Saved final model to '{MODEL_OUTPUT_FILE}'")

if __name__ == "__main__":
    main()
