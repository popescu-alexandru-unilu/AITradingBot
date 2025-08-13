# src/train_meta_model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from xgboost import XGBClassifier
import joblib

# 1) Load the labeled data
df = pd.read_csv("data/meta_labeled_dataset.csv", index_col="timestamp", parse_dates=True)
df.dropna(subset=["label"], inplace=True)

# 2) Define features & target
feature_cols = [c for c in df.columns 
                if c not in ["label","t1","vol","sma_fast","sma_slow","side"]]
X = df[feature_cols]
y = df["label"].map({-1:0, 1:1}) # map -1->0, +1->1

# 3) Time-series CV with GridSearchCV
tscv = TimeSeriesSplit(n_splits=5)
model = XGBClassifier(tree_method="gpu_hist", device="cuda", use_label_encoder=False, eval_metric='logloss')

param_grid = {
  "learning_rate": [0.01, 0.05, 0.1],
  "max_depth": [3,5],
  "n_estimators": [50,100,150],
  "subsample": [0.8,1.0],
  "colsample_bytree": [0.6,0.8,1.0],
}

grid = GridSearchCV(model, param_grid, cv=tscv, scoring="roc_auc", n_jobs=-1)
grid.fit(X, y)

print("Best params:", grid.best_params_)
print("Best ROC-AUC:", grid.best_score_)

# 4) Fit on all data and persist the best estimator
best_clf = grid.best_estimator_
joblib.dump(best_clf, "models/xgb_meta_labeler.pkl")
print("Meta-model saved to 'models/xgb_meta_labeler.pkl'")
