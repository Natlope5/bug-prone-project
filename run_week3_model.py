import json
import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, roc_curve

DATA_PATH = "data/pilot_metrics.csv"
OUTPUT_DIR = "results/week3"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load data
df = pd.read_csv(DATA_PATH)
print("Columns:", df.columns.tolist())

# Use nloc (LOC), ccn (complexity), params (parameter count)
feature_candidates = ["nloc", "ccn", "params"]
target_col = "bug_label"

# Basic checks
missing_features = [c for c in feature_candidates if c not in df.columns]
if missing_features:
    raise ValueError(f"Missing feature columns: {missing_features}")

if target_col not in df.columns:
    raise ValueError(f"Missing target column: {target_col}")

# Build modeling dataframe
model_df = df[feature_candidates + [target_col]].copy()
print("Total rows before dropna:", len(model_df))
model_df = model_df.dropna()
print("Total rows after dropna:", len(model_df))

# Need enough labeled rows to train/evaluate
if len(model_df) < 10:
    raise ValueError(
        f"Not enough labeled rows for training (got {len(model_df)}). "
        "Please label more functions before running Week 3 models."
    )

X = model_df[feature_candidates]
y = model_df[target_col].astype(int)

label_counts = y.value_counts(dropna=False)
print("Label distribution:", label_counts.to_dict())

# Need at least 2 classes for stratified split and ROC
if len(label_counts) < 2:
    raise ValueError(
        f"Need at least two classes in bug_label, but got {label_counts.to_dict()}"
    )

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# RandomForest baseline
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# Metrics
metrics = {
    "precision": float(precision_score(y_test, y_pred, zero_division=0)),
    "recall": float(recall_score(y_test, y_pred, zero_division=0)),
    "f1": float(f1_score(y_test, y_pred, zero_division=0)),
    "roc_auc": float(roc_auc_score(y_test, y_prob)),
    "n_rows_used": int(len(model_df)),
    "train_size": int(len(X_train)),
    "test_size": int(len(X_test)),
    "features": feature_candidates,
    "model": "RandomForestClassifier"
}

print("Metrics:", metrics)

# Save metrics as JSON
metrics_path = os.path.join(OUTPUT_DIR, "metrics.json")
with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=2)

# ROC curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, label=f"ROC AUC = {metrics['roc_auc']:.3f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Week 3 ROC Curve")
plt.legend()
plt.tight_layout()
roc_path = os.path.join(OUTPUT_DIR, "roc_curve.png")
plt.savefig(roc_path, dpi=200)
plt.close()

print(f"Saved metrics to {metrics_path}")
print(f"Saved ROC curve to {roc_path}")