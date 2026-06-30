import json
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Week 4 script: train a model on labeled function metrics and score all functions.

# Project root so data and results paths are stable.
ROOT = Path(__file__).resolve().parents[1]

# Metrics CSV with function-level features and labels.
DATA_PATH = ROOT / "data" / "pilot_metrics.csv"

# Where prediction output is saved.
OUTPUT_PATH = ROOT / "results" / "week4_predictions.json"

# Feature columns used for training and prediction.
FEATURES = [
    "start_line",
    "end_line",
    "nloc",
    "ccn",
    "token",
    "params",
    "length",
    "radon_cc",
]


def load_data(csv_path: Path) -> pd.DataFrame:
    """Load the metrics CSV into a DataFrame."""
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Data file not found at {csv_path}. "
            "Run the metrics extraction script first."
        )

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path}")
    return df


def prepare_features(df: pd.DataFrame, features):
    """Ensure all feature columns exist and are numeric."""
    prepared = df.copy()

    for feature in features:
        if feature not in prepared.columns:
            prepared[feature] = 0

        prepared[feature] = pd.to_numeric(prepared[feature], errors="coerce").fillna(0)

    return prepared


def train_model(labeled_df: pd.DataFrame):
    """Train a Random Forest model on labeled rows only."""
    X = labeled_df[FEATURES]
    y = labeled_df["bug_label"].astype(int)

    label_counts = y.value_counts(dropna=False)
    print("Label distribution (Week 4):", label_counts.to_dict())

    if len(label_counts) < 2:
        raise RuntimeError(
            "Need at least two classes in bug_label to train the Week 4 model, "
            f"but got distribution {label_counts.to_dict()}."
        )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced",
    )
    model.fit(X, y)

    return model, FEATURES


def clean_value(value):
    """Turn pandas NaN into None for nicer JSON output."""
    return None if pd.isna(value) else value


def score_all_functions(df: pd.DataFrame, model, features):
    """Score every function and return a list of prediction dicts."""
    scoring_df = prepare_features(df, features)
    X_all = scoring_df[features]
    bug_prob = model.predict_proba(X_all)[:, 1]

    results = []
    for original_row, prob in zip(df.to_dict("records"), bug_prob):
        results.append({
            "file": clean_value(original_row.get("file")),
            "function": clean_value(original_row.get("function")),
            "nloc": clean_value(original_row.get("nloc")),
            "ccn": clean_value(original_row.get("ccn")),
            "params": clean_value(original_row.get("params")),
            "bug_label": clean_value(original_row.get("bug_label")),
            "bug_probability": float(prob),
        })

    return results


def main():
    """Load data, train the Week 4 model, score all functions, and save JSON."""
    df = load_data(DATA_PATH)

    if "bug_label" not in df.columns:
        raise RuntimeError(
            "Missing required column: bug_label. "
            "Make sure your labeling script has already been run."
        )

    missing_features = [c for c in FEATURES if c not in df.columns]
    if missing_features:
        print(f"Warning: missing feature columns will be filled with 0: {missing_features}")

    working_df = prepare_features(df, FEATURES)
    working_df["bug_label"] = pd.to_numeric(working_df["bug_label"], errors="coerce")

    labeled_df = working_df.dropna(subset=["bug_label"]).copy()
    print(f"Labeled rows: {len(labeled_df)}")

    if labeled_df.empty:
        raise RuntimeError(
            "No labeled rows found in data/pilot_metrics.csv. "
            "Check that bug_label contains 0/1 values."
        )

    model, features = train_model(labeled_df)
    results = score_all_functions(df, model, features)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "features": features,
                "num_rows": len(df),
                "predictions": results,
            },
            f,
            indent=2,
        )

    print(f"Saved predictions for {len(df)} functions to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()