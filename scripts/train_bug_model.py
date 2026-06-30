import json
import pickle
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

# Train BugBuddy models from static metrics and optional repo-history metrics.
ROOT = Path(__file__).resolve().parents[1]

STATIC_CSV = ROOT / "data" / "pilot_metrics.csv"
HISTORY_CSV = ROOT / "data" / "repo_history_metrics.csv"

RESULTS_DIR = ROOT / "results" / "bug_model"
STATIC_MODEL_TOPLEVEL = ROOT / "results" / "bug_model.pkl"

STATIC_FEATURES = [
    "start_line",
    "end_line",
    "line_span",
    "nloc",
    "ccn",
    "token",
    "params",
    "length",
    "radon_cc",
    "is_test_file",
]

HISTORY_FEATURES = [
    "commit_count",
    "churn_added",
    "churn_deleted",
    "total_churn",
    "file_age_days",
    "days_since_last_change",
    "recent_7d_commits",
    "recent_30d_commits",
    "late_night_commits",
    "late_night_ratio",
    "weekend_commits",
    "weekend_ratio",
    "active_days",
    "avg_commits_per_active_day",
    "max_commits_one_day",
    "burstiness_score",
]


def ensure_output_dirs():
    """Create result folders if they do not exist yet."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    STATIC_MODEL_TOPLEVEL.parent.mkdir(parents=True, exist_ok=True)


def load_csv(path: Path):
    """Load a required CSV file."""
    if not path.exists():
        raise FileNotFoundError(f"Required input file not found: {path}")
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} rows from {path}")
    return df


def try_load_optional_csv(path: Path):
    """Load an optional CSV file, or return None if it is missing."""
    if not path.exists():
        print(f"Optional input file not found, skipping it: {path}")
        return None
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} rows from {path}")
    return df


def merge_datasets(static_df: pd.DataFrame, history_df: pd.DataFrame | None):
    """Merge static metrics with optional history metrics on function_id."""
    if "function_id" not in static_df.columns:
        raise RuntimeError(
            "Static metrics CSV is missing function_id. "
            "Update and rerun extract_metrics.py first."
        )

    if history_df is None:
        merged_df = static_df.copy()
        for col in HISTORY_FEATURES:
            if col not in merged_df.columns:
                merged_df[col] = 0
        return merged_df

    if "function_id" not in history_df.columns:
        raise RuntimeError(
            "Repository history CSV is missing function_id. "
            "Run extract_repo_history.py first."
        )

    history_keep_cols = ["function_id"] + [c for c in HISTORY_FEATURES if c in history_df.columns]
    merged_df = static_df.merge(
        history_df[history_keep_cols].drop_duplicates(subset=["function_id"]),
        on="function_id",
        how="left",
    )

    for col in HISTORY_FEATURES:
        if col not in merged_df.columns:
            merged_df[col] = 0

    return merged_df


def coerce_numeric_columns(df: pd.DataFrame, columns):
    """Ensure selected columns exist and are numeric."""
    for col in columns:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def prepare_labeled_data(df: pd.DataFrame, features):
    """Keep labeled rows only and return X/y for model training."""
    if "bug_label" not in df.columns:
        raise RuntimeError("The merged dataset does not contain bug_label.")

    df = df.copy()
    df["bug_label"] = pd.to_numeric(df["bug_label"], errors="coerce")

    labeled_df = df.dropna(subset=["bug_label"]).copy()
    labeled_df["bug_label"] = labeled_df["bug_label"].astype(int)

    if labeled_df.empty:
        raise RuntimeError(
            "No labeled rows found. Check that bug_label contains 0 and 1 values."
        )

    if labeled_df["bug_label"].nunique() < 2:
        raise RuntimeError(
            "You need both 0 and 1 in bug_label to train and evaluate a classifier."
        )

    X = labeled_df[features].fillna(0)
    y = labeled_df["bug_label"]
    return labeled_df, X, y


def train_models(X_train, y_train):
    """Train the Random Forest and Logistic Regression models."""
    rf_model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
    )
    rf_model.fit(X_train, y_train)

    lr_model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
    )
    lr_model.fit(X_train, y_train)

    return {
        "random_forest": rf_model,
        "logistic_regression": lr_model,
    }


def evaluate_model(model, X_test, y_test):
    """Compute standard classification metrics and ROC data."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    accuracy = accuracy_score(y_test, y_pred)

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "accuracy": float(accuracy),
        "roc_auc": float(roc_auc),
        "confusion_matrix": cm.tolist(),
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
    }


def get_feature_importance(model, feature_names):
    """Return feature importance values in descending order."""
    if hasattr(model, "feature_importances_"):
        return dict(
            sorted(
                zip(feature_names, model.feature_importances_),
                key=lambda x: x[1],
                reverse=True,
            )
        )

    if hasattr(model, "coef_"):
        coef = model.coef_[0]
        return dict(
            sorted(
                zip(feature_names, [abs(v) for v in coef]),
                key=lambda x: x[1],
                reverse=True,
            )
        )

    return {}


def score_all_rows(df: pd.DataFrame, model, features, model_name, feature_set_name):
    """Score every row in the dataset with the selected model."""
    scoring_df = df.copy()

    for feature in features:
        if feature not in scoring_df.columns:
            scoring_df[feature] = 0

    X_all = scoring_df[features].fillna(0)
    bug_prob = model.predict_proba(X_all)[:, 1]

    results = []
    for row, prob in zip(scoring_df.to_dict("records"), bug_prob):
        results.append({
            "model_name": model_name,
            "feature_set": feature_set_name,
            "repo_name": clean_value(row.get("repo_name")),
            "file": clean_value(row.get("file")),
            "function_id": clean_value(row.get("function_id")),
            "function": clean_value(row.get("function")),
            "language": clean_value(row.get("language")),
            "nloc": clean_value(row.get("nloc")),
            "ccn": clean_value(row.get("ccn")),
            "params": clean_value(row.get("params")),
            "length": clean_value(row.get("length")),
            "radon_cc": clean_value(row.get("radon_cc")),
            "commit_count": clean_value(row.get("commit_count")),
            "total_churn": clean_value(row.get("total_churn")),
            "days_since_last_change": clean_value(row.get("days_since_last_change")),
            "late_night_ratio": clean_value(row.get("late_night_ratio")),
            "burstiness_score": clean_value(row.get("burstiness_score")),
            "bug_label": clean_value(row.get("bug_label")),
            "bug_probability": float(prob),
        })

    return results


def clean_value(value):
    """Convert pandas NaN values to None for JSON output."""
    return None if pd.isna(value) else value


def save_pickle(path: Path, payload):
    """Save a Python object as a pickle file."""
    with path.open("wb") as f:
        pickle.dump(payload, f)


def save_json(path: Path, payload):
    """Save a Python object as formatted JSON."""
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def build_model_payload(result):
    """Build the model payload used by the Flask app."""
    return {
        "model": result["best_model"],
        "features": result["features"],
        "feature_set_name": result["feature_set_name"],
        "best_model_name": result["best_model_name"],
        "status_label": "Model ready",
    }


def train_feature_set(df: pd.DataFrame, feature_set_name: str, features):
    """Train and evaluate both models for one feature set."""
    df = coerce_numeric_columns(df.copy(), features)
    labeled_df, X, y = prepare_labeled_data(df, features)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
        stratify=y,
    )

    trained_models = train_models(X_train, y_train)

    model_results = {}
    best_model_name = None
    best_model = None
    best_auc = -1

    # Pick the model with the highest ROC AUC for this feature set.
    for model_name, model in trained_models.items():
        metrics = evaluate_model(model, X_test, y_test)
        importances = get_feature_importance(model, features)

        model_results[model_name] = {
            "metrics": metrics,
            "feature_importance": importances,
        }

        if metrics["roc_auc"] > best_auc:
            best_auc = metrics["roc_auc"]
            best_model_name = model_name
            best_model = model

    all_predictions = score_all_rows(df, best_model, features, best_model_name, feature_set_name)

    return {
        "feature_set_name": feature_set_name,
        "features": features,
        "num_total_rows": int(len(df)),
        "num_labeled_rows": int(len(labeled_df)),
        "best_model_name": best_model_name,
        "best_model_auc": float(best_auc),
        "all_model_results": model_results,
        "best_model": best_model,
        "predictions": all_predictions,
    }


def main():
    """Train static-only and combined models, then save outputs."""
    ensure_output_dirs()

    static_df = load_csv(STATIC_CSV)
    history_df = try_load_optional_csv(HISTORY_CSV)

    merged_df = merge_datasets(static_df, history_df)
    merged_df = coerce_numeric_columns(merged_df, STATIC_FEATURES + HISTORY_FEATURES)

    static_result = train_feature_set(
        merged_df,
        feature_set_name="static_only",
        features=STATIC_FEATURES,
    )

    combined_features = STATIC_FEATURES + HISTORY_FEATURES
    combined_result = train_feature_set(
        merged_df,
        feature_set_name="combined",
        features=combined_features,
    )

    summary = {
        "history_csv_found": history_df is not None,
        "static_only": {
            "features": static_result["features"],
            "num_total_rows": static_result["num_total_rows"],
            "num_labeled_rows": static_result["num_labeled_rows"],
            "best_model_name": static_result["best_model_name"],
            "best_model_auc": static_result["best_model_auc"],
            "all_model_results": static_result["all_model_results"],
        },
        "combined": {
            "features": combined_result["features"],
            "num_total_rows": combined_result["num_total_rows"],
            "num_labeled_rows": combined_result["num_labeled_rows"],
            "best_model_name": combined_result["best_model_name"],
            "best_model_auc": combined_result["best_model_auc"],
            "all_model_results": combined_result["all_model_results"],
        },
    }

    save_json(RESULTS_DIR / "training_summary.json", summary)

    static_payload = build_model_payload(static_result)
    combined_payload = build_model_payload(combined_result)

    save_pickle(RESULTS_DIR / "static_only_model.pkl", static_payload)
    save_pickle(RESULTS_DIR / "combined_model.pkl", combined_payload)
    save_pickle(STATIC_MODEL_TOPLEVEL, static_payload)

    save_json(RESULTS_DIR / "static_only_predictions.json", static_result["predictions"])
    save_json(RESULTS_DIR / "combined_predictions.json", combined_result["predictions"])

    print("\nTraining complete.")
    print(f"History CSV found: {history_df is not None}")
    print(f"Static-only best model: {static_result['best_model_name']}")
    print(f"Static-only best ROC AUC: {static_result['best_model_auc']:.3f}")
    print(f"Combined best model: {combined_result['best_model_name']}")
    print(f"Combined best ROC AUC: {combined_result['best_model_auc']:.3f}")
    print(f"Saved results to: {RESULTS_DIR}")
    print(f"Saved top-level static model to: {STATIC_MODEL_TOPLEVEL}")
    print(
        "UI status label for saved model: "
        f"{static_payload['status_label']}"
    )
    print(
        "Loaded saved model details (terminal only): "
        f"{static_payload['best_model_name']} with features: "
        f"{', '.join(static_payload['features'])}"
    )


if __name__ == "__main__":
    main()