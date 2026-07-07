import json
import pickle
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

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
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    STATIC_MODEL_TOPLEVEL.parent.mkdir(parents=True, exist_ok=True)


def load_csv(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Required input file not found: {path}")
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} rows from {path}")
    return df


def try_load_optional_csv(path: Path):
    if not path.exists():
        print(f"Optional input file not found, skipping it: {path}")
        return None
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} rows from {path}")
    return df


def merge_datasets(static_df: pd.DataFrame, history_df: pd.DataFrame | None):
    if "function_id" not in static_df.columns:
        raise RuntimeError("Static metrics CSV is missing function_id.")
    if "file_id" not in static_df.columns:
        raise RuntimeError("Static metrics CSV is missing file_id.")

    merged_df = static_df.copy()

    if history_df is not None:
        if "file_id" not in history_df.columns:
            raise RuntimeError("Repository history CSV is missing file_id.")

        history_keep_cols = ["file_id", "bug_label"] + [
            c for c in HISTORY_FEATURES if c in history_df.columns
        ]

        merged_df = merged_df.merge(
            history_df[history_keep_cols].drop_duplicates(subset=["file_id"]),
            on="file_id",
            how="left",
            suffixes=("", "_history"),
        )

        if "bug_label_history" in merged_df.columns and "bug_label" in merged_df.columns:
            merged_df["bug_label"] = pd.to_numeric(
                merged_df["bug_label_history"], errors="coerce"
            ).combine_first(pd.to_numeric(merged_df["bug_label"], errors="coerce"))
            merged_df = merged_df.drop(columns=["bug_label_history"])
        elif "bug_label_history" in merged_df.columns:
            merged_df = merged_df.rename(columns={"bug_label_history": "bug_label"})

    for col in HISTORY_FEATURES:
        if col not in merged_df.columns:
            merged_df[col] = 0

    return merged_df


def coerce_numeric_columns(df: pd.DataFrame, columns):
    for col in columns:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    return df


def prepare_labeled_data(df: pd.DataFrame, features):
    if "bug_label" not in df.columns:
        raise RuntimeError("The merged dataset does not contain bug_label.")

    df = df.copy()
    df["bug_label"] = pd.to_numeric(df["bug_label"], errors="coerce")
    labeled_df = df.dropna(subset=["bug_label"]).copy()
    labeled_df["bug_label"] = labeled_df["bug_label"].astype(int)

    if labeled_df.empty:
        raise RuntimeError(
            "No labeled rows found. Generate repo_history_metrics.csv with bug_label values first."
        )

    if labeled_df["bug_label"].nunique() < 2:
        raise RuntimeError("You need both 0 and 1 in bug_label.")

    if "repo_name" not in labeled_df.columns:
        labeled_df["repo_name"] = "unknown"
    labeled_df["repo_name"] = labeled_df["repo_name"].fillna("unknown").astype(str)

    X = labeled_df[features].fillna(0)
    y = labeled_df["bug_label"]
    groups = labeled_df["repo_name"]

    return labeled_df, X, y, groups


def split_with_groups(X, y, groups):
    gss_outer = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
    train_idx, temp_idx = next(gss_outer.split(X, y, groups=groups))

    X_train = X.iloc[train_idx]
    y_train = y.iloc[train_idx]

    X_temp = X.iloc[temp_idx]
    y_temp = y.iloc[temp_idx]
    groups_temp = groups.iloc[temp_idx]

    gss_inner = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=42)
    val_idx, test_idx = next(gss_inner.split(X_temp, y_temp, groups=groups_temp))

    X_val = X_temp.iloc[val_idx]
    y_val = y_temp.iloc[val_idx]
    X_test = X_temp.iloc[test_idx]
    y_test = y_temp.iloc[test_idx]

    return X_train, X_val, X_test, y_train, y_val, y_test


def train_models(X_train, y_train):
    rf_model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced_subsample",
        min_samples_leaf=2,
        n_jobs=-1,
    )
    rf_model.fit(X_train, y_train)

    lr_model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=5000,
            class_weight="balanced",
            random_state=42,
            solver="lbfgs",
        ),
    )
    lr_model.fit(X_train, y_train)

    return {
        "random_forest": rf_model,
        "logistic_regression": lr_model,
    }


def safe_roc_metrics(y_true, y_prob):
    unique_classes = pd.Series(y_true).dropna().unique()
    if len(unique_classes) < 2:
        return None, [], []
    roc_auc = roc_auc_score(y_true, y_prob)
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return float(roc_auc), fpr.tolist(), tpr.tolist()


def evaluate_model(model, X_eval, y_eval, threshold=0.5):
    y_prob = model.predict_proba(X_eval)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    precision = precision_score(y_eval, y_pred, zero_division=0)
    recall = recall_score(y_eval, y_pred, zero_division=0)
    f1 = f1_score(y_eval, y_pred, zero_division=0)
    accuracy = accuracy_score(y_eval, y_pred)
    cm = confusion_matrix(y_eval, y_pred)
    report = classification_report(y_eval, y_pred, zero_division=0, output_dict=True)

    roc_auc, fpr, tpr = safe_roc_metrics(y_eval, y_prob)

    return {
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "accuracy": float(accuracy),
        "roc_auc": roc_auc,
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "fpr": fpr,
        "tpr": tpr,
    }


def find_best_threshold(model, X_val, y_val, min_precision=0.05):
    y_prob = model.predict_proba(X_val)[:, 1]

    unique_classes = pd.Series(y_val).dropna().unique()
    if len(unique_classes) < 2:
        fallback_metrics = evaluate_model(model, X_val, y_val, threshold=0.5)
        fallback_metrics["selection_rule"] = "fallback_single_class_validation"
        return 0.5, fallback_metrics

    _, _, thresholds = precision_recall_curve(y_val, y_prob)

    threshold_candidates = sorted(set([0.5] + [float(t) for t in thresholds]))

    best_threshold = 0.5
    best_metrics = evaluate_model(model, X_val, y_val, threshold=0.5)
    best_score = float("-inf")

    for threshold in threshold_candidates:
        metrics = evaluate_model(model, X_val, y_val, threshold=threshold)

        precision = metrics["precision"]
        recall = metrics["recall"]
        f1 = metrics["f1_score"]

        if precision >= min_precision:
            score = f1 + (recall * 0.001)
        else:
            score = precision + (recall * 0.0001)

        if score > best_score:
            best_score = score
            best_threshold = threshold
            best_metrics = metrics

    best_metrics["selection_rule"] = f"best_f1_with_min_precision_{min_precision}"
    return float(best_threshold), best_metrics


def get_feature_importance(model, feature_names):
    if hasattr(model, "feature_importances_"):
        return dict(
            sorted(
                zip(feature_names, model.feature_importances_),
                key=lambda x: x[1],
                reverse=True,
            )
        )

    if hasattr(model, "named_steps") and "logisticregression" in model.named_steps:
        coef = model.named_steps["logisticregression"].coef_[0]
        return dict(
            sorted(
                zip(feature_names, [abs(v) for v in coef]),
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


def clean_value(value):
    return None if pd.isna(value) else value


def score_all_rows(df: pd.DataFrame, model, features, model_name, feature_set_name, threshold):
    scoring_df = df.copy()

    for feature in features:
        if feature not in scoring_df.columns:
            scoring_df[feature] = 0

    X_all = scoring_df[features].fillna(0)
    bug_prob = model.predict_proba(X_all)[:, 1]
    bug_pred = (bug_prob >= threshold).astype(int)

    results = []
    for row, prob, pred in zip(scoring_df.to_dict("records"), bug_prob, bug_pred):
        results.append({
            "model_name": model_name,
            "feature_set": feature_set_name,
            "threshold": float(threshold),
            "predicted_bug_label": int(pred),
            "repo_name": clean_value(row.get("repo_name")),
            "file": clean_value(row.get("file")),
            "file_id": clean_value(row.get("file_id")),
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


def save_pickle(path: Path, payload):
    with path.open("wb") as f:
        pickle.dump(payload, f)


def save_json(path: Path, payload):
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def build_model_payload(result):
    return {
        "model": result["best_model"],
        "features": result["features"],
        "feature_set_name": result["feature_set_name"],
        "best_model_name": result["best_model_name"],
        "threshold": result["best_threshold"],
        "status_label": "Model ready",
    }


def train_feature_set(df: pd.DataFrame, feature_set_name: str, features):
    df = coerce_numeric_columns(df.copy(), features)
    labeled_df, X, y, groups = prepare_labeled_data(df, features)

    X_train, X_val, X_test, y_train, y_val, y_test = split_with_groups(X, y, groups)
    trained_models = train_models(X_train, y_train)

    model_results = {}
    best_model_name = None
    best_model = None
    best_score = float("-inf")
    best_threshold = 0.5

    for model_name, model in trained_models.items():
        tuned_threshold, val_metrics = find_best_threshold(
            model, X_val, y_val, min_precision=0.05
        )
        test_metrics = evaluate_model(model, X_test, y_test, threshold=tuned_threshold)
        importances = get_feature_importance(model, features)

        model_results[model_name] = {
            "selected_threshold": float(tuned_threshold),
            "validation_metrics": val_metrics,
            "test_metrics": test_metrics,
            "feature_importance": importances,
        }

        selection_score = val_metrics.get("f1_score", 0.0)

        if selection_score > best_score:
            best_score = selection_score
            best_model_name = model_name
            best_model = model
            best_threshold = tuned_threshold

    if best_model_name is None:
        best_model_name, best_model = next(iter(trained_models.items()))
        best_threshold = 0.5
        best_score = model_results[best_model_name]["validation_metrics"].get("f1_score", 0.0)

    best_test_metrics = model_results[best_model_name]["test_metrics"]
    all_predictions = score_all_rows(
        df,
        best_model,
        features,
        best_model_name,
        feature_set_name,
        best_threshold,
    )
    best_auc = model_results[best_model_name]["validation_metrics"].get("roc_auc")

    return {
        "feature_set_name": feature_set_name,
        "features": features,
        "num_total_rows": int(len(df)),
        "num_labeled_rows": int(len(labeled_df)),
        "best_model_name": best_model_name,
        "best_threshold": float(best_threshold),
        "best_model_auc": None if best_auc is None else float(best_auc),
        "best_model_auc_or_f1_for_selection": float(best_score),
        "best_test_metrics": best_test_metrics,
        "all_model_results": model_results,
        "best_model": best_model,
        "predictions": all_predictions,
    }


def main():
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
            "best_threshold": static_result["best_threshold"],
            "best_validation_auc": static_result["best_model_auc"],
            "best_validation_auc_or_f1_for_selection": static_result["best_model_auc_or_f1_for_selection"],
            "best_test_metrics": static_result["best_test_metrics"],
            "all_model_results": static_result["all_model_results"],
        },
        "combined": {
            "features": combined_result["features"],
            "num_total_rows": combined_result["num_total_rows"],
            "num_labeled_rows": combined_result["num_labeled_rows"],
            "best_model_name": combined_result["best_model_name"],
            "best_threshold": combined_result["best_threshold"],
            "best_validation_auc": combined_result["best_model_auc"],
            "best_validation_auc_or_f1_for_selection": combined_result["best_model_auc_or_f1_for_selection"],
            "best_test_metrics": combined_result["best_test_metrics"],
            "all_model_results": combined_result["all_model_results"],
        },
        "note": "Labels should come from repo_history_metrics.csv bug-fix history, not from static threshold heuristics.",
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
    print(f"Static-only best threshold: {static_result['best_threshold']}")
    print(f"Static-only best validation ROC AUC: {static_result['best_model_auc']}")
    print(f"Static-only selection metric: {static_result['best_model_auc_or_f1_for_selection']}")
    print(f"Static-only test precision: {static_result['best_test_metrics']['precision']:.3f}")
    print(f"Static-only test recall: {static_result['best_test_metrics']['recall']:.3f}")
    print(f"Static-only test F1: {static_result['best_test_metrics']['f1_score']:.3f}")
    print(f"Static-only test ROC AUC: {static_result['best_test_metrics']['roc_auc']}")
    print(f"Combined best model: {combined_result['best_model_name']}")
    print(f"Combined best threshold: {combined_result['best_threshold']}")
    print(f"Combined best validation ROC AUC: {combined_result['best_model_auc']}")
    print(f"Combined selection metric: {combined_result['best_model_auc_or_f1_for_selection']}")
    print(f"Combined test precision: {combined_result['best_test_metrics']['precision']:.3f}")
    print(f"Combined test recall: {combined_result['best_test_metrics']['recall']:.3f}")
    print(f"Combined test F1: {combined_result['best_test_metrics']['f1_score']:.3f}")
    print(f"Combined test ROC AUC: {combined_result['best_test_metrics']['roc_auc']}")
    print(f"Saved results to: {RESULTS_DIR}")
    print(f"Saved top-level static model to: {STATIC_MODEL_TOPLEVEL}")


if __name__ == "__main__":
    main()