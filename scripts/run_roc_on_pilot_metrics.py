import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc

# This gets the root project folder by moving up from the current script file.
# I used this so the script can correctly find the data and results folders.
ROOT = Path(__file__).resolve().parents[1]

# This is the input CSV file that contains the labeled function metrics.
DATA_PATH = ROOT / "data" / "pilot_metrics_labeled.csv"

# This is the folder where I save the ROC curve output image.
RESULTS_DIR = ROOT / "results" / "multilang_roc"

# These are the numeric features I use to train the model.
# I included code size, complexity, parameter count, and Radon/Lizard complexity values.
FEATURES = ["nloc", "ccn", "params", "length", "radon_cc"]


def main():
    # This makes sure the output folder exists before saving the ROC curve image.
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # This loads the labeled metrics CSV into a pandas DataFrame.
    df = pd.read_csv(DATA_PATH)

    # This safely converts all feature columns into numeric values.
    # If a value cannot be converted, pandas changes it to NaN.
    for f in FEATURES:
        df[f] = pd.to_numeric(df[f], errors="coerce")

    # This safely converts the bug_label column into numeric form too.
    df["bug_label"] = pd.to_numeric(df["bug_label"], errors="coerce")

    # This removes rows that are missing required feature values or labels.
    labeled_df = df.dropna(subset=FEATURES + ["bug_label"]).copy()
    labeled_df["bug_label"] = labeled_df["bug_label"].astype(int)

    # This stops the script if no labeled rows are available.
    if labeled_df.empty:
        print("No labeled rows found in bug_label.")
        return

    # This stops the script if the labels do not contain both classes.
    # A ROC curve only works if I have both 0 and 1 labels.
    if labeled_df["bug_label"].nunique() < 2:
        print("You need both 0 and 1 in bug_label to build a ROC curve.")
        return

    # This creates the feature matrix X and target vector y.
    X = labeled_df[FEATURES].fillna(0)
    y = labeled_df["bug_label"]

    # This splits the data into training and testing sets.
    # I used stratify=y so both sets keep a similar class balance.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
        stratify=y
    )

    # This builds the Random Forest classifier.
    # I used class_weight="balanced" to help if the labels are uneven.
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced"
    )

    # This trains the model on the training data.
    model.fit(X_train, y_train)

    # This gets the predicted probability that each test row belongs to class 1.
    y_prob = model.predict_proba(X_test)[:, 1]

    # This computes the false positive rate, true positive rate, and AUC score.
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    # This creates the ROC curve figure.
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve for BugBuddy Pilot Model")
    plt.legend(loc="lower right")
    plt.tight_layout()

    # This saves the ROC curve image into the results folder.
    plt.savefig(RESULTS_DIR / "roc_curve.png", dpi=200)
    plt.close()

    # These print statements show how many labeled rows were used
    # and where the final ROC curve image was saved.
    print(f"Labeled rows used: {len(labeled_df)}")
    print(f"ROC AUC: {roc_auc:.3f}")
    print(f"Saved ROC curve to: {RESULTS_DIR / 'roc_curve.png'}")


if __name__ == "__main__":
    # This runs the script only when I execute this file directly.
    main()