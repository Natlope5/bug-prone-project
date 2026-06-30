import pandas as pd
from pathlib import Path

# This gets the root project folder by moving up from the current script file.
# I used this so the script can find the data folder correctly from inside the scripts folder.
ROOT = Path(__file__).resolve().parents[1]

# This is the input CSV file that contains the extracted function metrics.
DATA_PATH = ROOT / "data" / "pilot_metrics.csv"

# This is the output CSV file where I save the labeled copy.
# I used a different file name so the original source file stays unchanged.
OUTPUT_PATH = ROOT / "data" / "pilot_metrics_labeled.csv"

# These are the risk thresholds I used for the heuristic bug labeling.
# If any of these metrics are high enough, the row gets labeled as bug-prone.
CCN_HIGH = 6
NLOC_HIGH = 20
PARAMS_HIGH = 4
LENGTH_HIGH = 30


def main():
    # This loads the function metrics CSV into a pandas DataFrame.
    df = pd.read_csv(DATA_PATH)

    # This checks that the expected feature columns exist before continuing.
    # If one is missing, the script stops and tells me exactly which one is missing.
    for col in ["nloc", "ccn", "params", "length"]:
        if col not in df.columns:
            raise ValueError(f"Missing expected column: {col}")

    # This safely converts the feature columns into numeric values.
    # If a value cannot be converted, pandas changes it to NaN.
    for col in ["nloc", "ccn", "params", "length"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # This creates a heuristic bug label.
    # I mark the row as risky if any one of the key metrics crosses its threshold.
    risky = (
        (df["ccn"] >= CCN_HIGH) |
        (df["nloc"] >= NLOC_HIGH) |
        (df["params"] >= PARAMS_HIGH) |
        (df["length"] >= LENGTH_HIGH)
    )

    # This converts the True/False risky values into 1 and 0 labels.
    df["bug_label"] = risky.astype(int)

    # This makes sure the output folder exists before saving the new CSV file.
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # This saves the labeled copy so I keep the original metrics file unchanged.
    df.to_csv(OUTPUT_PATH, index=False)

    # This creates a small summary of how many rows were labeled 0 or 1.
    counts = df["bug_label"].value_counts().to_dict()

    # These print statements confirm where the file was saved and show the label counts.
    print("Saved labeled file to:", OUTPUT_PATH)
    print("Counts:", counts)


if __name__ == "__main__":
    # This runs the script only when I execute this file directly.
    main()