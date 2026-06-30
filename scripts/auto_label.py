import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "data" / "pilot_metrics.csv"


def to_int(value):
    try:
        if value is None or str(value).strip() == "":
            return None
        return int(float(value))
    except (TypeError, ValueError):
        return None


def auto_bug_label(ccn, grade, nloc, params, length):
    ccn = to_int(ccn)
    nloc = to_int(nloc)
    params = to_int(params)
    length = to_int(length)
    grade = str(grade).strip().upper()

    # High-risk / likely bug-prone
    if grade in {"D", "E", "F"}:
        return 1
    if ccn is not None and ccn >= 8:
        return 1
    if grade == "C":
        return 1
    if ccn is not None and nloc is not None and ccn >= 6 and nloc >= 10:
        return 1
    if params is not None and ccn is not None and params >= 4 and ccn >= 5:
        return 1
    if length is not None and ccn is not None and length >= 30 and ccn >= 5:
        return 1

    # Low-risk / likely not bug-prone
    if (
        grade == "A"
        and ccn is not None and ccn <= 2
        and (nloc is None or nloc <= 8)
        and (params is None or params <= 2)
    ):
        return 0

    if (
        grade == "B"
        and ccn is not None and ccn <= 3
        and (nloc is None or nloc <= 6)
        and (params is None or params <= 2)
    ):
        return 0

    return ""


def main():
    if not CSV_PATH.exists():
        print(f"File not found: {CSV_PATH}")
        return

    with CSV_PATH.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames

    if not fieldnames:import csv
from pathlib import Path

# This gets the project root folder by moving up from the current script file.
# I used this so the script can still find the data folder no matter where it is run from.
ROOT = Path(__file__).resolve().parents[1]

# This points to the pilot metrics CSV file that stores the function-level data.
CSV_PATH = ROOT / "data" / "pilot_metrics.csv"


def to_int(value):
    # This helper function safely converts a value into an integer.
    # If the value is missing or cannot be converted, I return None instead of crashing.
    try:
        if value is None or str(value).strip() == "":
            return None
        return int(float(value))
    except (TypeError, ValueError):
        return None


def auto_bug_label(ccn, grade, nloc, params, length):
    # This function applies my rule-based labeling logic.
    # It tries to decide whether a function should be marked as bug-prone (1) or not bug-prone (0).
    ccn = to_int(ccn)
    nloc = to_int(nloc)
    params = to_int(params)
    length = to_int(length)
    grade = str(grade).strip().upper()

    # High-risk / likely bug-prone rules.
    # If the code looks complex or poorly graded, I label it as 1.
    if grade in {"D", "E", "F"}:
        return 1

    if ccn is not None and ccn >= 8:
        return 1

    if grade == "C":
        return 1

    if ccn is not None and nloc is not None and ccn >= 6 and nloc >= 10:
        return 1

    if params is not None and ccn is not None and params >= 4 and ccn >= 5:
        return 1

    if length is not None and ccn is not None and length >= 30 and ccn >= 5:
        return 1

    # Low-risk / likely not bug-prone rules.
    # If the code looks simple and clean, I label it as 0.
    if (
        grade == "A"
        and ccn is not None and ccn <= 2
        and (nloc is None or nloc <= 8)
        and (params is None or params <= 2)
    ):
        return 0

    if (
        grade == "B"
        and ccn is not None and ccn <= 3
        and (nloc is None or nloc <= 6)
        and (params is None or params <= 2)
    ):
        return 0

    # If the row does not clearly match either side, I leave it blank.
    return ""


def main():
    # This is the main function that loads the CSV, updates missing labels,
    # and saves the new version back to the same file.
    if not CSV_PATH.exists():
        print(f"File not found: {CSV_PATH}")
        return

    # This opens the CSV file and reads all rows into memory.
    with CSV_PATH.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames

    # This makes sure the CSV actually has a header row.
    if not fieldnames:
        print("CSV has no header.")
        return

    # This makes sure the bug_label column exists.
    # If it does not exist yet, I add it so the script can write labels into the file.
    if "bug_label" not in fieldnames:
        fieldnames.append("bug_label")

    # These counters help me track what the script changed.
    updated = 0
    kept_existing = 0
    still_blank = 0

    # This loops through each row in the CSV.
    for row in rows:
        current = str(row.get("bug_label", "")).strip()

        # If the row already has a valid label, I keep it and do not overwrite it.
        if current in {"0", "1"}:
            kept_existing += 1
            continue

        # This generates a new bug label using my rule-based logic.
        new_label = auto_bug_label(
            row.get("ccn", ""),
            row.get("radon_grade", ""),
            row.get("nloc", ""),
            row.get("params", ""),
            row.get("length", ""),
        )

        # This stores the new label back into the row.
        row["bug_label"] = new_label

        # These counters track whether I updated the row or left it blank.
        if str(new_label) in {"0", "1"}:
            updated += 1
        else:
            still_blank += 1

    # This writes the updated rows back to the same CSV file.
    with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # These print statements summarize what the script did.
    print(f"Updated labels: {updated}")
    print(f"Kept existing labels: {kept_existing}")
    print(f"Still blank: {still_blank}")
    print(f"Saved: {CSV_PATH}")


if __name__ == "__main__":
    # This runs the script only when I execute the file directly.
    main()
        print("CSV has no header.")
        return

    updated = 0
    kept_existing = 0
    still_blank = 0

    for row in rows:
        current = str(row.get("bug_label", "")).strip()

        if current in {"0", "1"}:
            kept_existing += 1
            continue

        new_label = auto_bug_label(
            row.get("ccn", ""),
            row.get("radon_grade", ""),
            row.get("nloc", ""),
            row.get("params", ""),
            row.get("length", ""),
        )
        row["bug_label"] = new_label

        if str(new_label) in {"0", "1"}:
            updated += 1
        else:
            still_blank += 1

    with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Updated labels: {updated}")
    print(f"Kept existing labels: {kept_existing}")
    print(f"Still blank: {still_blank}")
    print(f"Saved: {CSV_PATH}")


if __name__ == "__main__":
    main()