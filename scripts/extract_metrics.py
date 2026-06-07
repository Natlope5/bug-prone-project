import csv          # For writing the pilot_metrics.csv file
import json         # For creating and saving the summary JSON
import re           # For parsing Lizard and Radon text output with regular expressions
import subprocess   # For calling Lizard and Radon as external commands
from pathlib import Path  # For clean, cross‑platform path handling

# ROOT points to the project root folder (bug_prone_project)
ROOT = Path(__file__).resolve().parents[1]

# SAMPLE_DIR is the folder containing the sample student-style Python files
SAMPLE_DIR = ROOT / "sample_code"

# OUT_CSV is the path where the pilot metrics CSV will be written
OUT_CSV = ROOT / "data" / "pilot_metrics.csv"


def parse_lizard(text):
    """
    Parse the plain-text output from `lizard sample_code` and extract
    one row per function with metrics like NLOC, CCN, tokens, params, and length.
    """
    rows = []
    for line in text.splitlines():
        # Remove leading/trailing spaces
        line = line.strip()

        # Skip header lines, separator lines, totals, and threshold messages
        if (
            not line
            or line.startswith("=")
            or line.startswith("-")
            or line.startswith("NLOC")
            or "file analyzed" in line
            or line.startswith("Total")
            or line.startswith("No thresholds")
        ):
            continue

        # Match data lines of the form:
        # NLOC CCN token PARAM length functionName@start-end@file_path
        m = re.match(
            r"(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(.+?)@(\d+)-(\d+)@(.+)",
            line
        )

        if m:
            # Build a dictionary of metrics for each function
            rows.append({
                "nloc": int(m.group(1)),        # Lines of code
                "ccn": int(m.group(2)),         # Cyclomatic complexity
                "token": int(m.group(3)),       # Token count
                "params": int(m.group(4)),      # Number of parameters
                "length": int(m.group(5)),      # Function length (lines)
                "function": m.group(6).strip(), # Function name
                "start_line": int(m.group(7)),  # Start line in the file
                "end_line": int(m.group(8)),    # End line in the file
                "file": m.group(9).strip(),     # File path
            })
    return rows


def parse_radon(text):
    """
    Parse the plain-text output from `radon cc sample_code -s` and
    extract complexity grades and CC numbers per function.
    Keyed by (file, function_name).
    """
    grades = {}
    current_file = None

    for line in text.splitlines():
        line = line.rstrip()

        # Lines that end with ".py" (and are not function lines) indicate a new file
        if line.endswith(".py") and not line.strip().startswith("F "):
            current_file = line.strip()
            continue

        # Match function lines like:
        # F 21:0 process_user_records - B (7)
        m = re.search(
            r"F\s+\d+:\d+\s+(.+)\s+-\s+([A-F])\s+\((\d+)\)",
            line
        )

        if m and current_file:
            func_name = m.group(1).strip()
            grade = m.group(2)
            cc = int(m.group(3))

            # Store the grade and complexity for this (file, function) pair
            grades[(current_file, func_name)] = {
                "radon_grade": grade,
                "radon_cc": cc
            }

    return grades


# Run Lizard on the sample_code folder and capture the textual output
lizard_out = subprocess.check_output(
    ["lizard", str(SAMPLE_DIR)],
    text=True
)

# Run Radon (cyclomatic complexity) on the sample_code folder and capture the output
radon_out = subprocess.check_output(
    ["radon", "cc", str(SAMPLE_DIR), "-s"],
    text=True
)

# Parse both outputs into structured lists/dicts
lizard_rows = parse_lizard(lizard_out)
radon_rows = parse_radon(radon_out)

# Make sure the data directory exists before writing the CSV
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

# Open the CSV file for writing the pilot metrics
with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "file", "function", "start_line", "end_line",
            "nloc", "ccn", "token", "params", "length",
            "radon_grade", "radon_cc", "bug_label"
        ]
    )
    # Write header row
    writer.writeheader()

    # For each function parsed from Lizard, merge in the Radon metrics
    for row in lizard_rows:
        key = (row["file"], row["function"])
        merged = {
            **row,
            # If Radon has a record for this (file, function), merge it;
            # otherwise use empty grade/cc fields
            **radon_rows.get(key, {"radon_grade": "", "radon_cc": ""}),
            # Placeholder label; will be filled later once bugs are labeled
            "bug_label": ""
        }
        writer.writerow(merged)

# Build a small summary dictionary describing this pilot run
summary = {
    "files_analyzed": len(list(SAMPLE_DIR.glob("*.py"))),  # How many .py files were in sample_code
    "functions_found": len(lizard_rows),                   # How many functions Lizard found
    "output_csv": str(OUT_CSV)                             # Where the CSV was written
}

# Save the summary as JSON in the results folder
(ROOT / "results" / "summary.json").write_text(
    json.dumps(summary, indent=2),
    encoding="utf-8"
)

# Also print the summary to the console for quick feedback
print(json.dumps(summary, indent=2))