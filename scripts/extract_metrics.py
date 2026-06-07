import csv
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_CSV = ROOT / "data" / "pilot_metrics.csv"
OUT_JSON = ROOT / "results" / "summary.json"


def parse_radon(text, target_name):
    rows = []

    for line in text.splitlines():
        line = line.strip()

        if not line.startswith("F "):
            continue

        parts = line.split(" - ")
        if len(parts) != 2:
            continue

        left, right = parts
        grade_part = right.strip()

        if "(" not in grade_part or ")" not in grade_part:
            continue

        grade = grade_part.split("(")[0].strip()
        cc = int(grade_part.split("(")[1].replace(")", "").strip())

        left_parts = left.split()
        if len(left_parts) < 3:
            continue

        line_info = left_parts[1]
        function_name = " ".join(left_parts[2:])
        start_line = int(line_info.split(":")[0])

        rows.append({
            "file": target_name,
            "function": function_name,
            "start_line": start_line,
            "end_line": "",
            "nloc": "",
            "ccn": cc,
            "token": "",
            "params": "",
            "length": "",
            "radon_grade": grade,
            "radon_cc": cc,
            "bug_label": ""
        })

    return rows


def ensure_output_dirs():
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)


def run_radon(target):
    result = subprocess.run(
        [sys.executable, "-m", "radon", "cc", str(target), "-s"],
        capture_output=True,
        text=True
    )
    return result


def write_csv(rows):
    fieldnames = [
        "file",
        "function",
        "start_line",
        "end_line",
        "nloc",
        "ccn",
        "token",
        "params",
        "length",
        "radon_grade",
        "radon_cc",
        "bug_label"
    ]

    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(summary):
    with OUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def main():
    if len(sys.argv) < 2:
        print("Usage: py extract_metrics.py sample_student_code.py")
        return

    target = Path(sys.argv[1])

    if not target.exists():
        print(f"File not found: {target}")
        return

    ensure_output_dirs()

    print(f"Analyzing file: {target.name}")

    radon_result = run_radon(target)

    if radon_result.returncode != 0:
        print("Radon failed to run.")
        print(radon_result.stderr.strip())
        return

    rows = parse_radon(radon_result.stdout, target.name)

    write_csv(rows)

    summary = {
        "file": target.name,
        "function_count": len(rows),
        "functions": rows
    }
    write_json(summary)

    print(f"Functions found: {len(rows)}")

    if rows:
        print("\nFunction-level metrics:")
        for row in rows:
            print(
                f"- {row['function']} | line {row['start_line']} | "
                f"CC={row['radon_cc']} | grade={row['radon_grade']}"
            )
    else:
        print("No functions were detected.")

    print(f"\nCSV saved to: {OUT_CSV}")
    print(f"JSON saved to: {OUT_JSON}")


if __name__ == "__main__":
    main()