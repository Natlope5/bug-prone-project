import csv
import json
import sys
from pathlib import Path

import lizard

ROOT = Path(__file__).resolve().parents[1]

OUT_CSV = ROOT / "data" / "pilot_metrics.csv"
OUT_JSON = ROOT / "results" / "summary.json"

ALLOWED_EXTENSIONS = {
    ".py", ".js", ".ts", ".java", ".c", ".cpp", ".h", ".hpp",
    ".cs", ".go", ".php", ".rb", ".swift", ".kt", ".m", ".mm",
}


def ensure_output_dirs():
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)


def detect_language(path: Path):
    ext = path.suffix.lower()
    mapping = {
        ".py": "Python",
        ".js": "JavaScript",
        ".ts": "TypeScript",
        ".java": "Java",
        ".c": "C",
        ".cpp": "C++",
        ".h": "C/C++ Header",
        ".hpp": "C++ Header",
        ".cs": "C#",
        ".go": "Go",
        ".php": "PHP",
        ".rb": "Ruby",
        ".swift": "Swift",
        ".kt": "Kotlin",
        ".m": "Objective-C",
        ".mm": "Objective-C++",
    }
    return mapping.get(ext, "Unknown")


def is_test_file(path: Path):
    lowered = str(path).lower()
    return int(
        "test" in path.name.lower()
        or "tests" in lowered
        or "__tests__" in lowered
        or "spec" in path.name.lower()
    )


def safe_int(value, default=0):
    try:
        if value == "" or value is None:
            return default
        return int(value)
    except Exception:
        return default


def collect_code_files(targets):
    code_files = []

    for raw_target in targets:
        target = Path(raw_target)

        if not target.is_absolute():
            target = (ROOT / target).resolve()

        if not target.exists():
            print(f"Warning: target not found and will be skipped: {raw_target}")
            continue

        if target.is_file():
            if target.suffix.lower() in ALLOWED_EXTENSIONS:
                code_files.append(target)
            else:
                print(f"Warning: unsupported file skipped: {target}")
        elif target.is_dir():
            for path in target.rglob("*"):
                if path.is_file() and path.suffix.lower() in ALLOWED_EXTENSIONS:
                    code_files.append(path)

    return sorted(set(code_files))


def build_repo_name(relative_name: str):
    parts = Path(relative_name).parts

    if "repos" in parts:
        repos_index = parts.index("repos")
        if repos_index + 1 < len(parts):
            return parts[repos_index + 1]

    if parts:
        return parts[0]

    return ""


def extract_lizard_metrics(file_path, target_name):
    rows = []

    try:
        result = lizard.analyze_file(str(file_path))
    except Exception as e:
        print(f"Warning: failed to analyze {file_path}: {e}")
        return rows

    repo_name = build_repo_name(target_name)
    language = detect_language(file_path)
    extension = file_path.suffix.lower()
    file_name = file_path.name
    test_flag = is_test_file(file_path)

    for func in result.function_list:
        try:
            params = len(func.parameters)
        except Exception:
            params = 0

        function_name = getattr(func, "name", "") or "anonymous closure"
        start_line = getattr(func, "start_line", "")
        end_line = getattr(func, "end_line", "")
        nloc = getattr(func, "nloc", "")
        ccn = getattr(func, "cyclomatic_complexity", "")
        token = getattr(func, "token_count", 0)
        length = getattr(func, "length", nloc)

        start_line_int = safe_int(start_line, 0)
        end_line_int = safe_int(end_line, 0)
        line_span = end_line_int - start_line_int + 1 if start_line_int and end_line_int else ""

        function_id = f"{target_name}:{function_name}:{start_line}"

        ccn_val = safe_int(ccn, 0)
        nloc_val = safe_int(nloc, 0)
        params_val = safe_int(params, 0)

        bug_label = 1 if (
            ccn_val >= 10
            or nloc_val >= 20
            or params_val >= 5
        ) else 0

        rows.append({
            "repo_name": repo_name,
            "file": target_name,
            "file_name": file_name,
            "relative_path": target_name,
            "extension": extension,
            "language": language,
            "is_test_file": test_flag,
            "function_id": function_id,
            "function": function_name,
            "start_line": start_line,
            "end_line": end_line,
            "line_span": line_span,
            "nloc": nloc,
            "ccn": ccn,
            "token": token,
            "params": params,
            "length": length,
            "radon_grade": "",
            "radon_cc": ccn,
            "bug_label": bug_label,
        })

    return rows


def write_csv(rows):
    fieldnames = [
        "repo_name",
        "file",
        "file_name",
        "relative_path",
        "extension",
        "language",
        "is_test_file",
        "function_id",
        "function",
        "start_line",
        "end_line",
        "line_span",
        "nloc",
        "ccn",
        "token",
        "params",
        "length",
        "radon_grade",
        "radon_cc",
        "bug_label",
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
        print("Usage:")
        print('  py "scripts\\extract_metrics.py" "data\\repos\\angular.js"')
        print('  py "scripts\\extract_metrics.py" "data\\repos"')
        print('  py "scripts\\extract_metrics.py" "data\\repos\\repo1" "data\\repos\\repo2"')
        return

    ensure_output_dirs()

    input_targets = sys.argv[1:]
    code_files = collect_code_files(input_targets)

    if not code_files:
        print("No supported code files were found in the provided input paths.")
        return

    all_rows = []
    analyzed_files = []
    skipped_files = []

    print(f"Found {len(code_files)} supported code files to analyze.")

    for file_path in code_files:
        print(f"Analyzing file: {file_path}")

        try:
            relative_name = str(file_path.relative_to(ROOT))
        except ValueError:
            relative_name = str(file_path)

        rows = extract_lizard_metrics(file_path, relative_name)

        if not rows:
            skipped_files.append(str(file_path))
            continue

        all_rows.extend(rows)
        analyzed_files.append({
            "file": relative_name,
            "repo_name": build_repo_name(relative_name),
            "language": detect_language(file_path),
            "function_count": len(rows),
        })

    write_csv(all_rows)

    summary = {
        "input_targets": input_targets,
        "files_analyzed": len(analyzed_files),
        "files_skipped": len(skipped_files),
        "total_functions": len(all_rows),
        "analyzed_files": analyzed_files,
        "skipped_files": skipped_files,
        "output_csv": str(OUT_CSV),
        "output_json": str(OUT_JSON),
    }

    write_json(summary)

    print(f"\nFiles successfully analyzed: {len(analyzed_files)}")
    print(f"Files skipped: {len(skipped_files)}")
    print(f"Total functions found: {len(all_rows)}")
    print(f"CSV saved to: {OUT_CSV}")
    print(f"JSON saved to: {OUT_JSON}")


if __name__ == "__main__":
    main()