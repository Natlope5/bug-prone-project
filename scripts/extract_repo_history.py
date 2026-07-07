import csv
import json
import re
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

REPOS_DIR = ROOT / "data" / "repos"
OUT_CSV = ROOT / "data" / "repo_history_metrics.csv"
OUT_JSON = ROOT / "results" / "repo_history_summary.json"

BUGFIX_PATTERN = re.compile(
    r"\b("
    r"fix|fixed|fixes|bug|bugs|bugfix|defect|defects|error|errors|issue|issues|"
    r"resolve|resolved|resolves|patch|patched|hotfix|repair|repaired|correct|corrected"
    r")\b",
    re.IGNORECASE,
)

FIELDNAMES = [
    "repo_name",
    "file_id",
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
    "bug_fix_commits",
    "bug_label",
]

ALLOWED_EXTENSIONS = {
    ".py", ".js", ".ts", ".java", ".c", ".cpp", ".h", ".hpp",
    ".cs", ".go", ".php", ".rb", ".swift", ".kt", ".m", ".mm",
}


def ensure_output_dirs():
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)


def normalize_rel_path(path_str: str):
    return Path(path_str).as_posix()


def build_file_id(repo_name: str, repo_relative_path: str):
    return normalize_rel_path(f"data/repos/{repo_name}/{repo_relative_path}")


def safe_int_numstat(value: str):
    value = (value or "").strip()
    if value in {"", "-"}:
        return 0
    try:
        return int(value)
    except Exception:
        return 0


def allowed_code_path(repo_relative_path: str):
    return Path(repo_relative_path).suffix.lower() in ALLOWED_EXTENSIONS


def run_git_log(repo_path: Path):
    cmd = [
        "git",
        "-C",
        str(repo_path),
        "log",
        "--numstat",
        "--date=iso-strict",
        "--no-merges",
        '--pretty=format:__COMMIT__|%H|%ad|%s',
    ]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or f"git log failed for {repo_path}")
    return result.stdout.splitlines()


def iter_repo_commits(repo_path: Path):
    lines = run_git_log(repo_path)
    current = None

    for line in lines:
        if line.startswith("__COMMIT__|"):
            if current is not None:
                yield current

            parts = line.split("|", 3)
            if len(parts) < 4:
                current = None
                continue

            _, commit_hash, date_str, subject = parts
            current = {
                "hash": commit_hash,
                "date_str": date_str,
                "subject": subject,
                "files": [],
            }
            continue

        if current is None:
            continue

        if not line.strip():
            continue

        parts = line.split("\t")
        if len(parts) != 3:
            continue

        added, deleted, file_path = parts
        file_path = normalize_rel_path(file_path)

        if not file_path or not allowed_code_path(file_path):
            continue

        current["files"].append({
            "added": safe_int_numstat(added),
            "deleted": safe_int_numstat(deleted),
            "file_path": file_path,
        })

    if current is not None:
        yield current


def parse_commit_dt(date_str: str):
    return datetime.fromisoformat(date_str.replace("Z", "+00:00")).astimezone(timezone.utc)


def collect_repo_history(repo_path: Path):
    repo_name = repo_path.name
    now = datetime.now(timezone.utc)

    file_stats = defaultdict(lambda: {
        "repo_name": repo_name,
        "commit_count": 0,
        "churn_added": 0,
        "churn_deleted": 0,
        "recent_7d_commits": 0,
        "recent_30d_commits": 0,
        "late_night_commits": 0,
        "weekend_commits": 0,
        "bug_fix_commits": 0,
        "first_seen": None,
        "last_seen": None,
        "active_days_set": set(),
        "commits_per_day": defaultdict(int),
    })

    commit_counter = 0
    commit_file_counter = 0

    for commit in iter_repo_commits(repo_path):
        commit_counter += 1

        try:
            dt = parse_commit_dt(commit["date_str"])
        except Exception:
            continue

        days_ago = (now - dt).total_seconds() / 86400.0
        is_recent_7d = days_ago <= 7
        is_recent_30d = days_ago <= 30
        is_late_night = dt.hour < 6
        is_weekend = dt.weekday() >= 5
        is_bugfix = bool(BUGFIX_PATTERN.search(commit["subject"] or ""))

        for changed in commit["files"]:
            repo_relative_path = changed["file_path"]
            if not repo_relative_path:
                continue

            commit_file_counter += 1
            file_id = build_file_id(repo_name, repo_relative_path)
            stat = file_stats[file_id]

            stat["commit_count"] += 1
            stat["churn_added"] += changed["added"]
            stat["churn_deleted"] += changed["deleted"]

            if is_recent_7d:
                stat["recent_7d_commits"] += 1
            if is_recent_30d:
                stat["recent_30d_commits"] += 1
            if is_late_night:
                stat["late_night_commits"] += 1
            if is_weekend:
                stat["weekend_commits"] += 1
            if is_bugfix:
                stat["bug_fix_commits"] += 1

            day_key = dt.date().isoformat()
            stat["active_days_set"].add(day_key)
            stat["commits_per_day"][day_key] += 1

            if stat["first_seen"] is None or dt < stat["first_seen"]:
                stat["first_seen"] = dt
            if stat["last_seen"] is None or dt > stat["last_seen"]:
                stat["last_seen"] = dt

    rows = []
    for file_id, stat in file_stats.items():
        commit_count = stat["commit_count"]
        churn_added = stat["churn_added"]
        churn_deleted = stat["churn_deleted"]
        total_churn = churn_added + churn_deleted
        active_days = len(stat["active_days_set"])
        avg_commits_per_active_day = commit_count / active_days if active_days else 0.0
        max_commits_one_day = max(stat["commits_per_day"].values()) if stat["commits_per_day"] else 0
        burstiness_score = (
            max_commits_one_day / avg_commits_per_active_day
            if avg_commits_per_active_day > 0
            else 0.0
        )

        first_seen = stat["first_seen"]
        last_seen = stat["last_seen"]

        file_age_days = int((now - first_seen).total_seconds() / 86400.0) if first_seen else 0
        days_since_last_change = int((now - last_seen).total_seconds() / 86400.0) if last_seen else 0

        late_night_ratio = stat["late_night_commits"] / commit_count if commit_count else 0.0
        weekend_ratio = stat["weekend_commits"] / commit_count if commit_count else 0.0

        bug_fix_commits = stat["bug_fix_commits"]
        bug_label = 1 if bug_fix_commits > 0 else 0

        rows.append({
            "repo_name": stat["repo_name"],
            "file_id": file_id,
            "commit_count": commit_count,
            "churn_added": churn_added,
            "churn_deleted": churn_deleted,
            "total_churn": total_churn,
            "file_age_days": file_age_days,
            "days_since_last_change": days_since_last_change,
            "recent_7d_commits": stat["recent_7d_commits"],
            "recent_30d_commits": stat["recent_30d_commits"],
            "late_night_commits": stat["late_night_commits"],
            "late_night_ratio": round(late_night_ratio, 6),
            "weekend_commits": stat["weekend_commits"],
            "weekend_ratio": round(weekend_ratio, 6),
            "active_days": active_days,
            "avg_commits_per_active_day": round(avg_commits_per_active_day, 6),
            "max_commits_one_day": max_commits_one_day,
            "burstiness_score": round(burstiness_score, 6),
            "bug_fix_commits": bug_fix_commits,
            "bug_label": bug_label,
        })

    summary = {
        "repo_name": repo_name,
        "commits_processed": commit_counter,
        "commit_file_events": commit_file_counter,
        "files_with_history": len(rows),
    }

    return rows, summary


def write_csv(rows):
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def write_json(summary):
    with OUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def collect_target_repos(input_targets):
    repos = []

    for raw_target in input_targets:
        target = Path(raw_target)

        if not target.is_absolute():
            target = (ROOT / target).resolve()

        if not target.exists():
            print(f"Warning: target not found and will be skipped: {raw_target}")
            continue

        if (target / ".git").exists():
            repos.append(target)
            continue

        if target.is_dir():
            for child in target.iterdir():
                if child.is_dir() and (child / ".git").exists():
                    repos.append(child)

    return sorted(set(repos))


def main():
    ensure_output_dirs()

    if len(sys.argv) < 2:
        input_targets = [str(REPOS_DIR)]
    else:
        input_targets = sys.argv[1:]

    repos = collect_target_repos(input_targets)

    if not repos:
        print("No git repositories found in the provided paths.")
        return

    all_rows = []
    repo_summaries = []
    skipped_repos = []

    print(f"Found {len(repos)} repositories to analyze.")

    for repo_path in repos:
        print(f"Analyzing repository history: {repo_path}")
        try:
            rows, summary = collect_repo_history(repo_path)
            all_rows.extend(rows)
            repo_summaries.append(summary)
        except Exception as e:
            print(f"Warning: failed to analyze repo history for {repo_path}: {e}")
            skipped_repos.append(str(repo_path))

    write_csv(all_rows)

    summary = {
        "input_targets": input_targets,
        "repositories_analyzed": len(repo_summaries),
        "repositories_skipped": len(skipped_repos),
        "files_with_history": len(all_rows),
        "repo_summaries": repo_summaries,
        "skipped_repos": skipped_repos,
        "output_csv": str(OUT_CSV),
        "output_json": str(OUT_JSON),
    }

    write_json(summary)

    print(f"\nRepositories successfully analyzed: {len(repo_summaries)}")
    print(f"Repositories skipped: {len(skipped_repos)}")
    print(f"Files with history rows: {len(all_rows)}")
    print(f"CSV saved to: {OUT_CSV}")
    print(f"JSON saved to: {OUT_JSON}")


if __name__ == "__main__":
    main()