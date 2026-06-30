# Bug-Prone Project

This project explores how to extract function-level software metrics from source code, build a pilot bug-prediction dataset, train machine-learning models, and expose the results through a small web-based prototype called **BugBuddy**.

The project began as a Python-only metrics experiment, but it now supports a broader static-analysis workflow for multiple languages using Lizard-based extraction, optional bug labels, model training with scikit-learn, and upload-based prediction through a Flask interface.

## Project Goal

The main goal of this project is to study whether function-level static code metrics can help identify functions that may be more likely to contain bugs, especially in student-style or GitHub-style repositories.

A second goal is to turn those metrics into an interactive prototype that lets a user upload a code file, see the riskiest functions, and read plain-language explanations of why those functions were flagged.

## Current Workflow

The project currently follows this workflow:

1. Extract function-level static metrics from one or more repositories or code files.
2. Build or update a pilot dataset in `data/pilot_metrics.csv`.
3. Add or refine `bug_label` values for the rows that will be used for supervised learning.
4. Train a bug-prediction model using the labeled rows and evaluate it with ROC-style metrics and related outputs.
5. Load the saved model into the BugBuddy Flask app so uploaded files can be scored and explained in the browser.

## Main Scripts

### `scripts/extract_metrics.py`

This script scans one or more files or folders, recursively finds supported source-code files, and extracts function-level metrics using **Lizard**.

It currently supports multiple languages, including Python, JavaScript, TypeScript, Java, C, C++, C#, Go, PHP, Ruby, Swift, Kotlin, and Objective-C style files.

For each detected function, the script records values such as:

- file path
- language
- function name
- start and end line
- `nloc`
- `ccn`
- token count
- parameter count
- length
- `radon_cc` placeholder
- `bug_label` for later annotation

The script saves:

- `data/pilot_metrics.csv` — the extracted function-level dataset
- `results/summary.json` — a summary of analyzed files, skipped files, and total function counts

### Model-training scripts

The project also includes training and evaluation code for turning labeled metrics into bug-probability predictions.

The model-training stage uses labeled rows from `data/pilot_metrics.csv`, trains a classifier such as a Random Forest model, and evaluates it with ROC/AUC-style outputs before saving model artifacts for later reuse in the web app.

### `app_week4-6.py`

This is the Flask-based BugBuddy interface.

The web app allows a user to:

- upload a supported code file
- analyze its functions
- rank them by bug probability or fallback risk
- ask quick questions such as “Top 5 functions” or “Why was this flagged?”
- read plain-language explanations of the underlying metrics

When a saved trained model is available, Python uploads can use that model for scoring; otherwise, the interface can fall back to rule-based scoring for broader language support.

## Example Metric Extraction Commands

Run the extraction script from the project root like this on Windows:

```bash
py "scripts\extract_metrics.py" "data\repos\python-calistus" "data\repos\python-3-playlist" "data\repos\Python"
```

You can also scan a whole folder of repositories:

```bash
py "scripts\extract_metrics.py" "data\repos"
```

Or scan a single repository or source file:

```bash
py "scripts\extract_metrics.py" "data\repos\some-repo"
```

These commands work because the extraction script accepts one or more files or folders as input and recursively searches supported source-code files inside them.

## Output Files

Common project outputs include:

- `data/pilot_metrics.csv` — function-level metrics dataset used for labeling and training
- `results/summary.json` — extraction summary showing analyzed and skipped files
- model evaluation outputs such as prediction summaries, ROC artifacts, or training results generated during the modeling phase
- saved model files used later by the BugBuddy web interface

## Repositories Analyzed

For the pilot phase, the project used repositories stored under `data/repos/`, including examples such as:

- `Python`
- `python-3-playlist`
- `python-calistus`

These repositories were used to create the first pilot metrics dataset and test the function-extraction pipeline before expanding the project workflow into training and web upload analysis.

## Metrics Used

The project focuses mainly on lightweight function-level static metrics because they are simple to extract, easy to explain, and practical for educational or prototype use cases.

Examples of metrics used in the dataset and model pipeline include:

- `nloc` for non-comment lines of code
- `ccn` for cyclomatic complexity
- `params` for parameter count
- `length` for overall function size
- token-related counts
- placeholders such as `radon_cc` depending on the extraction workflow

## Notes and Limitations

This is still a pilot project, so the dataset is relatively small and some bug labels may be weak, partial, or manually assigned rather than coming from a large verified defect corpus.

That means early results are best interpreted as proof that the extraction, training, and upload-analysis pipeline works, rather than as final evidence of real-world defect-prediction performance.

Some local IDE folders, generated results, and duplicate output folders may be ignored through `.gitignore` so they are not pushed to GitHub unnecessarily.

## Next Steps

Planned next steps for the project include:

- expanding the labeled dataset
- improving the labeling strategy
- comparing additional models
- refining multi-language support
- strengthening the BugBuddy interface for instructor or student feedback workflows

The long-term goal is to provide a lightweight educational tool that helps users focus code review on the functions most likely to deserve extra attention.