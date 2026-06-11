\# Bug-Prone Project



This project explores how to extract function-level software metrics from Python code and use them to study bug-prone functions in student or GitHub-style repositories.



\## Project Goal



The main goal of this project is to build a small pilot dataset of Python functions and their complexity-related metrics. These metrics can later be used to help identify which functions may be more likely to contain bugs.



\## What the Script Does



The project includes a Python script, `scripts/extract\_metrics.py`, that uses the Radon library to analyze Python files. The script:



\- accepts one or more Python files or folders as input

\- recursively searches folders for `.py` files

\- extracts function-level metrics such as file path, function name, line numbers, length, cyclomatic complexity, and Radon grade

\- saves the results to `data/pilot\_metrics.csv`

\- saves a summary report to `results/summary.json`



\## Repositories Analyzed



For the pilot run, the script was used on Python repositories stored under `data/repos/`, including:



\- `Python`

\- `python-3-playlist`

\- `python-calistus`



\## Example Command



Run the script from the project root like this:



```bash

py "scripts\\extract\_metrics.py" "data\\repos\\python-calistus" "data\\repos\\python-3-playlist" "data\\repos\\Python"

```



\## Output Files



\- `data/pilot\_metrics.csv` — function-level metrics dataset

\- `results/summary.json` — JSON summary of analyzed files and functions



\## Notes



Some local IDE folders and duplicate output folders are ignored through `.gitignore` so they are not pushed to GitHub.

