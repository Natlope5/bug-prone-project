# app.py
# This file is the main backend for BugBuddy.
# It handles:
# - Loading env variables and the trained model
# - Analyzing uploaded code files with lizard
# - Calling OpenRouter through the OpenAI-compatible API
# - Serving the main HTML page and wiring up the chat + table

from dotenv import load_dotenv  # lets me keep secrets (API key, model name) in a .env file
from openai import OpenAI       # OpenAI-style client, but pointed at OpenRouter
from flask import Flask, render_template, request  # Flask for HTTP routes and template rendering

import os        # for environment variables and file cleanup
import pickle    # for loading the saved bug prediction model
import tempfile  # for safely storing the uploaded file while analyzing it
import time      # for small retry delays when trying fallback models
from pathlib import Path  # for cleaner path handling across platforms

import lizard    # library that extracts function-level metrics from code
import pandas as pd  # for organizing the function metrics as a DataFrame

# Actually load variables from the .env file into the process
load_dotenv()

# Read the OpenRouter API key from the environment.
# If it's blank, LLM-based answers will be disabled with a helpful message.
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")

# Main OpenRouter model name, defaulting to the free shared model
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openrouter/free")

# A small list of fallback models in case the first one is busy or invalid.
# I include the main model in here so my retry loop is simple.
OPENROUTER_FALLBACK_MODELS = [
    OPENROUTER_MODEL,
    "openrouter/free",
    "cohere/north-mini-code:free",
    "meta-llama/llama-3.3-70b-instruct:free",
]

# Create the OpenAI-compatible client,
# but send the traffic to OpenRouter instead of OpenAI.
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

# Base application directory so everything else can be relative to this file
APP_DIR = Path(__file__).resolve().parent

# Create the Flask app and tell it where templates and static files live.
# Static includes my BugBuddy pose images and the JS file.
app = Flask(
    __name__,
    template_folder=str(APP_DIR / "templates"),
    static_folder=str(APP_DIR / "static"),
    static_url_path="/static",
)

# Path to the saved model that only uses static metrics
STATIC_MODEL_PATH = APP_DIR.parent / "results" / "bug_model" / "static_only_model.pkl"

# Path to the saved combined model (has more inputs) if it exists
COMBINED_MODEL_PATH = APP_DIR.parent / "results" / "bug_model" / "combined_model.pkl"

# File types I allow people to upload.
# These all work well enough with lizard.
ALLOWED_EXTENSIONS = {
    ".py", ".js", ".ts", ".java", ".c", ".cpp", ".h", ".hpp",
    ".cs", ".go", ".php", ".rb", ".swift", ".kt", ".m", ".mm",
}

# Python files are special because they can use the trained ML model.
PYTHON_EXTENSIONS = {".py"}

# These globals remember the most recent uploaded file,
# so chat and quick questions stay in sync with it.
LAST_UPLOADED_ROWS = []
LAST_UPLOADED_FILENAME = ""
LAST_UPLOADED_LANGUAGE = ""
LAST_UPLOADED_CODE = ""


# Small helper that replaces NaN/missing values with a fallback
def clean_value(value, fallback=""):
    if pd.isna(value):
        return fallback
    return value


# Quick check to see if an uploaded filename has a supported extension
def allowed_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


# Detects the language based only on file extension
def detect_language(filename: str) -> str:
    ext = Path(filename).suffix.lower()
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


# Helper to check if a file is Python (eligible for the trained model)
def is_python_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in PYTHON_EXTENSIONS


# Clean up function names so anonymous/empty ones are readable
def normalize_function_name(name):
    if pd.isna(name) or not name:
        return "anonymous closure"

    text = str(name).strip()
    if text in {"(anonymous)", "<anonymous>", "anonymous"}:
        return "anonymous closure"

    return text


# Try to load a saved ML model from disk.
# If nothing is there, I fall back to rule-based scoring.
def load_saved_model():
    # Use combined model if present, otherwise static-only model.
    model_path = COMBINED_MODEL_PATH if COMBINED_MODEL_PATH.exists() else STATIC_MODEL_PATH

    if not model_path.exists():
        return None, [], 0.5, "Upload a file to begin. Python files use the saved model when available."

    try:
        with model_path.open("rb") as f:
            payload = pickle.load(f)

        model = payload.get("model")
        features = payload.get("features", [])
        threshold = float(payload.get("threshold", 0.5))
        status_label = "BugBuddy is ready to review your code and rank functions by risk."

        return model, features, threshold, status_label
    except Exception as e:
        return None, [], 0.5, f"Could not load saved model: {type(e).__name__}: {e}"


# Build a short “why” explanation for the table
def short_explanation(row):
    parts = []
    ccn = row.get("ccn")
    nloc = row.get("nloc")
    params = row.get("params")
    length = row.get("length")

    if isinstance(ccn, (int, float)):
        if ccn > 10:
            parts.append("High complexity")
        elif ccn >= 6:
            parts.append("Moderate branching")

    if isinstance(nloc, (int, float)):
        if nloc >= 20:
            parts.append("Long function body")
        elif nloc >= 10:
            parts.append("Moderate length")

    if isinstance(params, (int, float)) and params >= 4:
        parts.append("Many parameters")

    if isinstance(length, (int, float)) and length >= 30:
        parts.append("Large overall size")

    return "; ".join(parts) if parts else "Lower relative risk in this file"


# Friendlier explanation string used in chat replies
def explain_row_simple(row):
    prob = row.get("bug_probability")
    risk_label = row.get("risk_label", "")
    ccn = row.get("ccn")
    nloc = row.get("nloc")
    params = row.get("params")

    parts = []

    if isinstance(ccn, (int, float)):
        if ccn > 10:
            parts.append("the function has a lot of decision-making steps")
        elif ccn >= 6:
            parts.append("the function has more branching than a simple function")

    if isinstance(nloc, (int, float)):
        if nloc >= 20:
            parts.append("the function is fairly long")
        elif nloc >= 10:
            parts.append("the function is a moderate size")

    if isinstance(params, (int, float)) and params >= 4:
        parts.append("the function takes several inputs")

    base = "; ".join(parts) if parts else "the function looks simpler than the highest-ranked items in this file"

    # If I have a probability, explain in those terms
    if prob is not None:
        if risk_label == "High":
            chance = f"It is currently ranked in the high-risk group for this file, with probability {prob:.3f}."
        elif risk_label == "Medium":
            chance = f"It is currently ranked in the medium-risk group for this file, with probability {prob:.3f}."
        else:
            chance = f"It is currently ranked in the lower-risk group for this file, with probability {prob:.3f}."
    else:
        # No probability, just use the label
        if risk_label == "High":
            chance = "It looks high risk based on its complexity metrics."
        elif risk_label == "Medium":
            chance = "It looks moderately risky based on its complexity metrics."
        else:
            chance = "It does not look especially risky right now."

    return f"In simpler terms, {base}. {chance}"


# Short refactor suggestions for each function row
def fix_suggestion(row):
    ccn = row.get("ccn", 0) or 0
    nloc = row.get("nloc", 0) or 0
    params = row.get("params", 0) or 0
    length = row.get("length", 0) or 0

    tips = []

    if ccn > 10:
        tips.append("Break the function into smaller helper functions so each branch handles one clear job.")
    elif ccn >= 6:
        tips.append("Reduce branching by moving repeated condition logic into helper functions or early returns.")

    if nloc >= 20 or length >= 30:
        tips.append("Split long sections into smaller named functions to make the logic easier to test and maintain.")
    elif nloc >= 10 or length >= 15:
        tips.append("Trim the function by extracting setup, validation, or formatting code into separate helpers.")

    if params >= 4:
        tips.append("Reduce the parameter count by grouping related inputs into one object or struct.")
    elif params >= 2:
        tips.append("Check whether some inputs can be combined or derived inside the function.")

    if not tips:
        tips.append(
            "Review edge cases, naming, and repeated logic; this function may mainly need cleanup "
            "rather than a large refactor."
        )

    return "Suggested fix: " + " ".join(tips[:2])


# Simple rule-based system used when no ML model is available or for non-Python files
def rule_based_risk(row):
    score = 0
    ccn = float(row.get("ccn", 0) or 0)
    nloc = float(row.get("nloc", 0) or 0)
    params = float(row.get("params", 0) or 0)
    length = float(row.get("length", 0) or 0)

    if ccn > 10:
        score += 3
    elif ccn >= 6:
        score += 2
    elif ccn >= 4:
        score += 1

    if nloc >= 20:
        score += 2
    elif nloc >= 10:
        score += 1

    if params >= 4:
        score += 2
    elif params >= 2:
        score += 1

    if length >= 30:
        score += 2
    elif length >= 15:
        score += 1

    if score >= 6:
        return "High"
    if score >= 3:
        return "Medium"
    return "Low"


# Use lizard to extract metrics for every function in the uploaded file
def analyze_file_with_lizard(filepath: str):
    analysis = lizard.analyze_file(str(filepath))
    rows = []

    for func in analysis.function_list:
        # Parameter count may be on different attributes depending on the language
        params = getattr(func, "parameter_count", None)
        if params is None:
            try:
                params = len(func.parameters)
            except Exception:
                params = 0

        complexity = clean_value(getattr(func, "cyclomatic_complexity", 0), 0)
        start_line = clean_value(getattr(func, "start_line", 0), 0)
        end_line = clean_value(getattr(func, "end_line", 0), 0)

        # Try to estimate line span from start/end
        line_span = 0
        try:
            if start_line and end_line:
                line_span = int(end_line) - int(start_line) + 1
        except Exception:
            line_span = 0

        rows.append(
            {
                "file": Path(filepath).name,
                "function": normalize_function_name(clean_value(getattr(func, "name", ""), "")),
                "language": detect_language(filepath),
                "start_line": start_line,
                "end_line": end_line,
                "line_span": line_span,
                "nloc": clean_value(getattr(func, "nloc", 0), 0),
                "ccn": complexity,
                "token": clean_value(getattr(func, "token_count", 0), 0),
                "params": clean_value(params, 0),
                "length": clean_value(getattr(func, "length", 0), 0),
                "radon_cc": complexity,
                "is_test_file": 0,
            }
        )

    return rows


# Turn ML scores into High / Medium / Low bands relative to this file
def apply_relative_risk_bands(scored_rows):
    if not scored_rows:
        return scored_rows

    ranked = sorted(
        scored_rows,
        key=lambda row: row.get("bug_probability", -1) if row.get("bug_probability") is not None else -1,
        reverse=True,
    )

    total = len(ranked)
    high_cutoff = max(1, int(total * 0.20))   # top 20% high
    medium_cutoff = max(high_cutoff + 1, int(total * 0.50))  # next ~30% medium

    for idx, row in enumerate(ranked, start=1):
        row["rank"] = idx

        if idx <= high_cutoff:
            row["risk_label"] = "High"
            row["risk_class"] = "risk-high"
        elif idx <= medium_cutoff:
            row["risk_label"] = "Medium"
            row["risk_class"] = "risk-medium"
        else:
            row["risk_label"] = "Low"
            row["risk_class"] = "risk-low"

        prob = row.get("bug_probability")
        row["probability_text"] = f"{prob:.3f}" if prob is not None else "—"
        row["display_score"] = f"#{idx} · {row['probability_text']} · {row['risk_label']}"
        row["explanation"] = short_explanation(row)
        row["simple_explanation"] = explain_row_simple(row)
        row["fix_suggestion"] = fix_suggestion(row)

    return ranked


# Main scoring pipeline that runs after an upload
def score_uploaded_file(filepath, model, features, threshold, display_name=None):
    rows = analyze_file_with_lizard(filepath)

    if not rows:
        return [], "No functions were detected in the uploaded code file."

    shown_name = display_name if display_name else Path(filepath).name
    is_python = is_python_file(filepath)
    df = pd.DataFrame(rows)
    scored_rows = []

    # Python + model available → use ML predictions
    if is_python and model is not None and features:
        for feature in features:
            if feature not in df.columns:
                df[feature] = 0
            df[feature] = pd.to_numeric(df[feature], errors="coerce").fillna(0)

        probs = model.predict_proba(df[features])[:, 1]
        preds = (probs >= threshold).astype(int)

        for row, prob, pred in zip(df.to_dict("records"), probs, preds):
            row["file"] = shown_name
            row["function"] = normalize_function_name(row.get("function"))
            row["bug_probability"] = float(prob)
            row["predicted_bug_label"] = int(pred)
            row["risk_label"] = "Low"        # placeholder; re‑banded below
            row["risk_class"] = "risk-low"
            row["probability_text"] = f"{prob:.3f}"
            row["display_score"] = f"{prob:.3f}"
            row["explanation"] = ""
            row["simple_explanation"] = ""
            row["fix_suggestion"] = ""
            scored_rows.append(row)

        scored_rows = apply_relative_risk_bands(scored_rows)

    # Anything else → rule-based scoring
    else:
        for idx, row in enumerate(df.to_dict("records"), start=1):
            row["file"] = shown_name
            row["function"] = normalize_function_name(row.get("function"))
            row["bug_probability"] = None
            row["predicted_bug_label"] = None
            row["risk_label"] = rule_based_risk(row)
            row["risk_class"] = (
                "risk-high"
                if row["risk_label"] == "High"
                else "risk-medium"
                if row["risk_label"] == "Medium"
                else "risk-low"
            )
            row["rank"] = idx
            row["probability_text"] = "—"
            row["display_score"] = row["risk_label"]
            row["explanation"] = short_explanation(row)
            row["simple_explanation"] = explain_row_simple(row)
            row["fix_suggestion"] = fix_suggestion(row)
            scored_rows.append(row)

        # Make sure High comes before Medium before Low
        def fallback_sort_value(row):
            mapping = {"High": 3, "Medium": 2, "Low": 1}
            return mapping.get(row.get("risk_label", "Low"), 0)

        scored_rows.sort(key=fallback_sort_value, reverse=True)

        for idx, row in enumerate(scored_rows, start=1):
            row["rank"] = idx
            row["display_score"] = f"#{idx} · {row['risk_label']}"

    # Final sort: probability if present, otherwise by label score
    def sort_value(row):
        if row.get("bug_probability") is not None:
            return row["bug_probability"]
        mapping = {"High": 3, "Medium": 2, "Low": 1}
        return mapping.get(row.get("risk_label", "Low"), 0)

    scored_rows.sort(key=sort_value, reverse=True)
    return scored_rows, ""


# Extra headers recommended by OpenRouter docs (not required, but nice to include)
def build_openrouter_headers():
    return {
        "HTTP-Referer": "http://localhost:5001",
        "X-OpenRouter-Title": "BugBuddy",
    }


# Turn raw client exceptions into friendlier messages for the UI
def humanize_llm_error(last_error):
    text = str(last_error)
    lowered = text.lower()

    if "rate limit" in lowered or "429" in lowered:
        return (
            "BugBuddy reached OpenRouter, but the free model is temporarily busy right now. "
            "Please retry in about 30 seconds, or switch to another free model in your .env file."
        )

    if "not a valid model id" in lowered or "deprecated" in lowered or "404" in lowered or "400" in lowered:
        return (
            "BugBuddy reached OpenRouter, but the selected model name is invalid or no longer supported. "
            "Set OPENROUTER_MODEL to openrouter/free or cohere/north-mini-code:free and restart the app."
        )

    if "api key" in lowered or "401" in lowered or "403" in lowered:
        return (
            "BugBuddy could not authenticate with OpenRouter. "
            "Check that OPENROUTER_API_KEY in your .env file is correct."
        )

    return f"LLM request failed: {type(last_error).__name__}: {last_error}"


# Call OpenRouter to get a deeper explanation about the current file
def ask_llm_about_code(user_question, file_name, language_name, code_text, top_rows=None):
    if not OPENROUTER_API_KEY:
        return (
            "I can answer deeper code questions, but your OpenRouter API key is missing. "
            "Add OPENROUTER_API_KEY to your .env file first."
        )

    if not code_text.strip():
        return "I need the uploaded code text before I can answer detailed code questions."

    # Build a compact summary of the top-ranked functions for extra context
    top_summary = ""
    if top_rows:
        lines = []
        for row in top_rows[:5]:
            lines.append(
                f"- #{row.get('rank')} {row.get('function')} | risk={row.get('risk_label')} | "
                f"ccn={row.get('ccn')} | nloc={row.get('nloc')} | params={row.get('params')} | "
                f"start={row.get('start_line')} | end={row.get('end_line')}"
            )
        top_summary = "\n".join(lines)

    # Main prompt the LLM sees
    prompt = f"""
You are BugBuddy, a friendly code-review robot.

The user uploaded this file:
File name: {file_name}
Language: {language_name}

Top ranked functions from local analysis:
{top_summary if top_summary else 'No ranked functions available.'}

User question:
{user_question}

Code:
```{language_name.lower()}
{code_text[:20000]}
```

Instructions:
- Answer clearly and simply.
- Stay specific to the uploaded code.
- If you mention a bug, explain why.
- If the user asks for a fix, show steps or an example.
- If useful, suggest a safer or cleaner refactor.
"""

    last_error = None
    unique_models = []

    # De-duplicate fallback models while preserving order
    for model_name in OPENROUTER_FALLBACK_MODELS:
        if model_name and model_name not in unique_models:
            unique_models.append(model_name)

    # Try each model in order until one works or all fail
    for idx, model_name in enumerate(unique_models):
        try:
            response = client.chat.completions.create(
                model=model_name,
                extra_headers=build_openrouter_headers(),
                messages=[
                    {
                        "role": "system",
                        "content": "You are BugBuddy, a friendly coding assistant focused on uploaded code.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
            )

            content = response.choices[0].message.content.strip()

            # If a fallback model was used, mention its name in the reply for transparency
            if model_name != OPENROUTER_MODEL:
                return f"{content}\n\n(BugBuddy used fallback model: {model_name})"

            return content

        except Exception as e:
            last_error = e
            # If there are more models to try, wait briefly and continue
            if idx < len(unique_models) - 1:
                time.sleep(1.5)
                continue

    # All attempts failed → return a human-friendly summary
    return humanize_llm_error(last_error)


# Decide how to answer chat messages:
# - If it looks like a quick question, answer locally
# - Otherwise, ask the LLM about the current file
def chatbot_response(message, rows, features, language_name):
    global LAST_UPLOADED_CODE, LAST_UPLOADED_FILENAME

    msg = message.lower().strip()

    # No file yet → nothing to analyze
    if not rows:
        return "Please upload a supported code file first so I can analyze its functions."

    # “Why was this flagged?” style questions
    if "why" in msg or "flagged" in msg:
        top = rows[0]
        return (
            f"The highest-ranked function in your uploaded {language_name} file is "
            f"{top['function']} in {top['file']}.\n\n"
            f"Why it stands out: {top['explanation']}.\n\n"
            f"{top['simple_explanation']}\n\n"
            f"{top['fix_suggestion']}"
        )

    # Ask for high-risk functions
    if "high-risk" in msg or "high risk" in msg:
        high = [row for row in rows if row.get("risk_label") == "High"]

        if not high:
            return f"I didn't find any high-risk functions in your uploaded {language_name} file."

        response = f"I found {len(high)} high-risk functions in your uploaded {language_name} file:\n\n"
        for row in high[:10]:
            response += f"- #{row['rank']} {row['function']} ({row['probability_text']})\n"
        return response

    # Ask for medium-risk functions
    if "medium" in msg:
        med = [row for row in rows if row.get("risk_label") == "Medium"]

        if not med:
            return f"I didn't find any medium-risk functions in your uploaded {language_name} file right now."

        response = f"I found {len(med)} medium-risk functions in your uploaded {language_name} file:\n\n"
        for row in med[:10]:
            response += f"- #{row['rank']} {row['function']} ({row['probability_text']})\n"
        return response

    # Ask about model metrics/features
    if "metrics" in msg or "features" in msg:
        feature_text = ", ".join(features) if features else "no saved feature list found"
        return (
            f"I analyze your uploaded {language_name} file using function-level metrics such as "
            f"nloc, ccn, params, and length. When the saved model is available, I use these model "
            f"features: {feature_text}. I rank functions within the uploaded file so the riskiest "
            f"ones are easier to review first."
        )

    # Ask for “top” or “most bug-prone” functions
    if "top" in msg or "most bug" in msg or "most bug-prone" in msg or "most complex" in msg:
        top = rows[:5]
        response = f"Here are the top 5 highest-ranked functions in your uploaded {language_name} file:\n\n"
        for row in top:
            response += f"{row['rank']}. {row['function']} ({row['probability_text']}, {row['risk_label']})\n"
        return response

    # If it sounds like explanation / fix / bug help, send the whole thing to the LLM
    if any(
        word in msg
        for word in ["fix", "solution", "refactor", "rewrite", "what does", "how does", "bug", "error", "issue", "explain"]
    ):
        return ask_llm_about_code(
            user_question=message,
            file_name=LAST_UPLOADED_FILENAME,
            language_name=language_name,
            code_text=LAST_UPLOADED_CODE,
            top_rows=rows,
        )

    # Default fallback is also to ask the LLM
    return ask_llm_about_code(
        user_question=message,
        file_name=LAST_UPLOADED_FILENAME,
        language_name=language_name,
        code_text=LAST_UPLOADED_CODE,
        top_rows=rows,
    )


# Main route for the UI.
# Handles:
# - GET: quick preset questions
# - POST: chat messages and file uploads
@app.route("/", methods=["GET", "POST"])
def index():
    global LAST_UPLOADED_ROWS, LAST_UPLOADED_FILENAME, LAST_UPLOADED_LANGUAGE, LAST_UPLOADED_CODE

    # Load the saved model, feature list, threshold, and a status message
    model, train_features, threshold, model_status = load_saved_model()
    status_message = model_status
    context_note = ""

    # Default greeting shown in the conversation card
    bot_greeting = (
        "Hi! I'm BugBuddy. Upload a supported code file and I'll analyze its functions for bug risk. "
        "Python files use the saved trained model when available, while other languages use rule-based risk analysis. "
        "I can also answer deeper code questions through OpenRouter when your API key is set."
    )

    user_message = ""
    bot_reply = ""
    uploaded_rows = LAST_UPLOADED_ROWS if LAST_UPLOADED_ROWS else []
    bugbuddy_mode = "welcome"

    # If a file is already active, keep showing that context
    if LAST_UPLOADED_ROWS and LAST_UPLOADED_FILENAME:
        context_note = (
            f"Current uploaded file: {LAST_UPLOADED_FILENAME} ({LAST_UPLOADED_LANGUAGE}). "
            f"Quick buttons and chat use this uploaded file only."
        )

    if request.method == "POST":
        # Text chat form submit
        if "message" in request.form and request.form.get("message", "").strip():
            user_message = request.form.get("message", "").strip()
            language_name = LAST_UPLOADED_LANGUAGE if LAST_UPLOADED_LANGUAGE else "code"
            bot_reply = chatbot_response(user_message, uploaded_rows, train_features, language_name)
            bugbuddy_mode = "explaining"

        # File upload form submit
        elif "code_file" in request.files:
            uploaded_file = request.files["code_file"]

            if uploaded_file and uploaded_file.filename and allowed_file(uploaded_file.filename):
                temp_path = None

                try:
                    # Save upload to a temporary file for analysis
                    suffix = Path(uploaded_file.filename).suffix
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                        uploaded_file.save(tmp_file.name)
                        temp_path = tmp_file.name

                    # Read the uploaded code as text so the LLM can use it later
                    with open(temp_path, "r", encoding="utf-8", errors="ignore") as f:
                        uploaded_code_text = f.read()

                    # Run scoring pipeline
                    uploaded_rows, upload_message = score_uploaded_file(
                        temp_path,
                        model,
                        train_features,
                        threshold,
                        display_name=uploaded_file.filename,
                    )

                    detected_language = detect_language(uploaded_file.filename)
                    user_message = f"Please analyze my uploaded file: {uploaded_file.filename}"

                    if uploaded_rows:
                        # Save as the current active file
                        LAST_UPLOADED_ROWS = uploaded_rows
                        LAST_UPLOADED_FILENAME = uploaded_file.filename
                        LAST_UPLOADED_LANGUAGE = detected_language
                        LAST_UPLOADED_CODE = uploaded_code_text

                        context_note = (
                            f"Current uploaded file: {uploaded_file.filename} ({detected_language}). "
                            f"Quick buttons and chat use this uploaded file only."
                        )

                        # Build the first summary to show right after upload
                        top = uploaded_rows[0]
                        bot_reply = (
                            f"I analyzed {uploaded_file.filename} as a {detected_language} file and found "
                            f"{len(uploaded_rows)} functions.\n\n"
                            f"Top result: #{top['rank']} {top['function']} "
                            f"({top['probability_text']}, {top['risk_label']}).\n\n"
                            f"Why it stands out: {top['explanation']}.\n\n"
                            f"{top['fix_suggestion']}"
                        )
                        bugbuddy_mode = "explaining"

                    else:
                        # No functions found in the uploaded file
                        LAST_UPLOADED_ROWS = []
                        LAST_UPLOADED_FILENAME = ""
                        LAST_UPLOADED_LANGUAGE = ""
                        LAST_UPLOADED_CODE = ""
                        context_note = ""
                        bot_reply = upload_message or (
                            f"I analyzed {uploaded_file.filename}, but I did not detect any functions that could be scored."
                        )
                        bugbuddy_mode = "welcome"

                except Exception as e:
                    # Catch and show any upload/analysis errors in the chat
                    user_message = f"Please analyze my uploaded file: {uploaded_file.filename}"
                    bot_reply = f"Upload failed: {type(e).__name__}: {e}"
                    bugbuddy_mode = "welcome"

                finally:
                    # Always remove the temporary file
                    if temp_path and os.path.exists(temp_path):
                        os.remove(temp_path)

            else:
                # Unsupported file type or empty filename
                user_message = "Please analyze my uploaded file."
                bot_reply = (
                    "Please upload a supported code file. Supported extensions include .py, .js, .ts, "
                    ".java, .c, .cpp, .cs, .go, .php, .rb, .swift, and others."
                )
                bugbuddy_mode = "welcome"

    else:
        # Handle quick links like /?q=top or /?q=why on GET
        q = request.args.get("q", "")

        if q == "top":
            user_message = "Which functions are most bug-prone?"
        elif q == "high":
            user_message = "Show high-risk functions"
        elif q == "medium":
            user_message = "Show medium-risk functions"
        elif q == "metrics":
            user_message = "What metrics does the model use?"
        elif q == "why":
            user_message = "Why was something flagged?"

        if user_message:
            language_name = LAST_UPLOADED_LANGUAGE if LAST_UPLOADED_LANGUAGE else "code"
            bot_reply = chatbot_response(user_message, uploaded_rows, train_features, language_name)
            bugbuddy_mode = "explaining"
        elif uploaded_rows:
            # If a file is loaded but no new question was asked, Buddy is “ready”
            bugbuddy_mode = "ready"

    # Render my Jinja template instead of embedding HTML in Python
    return render_template(
        "index.html",
        bot_greeting=bot_greeting,
        user_message=user_message,
        bot_reply=bot_reply,
        uploaded_rows=uploaded_rows,
        status_message=status_message,
        context_note=context_note,
        bugbuddy_mode=bugbuddy_mode,

    )


# Standard Flask entry point
if __name__ == "__main__":
    # I disable the reloader so the global state (uploaded file) stays predictable.
    app.run(debug=True, use_reloader=False, use_debugger=False, port=5001)