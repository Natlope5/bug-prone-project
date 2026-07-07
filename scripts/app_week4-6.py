from flask import Flask, render_template_string, request
import os
import pickle
import tempfile
from pathlib import Path

import lizard
import pandas as pd

app = Flask(__name__)

ROOT = Path(__file__).resolve().parents[1]

STATIC_MODEL_PATH = ROOT / "results" / "bug_model" / "static_only_model.pkl"
COMBINED_MODEL_PATH = ROOT / "results" / "bug_model" / "combined_model.pkl"

ALLOWED_EXTENSIONS = {
    ".py", ".js", ".ts", ".java", ".c", ".cpp", ".h", ".hpp",
    ".cs", ".go", ".php", ".rb", ".swift", ".kt", ".m", ".mm"
}

PYTHON_EXTENSIONS = {".py"}

LAST_UPLOADED_ROWS = []
LAST_UPLOADED_FILENAME = ""
LAST_UPLOADED_LANGUAGE = ""

TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>BugBuddy</title>
  <style>
    body {
      font-family: "Segoe UI", system-ui, sans-serif;
      margin: 0;
      background: linear-gradient(180deg, #fdf6ff 0%, #f4f8ff 100%);
      color: #2b2b2b;
    }
    .container {
      max-width: 1200px;
      margin: 0 auto;
      padding: 2rem;
    }
    h1 { margin-bottom: 0.25rem; }
    h2 { margin-top: 0; }
    .subtitle { margin-bottom: 1.5rem; color: #666; }

    .layout {
      display: grid;
      grid-template-columns: 1fr;
      gap: 1.5rem;
    }

    .panel {
      background: white;
      border: 1px solid #e5dff0;
      border-radius: 20px;
      padding: 1.25rem;
      box-shadow: 0 8px 24px rgba(110, 92, 155, 0.08);
    }

    .bot-header {
      display: flex;
      align-items: center;
      gap: 0.75rem;
      margin-bottom: 1rem;
    }
    .bot-avatar {
      width: 56px;
      height: 56px;
      border-radius: 50%;
      background: linear-gradient(135deg, #c9b6ff, #aee1ff);
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 1.8rem;
    }
    .bot-name { font-weight: 700; font-size: 1.2rem; }
    .bot-role { color: #777; font-size: 0.95rem; }

    .chat-window {
      background: #fcfbff;
      border: 1px solid #ece7f5;
      border-radius: 18px;
      padding: 1rem;
      min-height: 220px;
      margin-bottom: 1rem;
    }

    .bubble {
      padding: 0.85rem 1rem;
      border-radius: 16px;
      max-width: 78%;
      margin-bottom: 0.8rem;
      line-height: 1.5;
      white-space: pre-wrap;
    }
    .bot-bubble {
      background: #efe7ff;
      border-top-left-radius: 6px;
    }
    .user-bubble {
      background: #dff3ff;
      border-top-right-radius: 6px;
      margin-left: auto;
    }

    form {
      display: flex;
      gap: 0.75rem;
      flex-wrap: wrap;
      margin-bottom: 1rem;
    }
    input[type="text"], input[type="file"] {
      flex: 1;
      min-width: 240px;
      padding: 0.85rem 1rem;
      border: 1px solid #d6d6e7;
      border-radius: 14px;
      font-size: 1rem;
      background: white;
    }
    button {
      padding: 0.85rem 1rem;
      border: none;
      border-radius: 14px;
      background: #7a5cff;
      color: white;
      cursor: pointer;
      font-weight: 600;
    }
    button:hover { background: #6948f5; }

    .quick-buttons {
      display: flex;
      gap: 0.5rem;
      flex-wrap: wrap;
      margin-top: 1rem;
    }
    .quick-buttons a {
      text-decoration: none;
      padding: 0.6rem 0.85rem;
      background: #f3efff;
      color: #4e3fa3;
      border-radius: 999px;
      font-size: 0.9rem;
      border: 1px solid #e4dafd;
    }

    table {
      border-collapse: collapse;
      width: 100%;
      margin-top: 1rem;
      font-size: 0.92rem;
      background: white;
    }
    th, td {
      border: 1px solid #eee;
      padding: 0.65rem;
      text-align: left;
      vertical-align: top;
    }
    th { background: #f5f3fb; }
    tr:nth-child(even) { background: #fcfcff; }

    .risk-high { background-color: #ffe4ea !important; }
    .risk-medium { background-color: #fff6dd !important; }
    .risk-low { background-color: #e8faec !important; }

    .badge {
      display: inline-block;
      padding: 0.2rem 0.55rem;
      border-radius: 999px;
      font-size: 0.82rem;
      font-weight: 600;
    }
    .badge-high { background: #ffd8e1; color: #8b1e3f; }
    .badge-medium { background: #fff0c2; color: #8a6400; }
    .badge-low { background: #dff5e4; color: #216b33; }

    .small-note {
      font-size: 0.9rem;
      color: #666;
      margin-top: 0.75rem;
    }

    .status-note {
      margin-top: 1rem;
      padding: 0.9rem 1rem;
      border-radius: 14px;
      background: #fff8dd;
      border: 1px solid #f1e2a4;
      color: #6c5a19;
      white-space: pre-wrap;
    }

    .context-note {
      margin-top: 0.75rem;
      font-size: 0.9rem;
      color: #5a4f7d;
      background: #f7f2ff;
      border: 1px solid #e7dcff;
      padding: 0.8rem 0.9rem;
      border-radius: 12px;
    }

    .helper-list {
      margin-top: 1rem;
      padding-left: 1.2rem;
      color: #555;
    }
  </style>
</head>
<body>
  <div class="container">
    <h1>🤖 BugBuddy</h1>
    <p class="subtitle">A multilingual HCI prototype for analyzing uploaded source code and surfacing bug-prone functions.</p>

    <div class="layout">
      <div class="panel">
        <div class="bot-header">
          <div class="bot-avatar">🤖</div>
          <div>
            <div class="bot-name">BugBuddy</div>
            <div class="bot-role">Your friendly code-risk helper</div>
          </div>
        </div>

        <div class="chat-window">
          <div class="bubble bot-bubble">{{ bot_greeting }}</div>
          {% if user_message %}
            <div class="bubble user-bubble">{{ user_message }}</div>
            <div class="bubble bot-bubble">{{ bot_reply }}</div>
          {% endif %}
        </div>

        <form method="post">
          <input type="text" name="message" placeholder="Ask BugBuddy something...">
          <button type="submit">Send</button>
        </form>

        <form method="post" enctype="multipart/form-data">
          <input type="file" name="code_file" required>
          <button type="submit">Upload Code File</button>
        </form>

        <div class="quick-buttons">
          <a href="/?q=top">Top 5 functions</a>
          <a href="/?q=high">High-risk functions</a>
          <a href="/?q=medium">Medium-risk functions</a>
          <a href="/?q=metrics">Model metrics</a>
          <a href="/?q=why">Why was something flagged?</a>
        </div>

        <ul class="helper-list">
          <li>Python uploads use the saved trained model when available.</li>
          <li>Other supported languages use rule-based scoring so the interface still works across languages.</li>
          <li>Results are ranked within the uploaded file so you can review the riskiest functions first.</li>
        </ul>

        <p class="small-note">
          Upload a supported code file first, then use the quick buttons to explore only that uploaded file.
        </p>

        {% if context_note %}
        <div class="context-note">{{ context_note }}</div>
        {% endif %}

        {% if status_message %}
        <div class="status-note">{{ status_message }}</div>
        {% endif %}
      </div>

      {% if uploaded_rows %}
      <div class="panel">
        <h2>Uploaded File Analysis</h2>
        <table>
          <thead>
            <tr>
              <th>Rank</th>
              <th>File</th>
              <th>Function</th>
              <th>Language</th>
              <th>Start</th>
              <th>End</th>
              <th>nloc</th>
              <th>ccn</th>
              <th>params</th>
              <th>Length</th>
              <th>Probability</th>
              <th>Risk</th>
              <th>Why it stands out</th>
            </tr>
          </thead>
          <tbody>
            {% for row in uploaded_rows %}
            <tr class="{{ row.risk_class }}">
              <td>{{ row.rank }}</td>
              <td>{{ row.file }}</td>
              <td>{{ row.function }}</td>
              <td>{{ row.language }}</td>
              <td>{{ row.start_line }}</td>
              <td>{{ row.end_line }}</td>
              <td>{{ row.nloc }}</td>
              <td>{{ row.ccn }}</td>
              <td>{{ row.params }}</td>
              <td>{{ row.length }}</td>
              <td>{{ row.probability_text }}</td>
              <td>
                <span class="badge {% if row.risk_label == 'High' %}badge-high{% elif row.risk_label == 'Medium' %}badge-medium{% else %}badge-low{% endif %}">
                  {{ row.risk_label }}
                </span>
              </td>
              <td>{{ row.explanation }}</td>
            </tr>
            {% endfor %}
          </tbody>
        </table>
      </div>
      {% endif %}
    </div>
  </div>
</body>
</html>
"""


def clean_value(value, fallback=""):
    if pd.isna(value):
        return fallback
    return value


def allowed_file(filename):
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def detect_language(filename):
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


def is_python_file(filename):
    return Path(filename).suffix.lower() in PYTHON_EXTENSIONS


def normalize_function_name(name):
    if pd.isna(name) or not name:
        return "anonymous closure"

    text = str(name).strip()
    if text in {"(anonymous)", "<anonymous>", "anonymous"}:
        return "anonymous closure"

    return text


def load_saved_model():
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

    if prob is not None:
        if risk_label == "High":
            chance = f"It is currently ranked in the high-risk group for this file, with probability {prob:.3f}."
        elif risk_label == "Medium":
            chance = f"It is currently ranked in the medium-risk group for this file, with probability {prob:.3f}."
        else:
            chance = f"It is currently ranked in the lower-risk group for this file, with probability {prob:.3f}."
    else:
        if risk_label == "High":
            chance = "It looks high risk based on its complexity metrics."
        elif risk_label == "Medium":
            chance = "It looks moderately risky based on its complexity metrics."
        else:
            chance = "It does not look especially risky right now."

    return f"In simpler terms, {base}. {chance}"


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


def analyze_file_with_lizard(filepath):
    analysis = lizard.analyze_file(str(filepath))
    rows = []

    for func in analysis.function_list:
        params = getattr(func, "parameter_count", None)
        if params is None:
            try:
                params = len(func.parameters)
            except Exception:
                params = 0

        complexity = clean_value(getattr(func, "cyclomatic_complexity", 0), 0)
        start_line = clean_value(getattr(func, "start_line", 0), 0)
        end_line = clean_value(getattr(func, "end_line", 0), 0)

        line_span = 0
        try:
            if start_line and end_line:
                line_span = int(end_line) - int(start_line) + 1
        except Exception:
            line_span = 0

        rows.append({
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
        })

    return rows


def apply_relative_risk_bands(scored_rows):
    if not scored_rows:
        return scored_rows

    ranked = sorted(
        scored_rows,
        key=lambda row: row.get("bug_probability", -1) if row.get("bug_probability") is not None else -1,
        reverse=True,
    )

    total = len(ranked)
    high_cutoff = max(1, int(total * 0.20))
    medium_cutoff = max(high_cutoff + 1, int(total * 0.50))

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

    return ranked


def score_uploaded_file(filepath, model, features, threshold, display_name=None):
    rows = analyze_file_with_lizard(filepath)

    if not rows:
        return [], "No functions were detected in the uploaded code file."

    shown_name = display_name if display_name else Path(filepath).name
    is_python = is_python_file(filepath)
    df = pd.DataFrame(rows)
    scored_rows = []

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
            row["risk_label"] = "Low"
            row["risk_class"] = "risk-low"
            row["probability_text"] = f"{prob:.3f}"
            row["display_score"] = f"{prob:.3f}"
            row["explanation"] = ""
            row["simple_explanation"] = ""
            scored_rows.append(row)

        scored_rows = apply_relative_risk_bands(scored_rows)

    else:
        for idx, row in enumerate(df.to_dict("records"), start=1):
            row["file"] = shown_name
            row["function"] = normalize_function_name(row.get("function"))
            row["bug_probability"] = None
            row["predicted_bug_label"] = None
            row["risk_label"] = rule_based_risk(row)
            row["risk_class"] = (
                "risk-high" if row["risk_label"] == "High"
                else "risk-medium" if row["risk_label"] == "Medium"
                else "risk-low"
            )
            row["rank"] = idx
            row["probability_text"] = "—"
            row["display_score"] = row["risk_label"]
            row["explanation"] = short_explanation(row)
            row["simple_explanation"] = explain_row_simple(row)
            scored_rows.append(row)

        def fallback_sort_value(row):
            mapping = {"High": 3, "Medium": 2, "Low": 1}
            return mapping.get(row.get("risk_label", "Low"), 0)

        scored_rows.sort(key=fallback_sort_value, reverse=True)

        for idx, row in enumerate(scored_rows, start=1):
            row["rank"] = idx
            row["display_score"] = f"#{idx} · {row['risk_label']}"

    def sort_value(row):
        if row.get("bug_probability") is not None:
            return row["bug_probability"]
        mapping = {"High": 3, "Medium": 2, "Low": 1}
        return mapping.get(row.get("risk_label", "Low"), 0)

    scored_rows.sort(key=sort_value, reverse=True)
    return scored_rows, ""


def chatbot_response(message, rows, features, language_name):
    msg = message.lower().strip()

    if not rows:
        return "Please upload a supported code file first so I can analyze its functions."

    if "why" in msg or "explain" in msg or "flagged" in msg:
        top = rows[0]
        return (
            f"The highest-ranked function in your uploaded {language_name} file is "
            f"{top['function']} in {top['file']}.\n\n"
            f"Why it stands out: {top['explanation']}.\n\n"
            f"{top['simple_explanation']}"
        )

    if "high-risk" in msg or "high risk" in msg:
        high = [row for row in rows if row.get("risk_label") == "High"]
        if not high:
            return f"I didn’t find any high-risk functions in your uploaded {language_name} file."
        response = f"I found {len(high)} high-risk functions in your uploaded {language_name} file:\n"
        for row in high[:10]:
            response += f"- #{row['rank']} {row['function']} ({row['probability_text']})\n"
        return response

    if "medium" in msg:
        med = [row for row in rows if row.get("risk_label") == "Medium"]
        if not med:
            return f"I didn’t find any medium-risk functions in your uploaded {language_name} file right now."
        response = f"I found {len(med)} medium-risk functions in your uploaded {language_name} file:\n"
        for row in med[:10]:
            response += f"- #{row['rank']} {row['function']} ({row['probability_text']})\n"
        return response

    if "metrics" in msg or "features" in msg:
        feature_text = ", ".join(features) if features else "no saved feature list found"
        return (
            f"I analyze your uploaded {language_name} file using function-level metrics such as "
            f"nloc, ccn, params, and length. When the saved model is available, I use these model features: "
            f"{feature_text}. I rank functions within the uploaded file so the riskiest ones are easier to review first."
        )

    if "top" in msg or "most bug" in msg or "most bug-prone" in msg or "most complex" in msg:
        top = rows[:5]
        response = f"Here are the top 5 highest-ranked functions in your uploaded {language_name} file:\n"
        for row in top:
            response += f"{row['rank']}. {row['function']} ({row['probability_text']}, {row['risk_label']})\n"
        return response

    return "Ask about the top functions, high-risk functions, medium-risk functions, model metrics, or why something was flagged."


@app.route("/", methods=["GET", "POST"])
def index():
    global LAST_UPLOADED_ROWS, LAST_UPLOADED_FILENAME, LAST_UPLOADED_LANGUAGE

    model, train_features, threshold, model_status = load_saved_model()
    status_message = model_status
    context_note = ""

    bot_greeting = (
        "Hi! I’m BugBuddy 🤖\n"
        "Upload a supported code file and I’ll analyze its functions for bug risk.\n"
        "Python files use the saved trained model when available, while other languages use rule-based risk analysis.\n"
        "I rank functions within your uploaded file so you can review the riskiest code first."
    )

    user_message = ""
    bot_reply = ""
    uploaded_rows = LAST_UPLOADED_ROWS[:] if LAST_UPLOADED_ROWS else []

    if LAST_UPLOADED_ROWS and LAST_UPLOADED_FILENAME:
        context_note = (
            f"Current uploaded file: {LAST_UPLOADED_FILENAME} "
            f"({LAST_UPLOADED_LANGUAGE}). Quick buttons use this uploaded file only."
        )

    if request.method == "POST":
        if "message" in request.form and request.form.get("message", "").strip():
            user_message = request.form.get("message", "")
            language_name = LAST_UPLOADED_LANGUAGE if LAST_UPLOADED_LANGUAGE else "code"
            bot_reply = chatbot_response(user_message, uploaded_rows, train_features, language_name)

        elif "code_file" in request.files:
            uploaded_file = request.files["code_file"]

            if uploaded_file and uploaded_file.filename and allowed_file(uploaded_file.filename):
                temp_path = None
                try:
                    suffix = Path(uploaded_file.filename).suffix
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                        uploaded_file.save(temp_file.name)
                        temp_path = temp_file.name

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
                        LAST_UPLOADED_ROWS = uploaded_rows[:]
                        LAST_UPLOADED_FILENAME = uploaded_file.filename
                        LAST_UPLOADED_LANGUAGE = detected_language
                        context_note = (
                            f"Current uploaded file: {uploaded_file.filename} "
                            f"({detected_language}). Quick buttons use this uploaded file only."
                        )

                        top = uploaded_rows[0]
                        bot_reply = (
                            f"I analyzed {uploaded_file.filename} as a {detected_language} file and found "
                            f"{len(uploaded_rows)} functions.\n\n"
                            f"Top result: #{top['rank']} {top['function']} "
                            f"({top['probability_text']}, {top['risk_label']}).\n"
                            f"Why it stands out: {top['explanation']}."
                        )
                    else:
                        LAST_UPLOADED_ROWS = []
                        LAST_UPLOADED_FILENAME = ""
                        LAST_UPLOADED_LANGUAGE = ""
                        context_note = ""
                        bot_reply = upload_message or (
                            f"I analyzed {uploaded_file.filename}, but I did not detect any functions "
                            f"that could be scored."
                        )

                except Exception as e:
                    user_message = f"Please analyze my uploaded file: {uploaded_file.filename}"
                    bot_reply = f"Upload failed: {type(e).__name__}: {e}"

                finally:
                    if temp_path and os.path.exists(temp_path):
                        os.remove(temp_path)
            else:
                user_message = "Please analyze my uploaded file."
                bot_reply = (
                    "Please upload a supported code file. Supported extensions include "
                    ".py, .js, .ts, .java, .c, .cpp, .cs, .go, .php, .rb, .swift, and others."
                )

    else:
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

    return render_template_string(
        TEMPLATE,
        bot_greeting=bot_greeting,
        user_message=user_message,
        bot_reply=bot_reply,
        uploaded_rows=uploaded_rows,
        status_message=status_message,
        context_note=context_note,
    )


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False, use_debugger=False, port=5001)