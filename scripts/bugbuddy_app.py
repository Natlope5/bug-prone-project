# Loads variables from the .env file so I can keep the API key and model name outside the code
from dotenv import load_dotenv

# OpenAI client is used here because OpenRouter supports the OpenAI-compatible API format
from openai import OpenAI

# Flask handles the website, page rendering, and form requests
from flask import Flask, render_template_string, request

# os helps read environment variables and work with files
import os

# pickle is used to load the saved bug prediction model
import pickle

# tempfile is used when a user uploads a file so I can safely analyze it
import tempfile

# time is used for small retry delays when trying fallback models
import time

# Path makes file paths easier to build and read
from pathlib import Path

# Lizard is used to analyze code and get function-level metrics
import lizard

# pandas is used to organize function data before scoring it
import pandas as pd

# Actually loads the .env file into the program
load_dotenv()

# Gets the OpenRouter API key from the .env file
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")

# Gets the main model name from the .env file
# Defaults to openrouter/free if nothing is set
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openrouter/free")

# These are fallback models in case the first model is busy or fails
OPENROUTER_FALLBACK_MODELS = [
    OPENROUTER_MODEL,
    "openrouter/free",
    "cohere/north-mini-code:free",
    "meta-llama/llama-3.3-70b-instruct:free",
]

# Creates the OpenAI-compatible client pointed at OpenRouter instead of OpenAI
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

# Gets the folder this app.py file is in
APP_DIR = Path(__file__).resolve().parent

# Creates the Flask app and tells it where the static files are
# Static files include the BugBuddy pose images
app = Flask(
    __name__,
    static_folder=str(APP_DIR / "static"),
    static_url_path="/static"
)

# Path to the saved model that uses static metrics only
STATIC_MODEL_PATH = APP_DIR.parent / "results" / "bug_model" / "static_only_model.pkl"

# Path to the saved combined model if that file exists
COMBINED_MODEL_PATH = APP_DIR.parent / "results" / "bug_model" / "combined_model.pkl"

# File types the app allows people to upload
ALLOWED_EXTENSIONS = {
    ".py", ".js", ".ts", ".java", ".c", ".cpp", ".h", ".hpp",
    ".cs", ".go", ".php", ".rb", ".swift", ".kt", ".m", ".mm"
}

# Python files are treated a little differently because they can use the trained model
PYTHON_EXTENSIONS = {".py"}

# These global variables hold the most recent uploaded file data
# That way quick questions and follow-up chat still know what file to talk about
LAST_UPLOADED_ROWS = []
LAST_UPLOADED_FILENAME = ""
LAST_UPLOADED_LANGUAGE = ""
LAST_UPLOADED_CODE = ""

# This is the full HTML template for the BugBuddy website
# It includes the page design, the upload form, the table, the chatbot, and the animated Buddy area
TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
  <!-- Sets the character encoding -->
  <meta charset="UTF-8" />

  <!-- Makes the layout scale correctly on phones and smaller screens -->
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />

  <!-- Title shown in the browser tab -->
  <title>BugBuddy</title>

  <!-- Preconnects to Google Fonts for faster loading -->
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>

  <!-- Imports the fonts used in the interface -->
  <link href="https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700;800&family=Space+Grotesk:wght@500;700&display=swap" rel="stylesheet">

  <style>
    /* Main color variables used throughout the design */
    :root {
      --bg: #07070a;
      --surface: rgba(14,14,18,.92);
      --border: rgba(255, 111, 181, 0.16);
      --text: #fff3f8;
      --muted: #f2a8ca;
      --primary: #ff67b1;
      --primary-strong: #ff2d93;
      --primary-soft: rgba(255, 103, 177, 0.14);
      --shadow: 0 22px 60px rgba(0,0,0,.42);
    }

    /* Makes padding and borders easier to size consistently */
    * { box-sizing: border-box; }

    /* Smooth scrolling for the guided Buddy tour */
    html { scroll-behavior: smooth; }

    /* Main page styling */
    body {
      margin: 0;
      font-family: 'Nunito', sans-serif;
      color: var(--text);
      background:
        radial-gradient(circle at top left, rgba(255, 81, 162, 0.13), transparent 28%),
        radial-gradient(circle at top right, rgba(255, 133, 201, 0.10), transparent 24%),
        linear-gradient(180deg, #050507 0%, #0c0c11 100%);
      min-height: 100vh;
    }

    /* Centers the whole page content and gives it spacing */
    .shell { max-width: 1450px; margin: 0 auto; padding: 24px; }

    /* Top bar layout for logo and status pill */
    .topbar { display: flex; justify-content: space-between; align-items: center; gap: 16px; margin-bottom: 24px; }

    /* Brand/logo area */
    .brand { display: flex; align-items: center; gap: 14px; }

    /* Pink icon box behind the BugBuddy logo */
    .brand-mark {
      width: 52px; height: 52px; border-radius: 18px; display: grid; place-items: center;
      background: linear-gradient(145deg, #ff93c8, #ff429f);
      box-shadow: 0 14px 32px rgba(255, 80, 163, 0.28);
    }

    /* Logo SVG size */
    .brand-mark svg { width: 30px; height: 30px; }

    /* Title text next to the logo */
    .brand-text h1 { margin: 0; font-family: 'Space Grotesk', sans-serif; font-size: 1.35rem; letter-spacing: -0.04em; }

    /* Smaller label under the title */
    .brand-text p { margin: 2px 0 0; color: var(--muted); font-size: 0.92rem; font-weight: 700; }

    /* Small top-right pill */
    .top-pill {
      padding: 10px 14px; border-radius: 999px; background: rgba(14,14,18,.72);
      border: 1px solid var(--border); color: var(--muted); font-size: 0.9rem; white-space: nowrap;
    }

    /* Main layout grid */
    .layout { display: grid; grid-template-columns: 1fr; gap: 24px; align-items: start; }

    /* Vertical stack for sections */
    .stack { display: grid; gap: 24px; }

    /* Shared panel/card styling */
    .panel {
      background: rgba(11,11,15,.86); border: 1px solid rgba(255, 111, 181, 0.12);
      border-radius: 30px; box-shadow: var(--shadow); backdrop-filter: blur(18px);
    }

    /* Hero section at the top */
    .hero {
      display: grid; grid-template-columns: 1fr; gap: 20px; padding: 28px; min-height: 450px;
      overflow: hidden; position: relative;
    }

    /* Small label above the heading */
    .eyebrow {
      display: inline-flex; align-items: center; gap: 8px; padding: 10px 14px; border-radius: 999px;
      background: var(--primary-soft); border: 1px solid rgba(255, 111, 181, 0.18);
      color: #ff93c8; font-size: 0.85rem; font-weight: 800; margin-bottom: 16px;
    }

    /* Main page heading */
    .hero h2 {
      margin: 0 0 14px; font-family: 'Space Grotesk', sans-serif; font-size: clamp(2.2rem, 4vw, 4.2rem);
      line-height: 0.96; letter-spacing: -0.05em; max-width: 10ch;
    }

    /* Supporting paragraph under the heading */
    .hero p.lead { margin: 0 0 20px; color: var(--muted); max-width: 56ch; font-size: 1.03rem; line-height: 1.6; }

    /* Layout for the ask-BugBuddy question form */
    .hero-form {
      display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 16px; position: relative;
      scroll-margin-top: 110px; transition: transform .35s ease, box-shadow .35s ease, border-color .35s ease;
    }

    /* Shared styling for text and file inputs */
    input[type="text"], input[type="file"] {
      flex: 1; min-width: 220px; padding: 13px 15px; border-radius: 16px;
      border: 1px solid var(--border); background: rgba(17,17,22,.92); color: var(--text); font: inherit;
    }

    /* Makes the file input stretch all the way across */
    input[type="file"] { min-width: 100%; }

    /* Shared button styling */
    button {
      border: 0; border-radius: 16px; padding: 13px 18px; font-weight: 800; font-family: inherit;
      cursor: pointer; transition: transform .2s ease, box-shadow .2s ease, background .2s ease;
    }

    /* Small lift effect on hover */
    button:hover { transform: translateY(-2px); }

    /* Main pink button */
    .primary-btn {
      background: linear-gradient(145deg, var(--primary), var(--primary-strong));
      color: white; box-shadow: 0 14px 28px rgba(255, 77, 158, 0.24);
    }

    /* Dark outlined button */
    .ghost-btn { background: rgba(18,18,22,.92); border: 1px solid var(--border); color: var(--text); }

    /* Voice toggle button styling */
    .voice-toggle {
      display: inline-flex; align-items: center; gap: 8px; margin-bottom: 16px;
      background: rgba(18,18,22,.92); border: 1px solid var(--border); color: var(--text);
    }

    /* Upload area styling */
    .upload-box {
      padding: 18px; border: 2px dashed rgba(255,111,181,.22); border-radius: 24px;
      background: linear-gradient(180deg, rgba(16,16,20,.95), rgba(28,16,25,.94)); margin: 14px 0 18px;
      position: relative; scroll-margin-top: 110px;
      transition: transform .35s ease, box-shadow .35s ease, border-color .35s ease;
    }

    /* Upload title */
    .upload-box strong { display: block; margin-bottom: 6px; }

    /* Upload helper text */
    .upload-box small { color: var(--muted); display: block; margin-bottom: 12px; }

    /* Highlight effect used during the guided intro */
    .tour-focus {
      border-color: rgba(255, 111, 181, 0.65) !important;
      box-shadow: 0 0 0 4px rgba(255, 103, 177, 0.16), 0 18px 40px rgba(255, 77, 158, 0.20);
      transform: translateY(-2px) scale(1.01);
    }

    /* Container that holds BugBuddy and the speech bubble */
    #bugBuddyStage {
      position: absolute; left: 430px; top: 88px; width: 260px; height: 360px; z-index: 9999;
      pointer-events: none;
      transition: left 1.2s cubic-bezier(.22,1,.36,1), top 1.2s cubic-bezier(.22,1,.36,1),
                  bottom 1.2s cubic-bezier(.22,1,.36,1), transform .6s ease;
    }

    /* Speech bubble next to BugBuddy */
    .speech {
      position: absolute; left: 205px; top: -6px; width: 235px; background: rgba(12,12,16,.97);
      border: 1px solid rgba(255,111,181,.22); border-radius: 22px; padding: 16px 16px 18px;
      box-shadow: 0 14px 30px rgba(0,0,0,.3); z-index: 4; animation: bubbleIn .45s cubic-bezier(.16,1,.3,1);
    }

    /* Small arrow shape under the speech bubble */
    .speech::after {
      content: ''; position: absolute; left: -8px; top: 78px; width: 18px; height: 18px;
      background: rgba(12,12,16,.97); border-left: 1px solid rgba(255,111,181,.22);
      border-bottom: 1px solid rgba(255,111,181,.22); transform: rotate(45deg);
    }

    /* Label inside the speech bubble */
    .speech .name {
      font-size: 13px; text-transform: uppercase; letter-spacing: .08em; color: #ff93c8;
      font-weight: 800; margin-bottom: 8px;
    }

    /* Speech text styling */
    .speech p { margin: 0; font-size: 0.98rem; line-height: 1.45; }

    /* Adds animated dots when BugBuddy is talking */
    .speech.talking p::after { content: " • • •"; color: #ff93c8; animation: dots 1.1s steps(3, end) infinite; }

    /* Wrap for all the different BugBuddy pose images */
    .robot-wrap { position: absolute; inset: 0; width: 100%; height: 100%; z-index: 2; }

    /* All Buddy images start hidden except the active one */
    .robot-pose {
      position: absolute; inset: 0; width: 100%; height: 100%; object-fit: contain; opacity: 0;
      transform: translateY(8px) scale(.985); transition: opacity .35s ease, transform .35s ease;
      filter: drop-shadow(0 24px 42px rgba(255, 83, 166, 0.18)); pointer-events: none;
    }

    /* The currently active pose becomes visible */
    .robot-pose.active { opacity: 1; transform: translateY(0) scale(1); }

    /* Dashboard section that holds the uploaded file analysis table */
    .dashboard { padding: 24px; }

    /* Header row for sections */
    .section-head { display: flex; justify-content: space-between; align-items: center; gap: 12px; margin-bottom: 16px; flex-wrap: wrap; }

    /* Section title styling */
    .section-head h3 { margin: 0; font-family: 'Space Grotesk', sans-serif; font-size: 1.3rem; }

    /* Generic rounded pill */
    .pill { padding: 10px 14px; border-radius: 999px; background: rgba(17,17,22,.92); border: 1px solid var(--border); color: var(--muted); font-size: 0.9rem; }

    /* Row of smaller status chips */
    .status-line { display: flex; gap: 10px; flex-wrap: wrap; margin-bottom: 16px; }

    /* Default chip styling */
    .status-chip {
      padding: 10px 12px; border-radius: 999px; background: rgba(18,18,22,.96);
      border: 1px solid rgba(255,111,181,.16); color: var(--muted); font-size: 0.88rem;
    }

    /* Active status chip styling */
    .status-chip.active {
      background: linear-gradient(145deg, var(--primary), var(--primary-strong));
      color: white; box-shadow: 0 8px 20px rgba(255,77,158,.24);
    }

    /* Container around the results table so it can scroll horizontally if needed */
    .table-wrap { overflow: auto; border-radius: 24px; border: 1px solid var(--border); background: rgba(12,12,16,.92); }

    /* Table setup */
    table { border-collapse: collapse; width: 100%; min-width: 880px; }

    /* Table cell spacing */
    th, td { padding: 14px 16px; text-align: left; vertical-align: top; }

    /* Header cell styling */
    th {
      font-size: 12px; text-transform: uppercase; letter-spacing: .08em;
      color: var(--muted); background: rgba(24,17,24,.98);
    }

    /* Row separators */
    tr + tr td { border-top: 1px solid rgba(255,111,181,.08); }

    /* Hover effect for rows */
    tr:hover { background: rgba(30,19,27,.95); }

    /* Shared styling for risk badges */
    .risk {
      display: inline-flex; align-items: center; gap: 8px; font-weight: 800;
      border-radius: 999px; padding: 8px 10px; font-size: .86rem;
    }

    /* High-risk badge color */
    .risk.high { background: rgba(255,85,150,.18); color: #ff9bc7; }

    /* Medium-risk badge color */
    .risk.medium { background: rgba(255,201,120,.14); color: #ffd79f; }

    /* Low-risk badge color */
    .risk.low { background: rgba(126,224,177,.12); color: #abf1d0; }

    /* Right-side stacked panel layout */
    .side-panel { padding: 20px; display: grid; gap: 18px; }

    /* Individual cards inside the side panel */
    .side-card { padding: 20px; background: rgba(14,14,18,.92); border-radius: 24px; border: 1px solid var(--border); }

    /* Side card titles */
    .side-card h4 { margin: 0 0 12px; font-family: 'Space Grotesk', sans-serif; font-size: 1.05rem; }

    /* Side card paragraph text */
    .side-card p { margin: 0; color: var(--muted); line-height: 1.55; }

    /* Scrollable chat area */
    .chat { display: grid; gap: 12px; max-height: 340px; overflow: auto; padding-right: 4px; }

    /* Shared message bubble styling */
    .msg {
      border-radius: 20px; padding: 14px 16px; max-width: 92%; line-height: 1.5;
      white-space: pre-wrap; animation: bubbleIn .45s cubic-bezier(.16,1,.3,1);
    }

    /* Bot message bubble */
    .msg.bot { background: linear-gradient(180deg, rgba(27,18,26,.98), rgba(12,12,16,.98)); border: 1px solid rgba(255,108,167,.18); }

    /* User message bubble */
    .msg.user { background: rgba(28,20,31,.96); margin-left: auto; border: 1px solid rgba(255,153,208,.16); }

    /* Grid for the quick-question links */
    .prompt-grid { display: grid; gap: 10px; }

    /* Quick-question link styling */
    .prompt-grid a {
      text-decoration: none; padding: 13px 16px; border-radius: 16px; background: rgba(16,16,20,.88);
      border: 1px solid var(--border); color: var(--text); font-weight: 800;
    }

    /* Small helper note text */
    .note { color: var(--muted); font-size: 0.94rem; line-height: 1.55; }

    /* Bubble fade/slide animation */
    @keyframes bubbleIn {
      from { opacity: 0; transform: translateY(12px) scale(.96); }
      to { opacity: 1; transform: translateY(0) scale(1); }
    }

    /* Talking dots animation */
    @keyframes dots {
      0% { opacity: .25; }
      50% { opacity: 1; }
      100% { opacity: .25; }
    }

    /* Tablet-ish layout adjustments */
    @media (max-width: 1120px) {
      .layout { grid-template-columns: 1fr; }
      #bugBuddyStage { left: 300px; top: 96px; width: 210px; height: 300px; }
      .speech { left: 120px; width: 220px; }
    }

    /* Phone layout adjustments */
    @media (max-width: 640px) {
      .shell { padding: 16px; }
      .topbar { flex-direction: column; align-items: stretch; }
      .hero-form { flex-direction: column; }
      #bugBuddyStage { left: 300px; top: 130px; width: 170px; height: 240px; }
      .speech { left: 85px; width: 190px; font-size: 0.85rem; }
    }
  </style>
</head>
<body>
  <!-- Main page wrapper -->
  <div class="shell">

    <!-- Top header with logo and status -->
    <header class="topbar">
      <div class="brand">

        <!-- Logo icon -->
        <div class="brand-mark" aria-hidden="true">
          <svg viewBox="0 0 48 48" fill="none">
            <rect x="10" y="10" width="28" height="22" rx="10" fill="white" opacity=".95"/>
            <circle cx="19" cy="21" r="3" fill="#ff4a9c"/>
            <circle cx="29" cy="21" r="3" fill="#ff4a9c"/>
            <path d="M18 28c2.3 2.4 9.7 2.4 12 0" stroke="#ff4a9c" stroke-width="3" stroke-linecap="round"/>
            <path d="M24 6v5" stroke="white" stroke-width="3" stroke-linecap="round"/>
            <circle cx="24" cy="5" r="3" fill="#fff"/>
          </svg>
        </div>

        <!-- BugBuddy brand text -->
        <div class="brand-text">
          <h1>BugBuddy</h1>
          <p>Your animated code-risk helper</p>
        </div>
      </div>

      <!-- Top status pill -->
      <div class="top-pill">Bug analysis ready</div>
    </header>

    <!-- Main page content -->
    <main class="layout">
      <section class="stack">

        <!-- Hero panel -->
        <div class="panel hero">
          <div>
            <div class="eyebrow">● BugBuddy is active</div>

            <!-- Main heading -->
            <h2>Upload code and let BugBuddy guide the review.</h2>

            <!-- Intro paragraph -->
            <p class="lead">
              BugBuddy walks across the page, points at where you need to go,
              and helps explain which functions in your uploaded file look most risky.
            </p>

            <!-- Button that activates or toggles speech -->
            <button type="button" class="voice-toggle" id="voiceToggle">🔊 Activate BugBuddy voice</button>

            <!-- Ask BugBuddy a text question -->
            <form method="post" class="hero-form" id="questionSection">
              <input type="text" id="questionInput" name="message" placeholder="Ask BugBuddy something..." />
              <button type="submit" class="ghost-btn">Send</button>
            </form>

            <!-- Upload a source code file -->
            <form method="post" enctype="multipart/form-data" class="upload-box" id="uploadSection">
              <strong>Upload a source-code file</strong>
              <small>Python files use the trained model when available. Other languages use rule-based risk scoring. LLM answers use OpenRouter when your API key is set.</small>
              <input type="file" name="code_file" required />
              <div style="margin-top:12px;">
                <button type="submit" class="primary-btn">Upload Code File</button>
              </div>
            </form>
          </div>
        </div>

        <!-- Right-side stacked info cards -->
        <div class="panel side-panel">

          <!-- Current file information -->
          <div class="side-card">
            <h4>Current file context</h4>
            <p>
              {% if context_note %}
                {{ context_note }}
              {% else %}
                Upload a file to begin, then use the quick buttons or ask BugBuddy questions about the riskiest functions.
              {% endif %}
            </p>

            <!-- Model status message -->
            {% if status_message %}
            <p style="margin-top:12px;">{{ status_message }}</p>
            {% endif %}
          </div>

          <!-- Chat conversation area -->
          <div class="side-card">
            <h4>Conversation</h4>
            <div class="chat">
              <div class="msg bot">{{ bot_greeting }}</div>
              {% if user_message %}
                <div class="msg user">{{ user_message }}</div>
                <div class="msg bot">{{ bot_reply }}</div>
              {% endif %}
            </div>
          </div>

          <!-- Quick links that trigger preset questions -->
          <div class="side-card">
            <h4>Quick questions</h4>
            <div class="prompt-grid">
              <a href="/?q=top">Top 5 functions</a>
              <a href="/?q=high">High-risk functions</a>
              <a href="/?q=medium">Medium-risk functions</a>
              <a href="/?q=metrics">Model metrics</a>
              <a href="/?q=why">Why was something flagged?</a>
            </div>
          </div>
        </div>

        <!-- Results dashboard -->
        <div class="panel dashboard">
          <div class="section-head">
            <h3>Uploaded file analysis</h3>
            <div class="pill">Current mode: {{ bugbuddy_mode|capitalize }}</div>
          </div>

          <!-- Small progress/status chips -->
          <div class="status-line">
            <div class="status-chip {% if bugbuddy_mode in ['welcome','ready','explaining','analyzing'] %}active{% endif %}">Welcome</div>
            <div class="status-chip {% if bugbuddy_mode in ['analyzing','explaining','ready'] %}active{% endif %}">Analyze</div>
            <div class="status-chip {% if bugbuddy_mode in ['explaining','ready'] %}active{% endif %}">Explain</div>
          </div>

          <!-- If rows exist, show the analysis table -->
          {% if uploaded_rows %}
          <div class="table-wrap">
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
                <tr>
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
                  <td><span class="risk {{ row.risk_label|lower }}">{{ row.risk_label }}</span></td>
                  <td>{{ row.explanation }}</td>
                </tr>
                {% endfor %}
              </tbody>
            </table>
          </div>

          <!-- If there is no uploaded file yet, show helper text -->
          {% else %}
          <p class="note">Upload a supported file and BugBuddy will populate this table with ranked functions.</p>
          {% endif %}
        </div>
      </section>
    </main>
  </div>

  <!-- Floating Buddy stage -->
  <div id="bugBuddyStage">

    <!-- Buddy speech bubble -->
    <div class="speech" id="speechBubble">
      <div class="name">BugBuddy</div>
      <p id="speechText">{{ bot_reply if bot_reply else bot_greeting }}</p>
    </div>

    <!-- Buddy image poses -->
    <div class="robot-wrap" id="robotWrap">
      <img class="robot-pose active" data-pose="welcome" src="{{ url_for('static', filename='assets/bugbuddy-welcome.png') }}" alt="BugBuddy welcome pose" width="800" height="1200" loading="eager">
      <img class="robot-pose" data-pose="wave" src="{{ url_for('static', filename='assets/bugbuddy-wave.png') }}" alt="BugBuddy wave pose" width="800" height="1200" loading="lazy">
      <img class="robot-pose" data-pose="idea" src="{{ url_for('static', filename='assets/bugbuddy-idea.png') }}" alt="BugBuddy idea pose" width="800" height="1200" loading="lazy">
      <img class="robot-pose" data-pose="point" src="{{ url_for('static', filename='assets/bugbuddy-point.png') }}" alt="BugBuddy point pose" width="800" height="1200" loading="lazy">
      <img class="robot-pose" data-pose="think" src="{{ url_for('static', filename='assets/bugbuddy-think.png') }}" alt="BugBuddy think pose" width="800" height="1200" loading="lazy">
      <img class="robot-pose" data-pose="blink" src="{{ url_for('static', filename='assets/bugbuddy-blink.png') }}" alt="BugBuddy blink pose" width="800" height="1200" loading="lazy">
      <img class="robot-pose" data-pose="walk1" src="{{ url_for('static', filename='assets/bugbuddy-walk-1.png') }}" alt="BugBuddy walk pose 1" width="800" height="1200" loading="lazy">
      <img class="robot-pose" data-pose="walk2" src="{{ url_for('static', filename='assets/bugbuddy-walk-2.png') }}" alt="BugBuddy walk pose 2" width="800" height="1200" loading="lazy">
    </div>
  </div>

  <script>
    // Gets the current mode passed in from Flask
    const initialMode = {{ bugbuddy_mode|tojson }};

    // Main Buddy stage element
    const stage = document.getElementById('bugBuddyStage');

    // All pose images
    const poseEls = Array.from(document.querySelectorAll('.robot-pose'));

    // Speech bubble text area
    const speechText = document.getElementById('speechText');

    // Whole speech bubble
    const speechBubble = document.getElementById('speechBubble');

    // Upload section used in the tour
    const uploadSection = document.getElementById('uploadSection');

    // Question section used in the tour
    const questionSection = document.getElementById('questionSection');

    // Button for turning voice on/off
    const voiceToggle = document.getElementById('voiceToggle');

    // Timers used for movement and animation loops
    let walkTimer;
    let blinkTimer;
    let idleTimer;
    let talkTimer;

    // Prevents the intro from running multiple times at once
    let introRunning = false;

    // Voice starts enabled, but browser speech usually has to be unlocked by a click first
    let voiceEnabled = true;

    // Stores all available browser voices
    let availableVoices = [];

    // Tracks whether the user has clicked to allow speech
    let speechUnlocked = false;


    // Small helper that pauses the animation flow for a certain amount of time
    function wait(ms) { return new Promise(resolve => setTimeout(resolve, ms)); }

    // Loads the browser voices if speech synthesis is available
    function loadVoices() { availableVoices = window.speechSynthesis ? window.speechSynthesis.getVoices() : []; }

    // Stops whatever BugBuddy is currently saying
    function stopSpeaking() { if ('speechSynthesis' in window) { window.speechSynthesis.cancel(); } }

    // Makes BugBuddy speak the current text out loud
    function speakText(text) {
      // If voice is off, speech is locked, or text is missing, just stop here
      if (!voiceEnabled || !('speechSynthesis' in window) || !text || !speechUnlocked) return;

      // Clears any previous speech so the new message does not overlap
      window.speechSynthesis.cancel();

      // Creates the speech object for the current text
      const utterance = new SpeechSynthesisUtterance(text);

      // Tries to choose a voice that sounds a little friendlier if one exists
      const preferredVoice =
        availableVoices.find(v => /female|zira|samantha|victoria|aria|ava/i.test(v.name)) ||
        availableVoices.find(v => /en-US|en_US/i.test(v.lang)) ||
        availableVoices[0];

      // If a voice was found, use it
      if (preferredVoice) utterance.voice = preferredVoice;

      // These control how fast, high, and loud Buddy sounds
      utterance.rate = 1.0;
      utterance.pitch = 1.12;
      utterance.volume = 1.0;

      // Actually speaks the message
      window.speechSynthesis.speak(utterance);
    }

    // If browser speech exists, load the voices now and again if the browser updates them
    if ('speechSynthesis' in window) {
      loadVoices();
      window.speechSynthesis.onvoiceschanged = loadVoices;
    }

    // Shows only the selected Buddy pose and hides the rest
    function showPose(name) {
      poseEls.forEach(el => el.classList.toggle('active', el.dataset.pose === name));
    }

    // Updates the speech bubble text and optionally makes Buddy talk
    function setSpeech(text, talking = false, speak = true) {
      if (speechText) speechText.textContent = text;
      if (speechBubble) speechBubble.classList.toggle('talking', talking);
      if (speak) speakText(text);
    }

    // Removes the highlight effect from the upload and question areas
    function clearHighlights() {
      uploadSection?.classList.remove('tour-focus');
      questionSection?.classList.remove('tour-focus');
    }

    // Highlights one section during the intro tour
    function focusSection(el) {
      clearHighlights();
      el?.classList.add('tour-focus');
    }

    // Clears all timers so animations do not stack on top of each other
    function clearAllTimers() {
      clearInterval(walkTimer);
      clearInterval(blinkTimer);
      clearInterval(idleTimer);
      clearInterval(talkTimer);
      speechBubble?.classList.remove('talking');
      stopSpeaking();
    }

    // Makes Buddy blink quickly, then switch back to the pose it was using before
    function blinkNow(returnPose = 'welcome') {
      showPose('blink');
      setTimeout(() => showPose(returnPose), 180);
    }

    // Repeats a blink every few seconds unless Buddy is already walking or blinking
    function startBlinkLoop() {
      clearInterval(blinkTimer);
      blinkTimer = setInterval(() => {
        const active = poseEls.find(el => el.classList.contains('active'));
        const current = active ? active.dataset.pose : 'welcome';

        // Skip blinking while walking or already blinking
        if (['walk1', 'walk2', 'blink'].includes(current)) return;

        blinkNow(current);
      }, 3200);
    }

    // Cycles through a few idle poses so Buddy still feels animated when waiting
    function startIdleLoop() {
      clearInterval(idleTimer);

      // Simple loop of poses for the idle animation
      const idleSequence = ['welcome', 'wave', 'welcome', 'idea', 'welcome'];
      let index = 0;

      showPose('welcome');

      idleTimer = setInterval(() => {
        index = (index + 1) % idleSequence.length;
        showPose(idleSequence[index]);
      }, 2300);
    }

    // Sends Buddy back to the default home position on the page
    function homeBuddy() {
      stage.style.left = '550px';
      stage.style.top = '88px';
      stage.style.bottom = 'auto';
    }

    // Moves Buddy closer to a section of the screen
    function moveBuddyTo(x, y) {
      stage.style.left = `${Math.max(40, x - 120)}px`;
      stage.style.top = `${Math.max(110, y - 180)}px`;
      stage.style.bottom = 'auto';
    }

    // Gets a point on the page based on where an element is located
    // This helps Buddy know where to walk
    function pagePointForElement(el) {
      const rect = el.getBoundingClientRect();
      return { x: rect.left + window.scrollX + rect.width * 0.55, y: rect.top + window.scrollY + rect.height * 0.85 };
    }

    // Makes Buddy walk to a section and optionally say something while moving
    async function walkToElement(el, speech = '') {
      if (!el) return;

      const point = pagePointForElement(el);

      if (speech) setSpeech(speech, true, true);

      clearInterval(walkTimer);

      let step = false;
      showPose('walk1');

      // Alternates between the two walking images
      walkTimer = setInterval(() => {
        step = !step;
        showPose(step ? 'walk2' : 'walk1');
      }, 240);

      moveBuddyTo(point.x, point.y);
      await wait(1500);
      clearInterval(walkTimer);
    }

    // Used when BugBuddy is actively analyzing an uploaded file
    function playThinkingLoop() {
      clearAllTimers();
      homeBuddy();
      showPose('think');
      setSpeech("I'm checking your uploaded file and ranking the riskiest functions now.", true, true);
    }

    // Used when BugBuddy is in explanation mode
    function playExplainingLoop() {
      clearAllTimers();
      homeBuddy();

      // Right now this sequence only contains one pose, but I left it like this in case I want to add more later
      const talkSequence = [ 'welcome'];
      let index = 0;

      showPose('point');
      speechBubble?.classList.add('talking');

      talkTimer = setInterval(() => {
        index = (index + 1) % talkSequence.length;
        showPose(talkSequence[index]);
      }, 850);
    }

    // Intro tour that walks the user through where to upload files and ask questions
    async function runIntroTour() {
      // Stops the intro from running twice at once
      if (introRunning) return;
      introRunning = true;

      clearAllTimers();
      clearHighlights();
      homeBuddy();
      showPose('welcome');

      // Buddy starts by greeting the user
      setSpeech("Hi! I'm BugBuddy. I can guide your review and speak my tips out loud too.", true, true);
      await wait(6500);

      // Scrolls to the upload area and points it out
      uploadSection?.scrollIntoView({ behavior: 'smooth', block: 'center' });
      await wait(900);
      focusSection(uploadSection);
      await walkToElement(uploadSection, "Upload your file here so I can analyze your code.");
      showPose('point');
      await wait(1800);

      // Scrolls to the question area and points it out
      questionSection?.scrollIntoView({ behavior: 'smooth', block: 'center' });
      await wait(900);
      focusSection(questionSection);
      await walkToElement(questionSection, "Ask me questions here.");
      showPose('point');
      await wait(1800);

      // Clears highlights and resets the view back near the top
      clearHighlights();
      window.scrollTo({ top: 0, behavior: 'smooth' });
      await wait(1000);

      // Ends the intro with Buddy ready to help
      homeBuddy();
      showPose('idea');
      setSpeech("I'm ready. Upload a file or ask a question to get started.", false, true);
      introRunning = false;
    }

    // Decides which Buddy mode to start based on what Flask sends to the page
    function setInitialBugBuddyMode(mode) {
      if (!mode || mode === 'welcome') return runIntroTour();
      if (mode === 'uploading' || mode === 'analyzing') return playThinkingLoop();
      if (mode === 'explaining' || mode === 'talk' || mode === 'ready') return playExplainingLoop();

      // If no special mode matches, use idle behavior
      homeBuddy();
      startIdleLoop();
      startBlinkLoop();
    }

    // Updates the text on the voice toggle button
    function updateVoiceToggle() {
      if (!voiceToggle) return;

      // If speech has not been unlocked by the user yet, show the activation message
      if (!speechUnlocked) {
        voiceToggle.textContent = '🔊 Activate BugBuddy voice';
        return;
      }

      // Once unlocked, show whether voice is on or off
      voiceToggle.textContent = voiceEnabled ? '🔊 Voice on' : '🔇 Voice off';
    }

    // Handles clicking the voice button
    voiceToggle?.addEventListener('click', () => {
      // First click unlocks browser speech
      if (!speechUnlocked) {
        speechUnlocked = true;
        voiceEnabled = true;
        updateVoiceToggle();

        // If there is already text in the bubble, speak it right away
        if (speechText && speechText.textContent) speakText(speechText.textContent);
        return;
      }

      // After that, clicking just toggles voice on and off
      voiceEnabled = !voiceEnabled;

      if (!voiceEnabled) stopSpeaking();
      else if (speechText && speechText.textContent) speakText(speechText.textContent);

      updateVoiceToggle();
    });

    // Starts Buddy in the right mode as soon as the page loads
    setInitialBugBuddyMode(initialMode);
  </script>
</body>
</html>
"""

# Small helper that replaces missing values with a fallback value
def clean_value(value, fallback=""):
    if pd.isna(value):
        return fallback
    return value

# Checks whether the uploaded file extension is allowed
def allowed_file(filename):
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS

# Detects the programming language based on file extension
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

# Checks if the file is Python
# Python files can use the trained model path
def is_python_file(filename):
    return Path(filename).suffix.lower() in PYTHON_EXTENSIONS

# Cleans up function names so empty or anonymous names are easier to read
def normalize_function_name(name):
    if pd.isna(name) or not name:
        return "anonymous closure"

    text = str(name).strip()

    if text in {"(anonymous)", "<anonymous>", "anonymous"}:
        return "anonymous closure"

    return text

# Loads the saved machine learning model if one exists
# If not, the app still works but falls back to rule-based scoring
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

# Creates a short explanation for why a function may look risky
# This is mainly used in the results table
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

# Creates a more conversational explanation for chat replies
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

    # If the model gave a probability, explain the risk in that way
    if prob is not None:
        if risk_label == "High":
            chance = f"It is currently ranked in the high-risk group for this file, with probability {prob:.3f}."
        elif risk_label == "Medium":
            chance = f"It is currently ranked in the medium-risk group for this file, with probability {prob:.3f}."
        else:
            chance = f"It is currently ranked in the lower-risk group for this file, with probability {prob:.3f}."

    # If the model did not give a probability, explain it using the simpler metric-based result
    else:
        if risk_label == "High":
            chance = "It looks high risk based on its complexity metrics."
        elif risk_label == "Medium":
            chance = "It looks moderately risky based on its complexity metrics."
        else:
            chance = "It does not look especially risky right now."

    return f"In simpler terms, {base}. {chance}"

# Gives a short suggestion for how the function could be improved
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
        tips.append("Review edge cases, naming, and repeated logic; this function may mainly need cleanup rather than a large refactor.")

    return "Suggested fix: " + " ".join(tips[:2])

# Simple fallback scoring system used when no trained model is available
# This is also used for non-Python files
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

# Uses Lizard to analyze the uploaded file and pull out function-level metrics
def analyze_file_with_lizard(filepath):
    analysis = lizard.analyze_file(str(filepath))
    rows = []

    for func in analysis.function_list:
        params = getattr(func, "parameter_count", None)

        # If parameter_count is missing, try to count parameters another way
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

        # Stores the metrics for each detected function
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

# Converts raw prediction results into High / Medium / Low relative rankings
def apply_relative_risk_bands(scored_rows):
    if not scored_rows:
        return scored_rows

    # Sorts functions by bug probability from highest to lowest
    ranked = sorted(
        scored_rows,
        key=lambda row: row.get("bug_probability", -1) if row.get("bug_probability") is not None else -1,
        reverse=True,
    )

    total = len(ranked)

    # Top 20% becomes High, next section becomes Medium, rest becomes Low
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
        row["fix_suggestion"] = fix_suggestion(row)

    return ranked

# Main scoring function used after a file is uploaded
def score_uploaded_file(filepath, model, features, threshold, display_name=None):
    rows = analyze_file_with_lizard(filepath)

    if not rows:
        return [], "No functions were detected in the uploaded code file."

    shown_name = display_name if display_name else Path(filepath).name
    is_python = is_python_file(filepath)
    df = pd.DataFrame(rows)
    scored_rows = []

    # If this is a Python file and the model is available, use ML predictions
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
            row["fix_suggestion"] = ""
            scored_rows.append(row)

        scored_rows = apply_relative_risk_bands(scored_rows)

    # Otherwise use the rule-based fallback system
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
            row["fix_suggestion"] = fix_suggestion(row)
            scored_rows.append(row)

        # Sorts fallback rows by High, Medium, then Low
        def fallback_sort_value(row):
            mapping = {"High": 3, "Medium": 2, "Low": 1}
            return mapping.get(row.get("risk_label", "Low"), 0)

        scored_rows.sort(key=fallback_sort_value, reverse=True)

        for idx, row in enumerate(scored_rows, start=1):
            row["rank"] = idx
            row["display_score"] = f"#{idx} · {row['risk_label']}"

    # Final sort to keep highest-risk items at the top
    def sort_value(row):
        if row.get("bug_probability") is not None:
            return row["bug_probability"]

        mapping = {"High": 3, "Medium": 2, "Low": 1}
        return mapping.get(row.get("risk_label", "Low"), 0)

    scored_rows.sort(key=sort_value, reverse=True)
    return scored_rows, ""

# Headers sent with OpenRouter requests
def build_openrouter_headers():
    return {
        "HTTP-Referer": "http://localhost:5001",
        "X-OpenRouter-Title": "BugBuddy",
    }

# Turns technical LLM errors into simpler messages for the user
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

# Sends the uploaded code and user question to OpenRouter for a deeper explanation
def ask_llm_about_code(user_question, file_name, language_name, code_text, top_rows=None):
    if not OPENROUTER_API_KEY:
        return (
            "I can answer deeper code questions, but your OpenRouter API key is missing. "
            "Add OPENROUTER_API_KEY to your .env file first."
        )

    if not code_text.strip():
        return "I need the uploaded code text before I can answer detailed code questions."

    # Summarizes the top ranked functions to help the model stay grounded
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

    # Main prompt sent to OpenRouter
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

    # Removes duplicate models from the fallback list
    for model_name in OPENROUTER_FALLBACK_MODELS:
        if model_name and model_name not in unique_models:
            unique_models.append(model_name)

    # Tries each model in order until one works
    for idx, model_name in enumerate(unique_models):
        try:
            response = client.chat.completions.create(
                model=model_name,
                extra_headers=build_openrouter_headers(),
                messages=[
                    {"role": "system", "content": "You are BugBuddy, a friendly coding assistant focused on uploaded code."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
            )

            content = response.choices[0].message.content.strip()

            # If a fallback model was used, mention that in the reply
            if model_name != OPENROUTER_MODEL:
                return f"{content}\n\n(BugBuddy used fallback model: {model_name})"

            return content

        except Exception as e:
            last_error = e
            if idx < len(unique_models) - 1:
                time.sleep(1.5)
                continue

    return humanize_llm_error(last_error)

# Handles the chatbot replies based on the user's message
def chatbot_response(message, rows, features, language_name):
    global LAST_UPLOADED_CODE, LAST_UPLOADED_FILENAME

    msg = message.lower().strip()

    # If no file is uploaded yet, BugBuddy cannot analyze anything
    if not rows:
        return "Please upload a supported code file first so I can analyze its functions."

    # If the user asks why something was flagged, explain the top-ranked function
    if "why" in msg or "flagged" in msg:
        top = rows[0]
        return (
            f"The highest-ranked function in your uploaded {language_name} file is "
            f"{top['function']} in {top['file']}.\n\n"
            f"Why it stands out: {top['explanation']}.\n\n"
            f"{top['simple_explanation']}\n\n"
            f"{top['fix_suggestion']}"
        )

    # If the user asks for high-risk functions, list them
    if "high-risk" in msg or "high risk" in msg:
        high = [row for row in rows if row.get("risk_label") == "High"]

        if not high:
            return f"I didn't find any high-risk functions in your uploaded {language_name} file."

        response = f"I found {len(high)} high-risk functions in your uploaded {language_name} file:\n\n"
        for row in high[:10]:
            response += f"- #{row['rank']} {row['function']} ({row['probability_text']})\n"
        return response

    # If the user asks for medium-risk functions, list them
    if "medium" in msg:
        med = [row for row in rows if row.get("risk_label") == "Medium"]

        if not med:
            return f"I didn't find any medium-risk functions in your uploaded {language_name} file right now."

        response = f"I found {len(med)} medium-risk functions in your uploaded {language_name} file:\n\n"
        for row in med[:10]:
            response += f"- #{row['rank']} {row['function']} ({row['probability_text']})\n"
        return response

    # If the user asks what metrics are used, explain the inputs
    if "metrics" in msg or "features" in msg:
        feature_text = ", ".join(features) if features else "no saved feature list found"
        return (
            f"I analyze your uploaded {language_name} file using function-level metrics such as "
            f"nloc, ccn, params, and length. When the saved model is available, I use these model "
            f"features: {feature_text}. I rank functions within the uploaded file so the riskiest "
            f"ones are easier to review first."
        )

    # If the user asks for the top bug-prone functions, show the top 5
    if "top" in msg or "most bug" in msg or "most bug-prone" in msg or "most complex" in msg:
        top = rows[:5]
        response = f"Here are the top 5 highest-ranked functions in your uploaded {language_name} file:\n\n"
        for row in top:
            response += f"{row['rank']}. {row['function']} ({row['probability_text']}, {row['risk_label']})\n"
        return response

    # If the question sounds more open-ended, send it to the LLM
    if any(word in msg for word in ["fix", "solution", "refactor", "rewrite", "what does", "how does", "bug", "error", "issue", "explain"]):
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

# Main route for the website
# Handles page loading, messages, uploads, and quick question links
@app.route("/", methods=["GET", "POST"])
def index():
    global LAST_UPLOADED_ROWS, LAST_UPLOADED_FILENAME, LAST_UPLOADED_LANGUAGE, LAST_UPLOADED_CODE

    # Loads the saved model and current model status
    model, train_features, threshold, model_status = load_saved_model()
    status_message = model_status
    context_note = ""

    # Default greeting shown in the conversation box
    bot_greeting = (
        "Hi! I'm BugBuddy. Upload a supported code file and I'll analyze its functions for bug risk. "
        "Python files use the saved trained model when available, while other languages use rule-based risk analysis. "
        "I can also answer deeper code questions through OpenRouter when your API key is set."
    )

    user_message = ""
    bot_reply = ""
    uploaded_rows = LAST_UPLOADED_ROWS if LAST_UPLOADED_ROWS else []
    bugbuddy_mode = "welcome"

    # If a file is already stored, keep showing its context
    if LAST_UPLOADED_ROWS and LAST_UPLOADED_FILENAME:
        context_note = (
            f"Current uploaded file: {LAST_UPLOADED_FILENAME} ({LAST_UPLOADED_LANGUAGE}). "
            f"Quick buttons and chat use this uploaded file only."
        )

    # Handles form submissions
    if request.method == "POST":

        # If the user submitted a text question
        if "message" in request.form and request.form.get("message", "").strip():
            user_message = request.form.get("message", "").strip()
            language_name = LAST_UPLOADED_LANGUAGE if LAST_UPLOADED_LANGUAGE else "code"
            bot_reply = chatbot_response(user_message, uploaded_rows, train_features, language_name)
            bugbuddy_mode = "explaining"

        # If the user uploaded a file
        elif "code_file" in request.files:
            uploaded_file = request.files["code_file"]

            if uploaded_file and uploaded_file.filename and allowed_file(uploaded_file.filename):
                temp_path = None

                try:
                    # Saves the file temporarily for analysis
                    suffix = Path(uploaded_file.filename).suffix
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                        uploaded_file.save(tmp_file.name)
                        temp_path = tmp_file.name

                    # Reads the uploaded code into text so the LLM can answer questions about it later
                    with open(temp_path, "r", encoding="utf-8", errors="ignore") as f:
                        uploaded_code_text = f.read()

                    # Scores the uploaded file
                    uploaded_rows, upload_message = score_uploaded_file(
                        temp_path,
                        model,
                        train_features,
                        threshold,
                        display_name=uploaded_file.filename,
                    )

                    detected_language = detect_language(uploaded_file.filename)
                    user_message = f"Please analyze my uploaded file: {uploaded_file.filename}"

                    # If scoring worked and functions were found
                    if uploaded_rows:
                        LAST_UPLOADED_ROWS = uploaded_rows
                        LAST_UPLOADED_FILENAME = uploaded_file.filename
                        LAST_UPLOADED_LANGUAGE = detected_language
                        LAST_UPLOADED_CODE = uploaded_code_text

                        context_note = (
                            f"Current uploaded file: {uploaded_file.filename} ({detected_language}). "
                            f"Quick buttons and chat use this uploaded file only."
                        )

                        # Builds the first BugBuddy summary reply for the upload
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

                    # If no functions were found, clear the saved file state
                    else:
                        LAST_UPLOADED_ROWS = []
                        LAST_UPLOADED_FILENAME = ""
                        LAST_UPLOADED_LANGUAGE = ""
                        LAST_UPLOADED_CODE = ""
                        context_note = ""
                        bot_reply = upload_message or (
                            f"I analyzed {uploaded_file.filename}, but I did not detect any functions that could be scored."
                        )
                        bugbuddy_mode = "welcome"

                # If something breaks during upload or analysis, show the error
                except Exception as e:
                    user_message = f"Please analyze my uploaded file: {uploaded_file.filename}"
                    bot_reply = f"Upload failed: {type(e).__name__}: {e}"
                    bugbuddy_mode = "welcome"

                # Always delete the temporary uploaded file when done
                finally:
                    if temp_path and os.path.exists(temp_path):
                        os.remove(temp_path)

            # If the file type is not supported, show a helpful message
            else:
                user_message = "Please analyze my uploaded file."
                bot_reply = (
                    "Please upload a supported code file. Supported extensions include .py, .js, .ts, "
                    ".java, .c, .cpp, .cs, .go, .php, .rb, .swift, and others."
                )
                bugbuddy_mode = "welcome"

    # Handles quick links like ?q=top and ?q=why
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
            bugbuddy_mode = "explaining"
        elif uploaded_rows:
            bugbuddy_mode = "ready"

    # Renders the full template with all the current values
    return render_template_string(
        TEMPLATE,
        bot_greeting=bot_greeting,
        user_message=user_message,
        bot_reply=bot_reply,
        uploaded_rows=uploaded_rows,
        status_message=status_message,
        context_note=context_note,
        bugbuddy_mode=bugbuddy_mode,
    )

# Starts the Flask app locally on port 5001
if __name__ == "__main__":
    app.run(debug=True, use_reloader=False, use_debugger=False, port=5001)
























    