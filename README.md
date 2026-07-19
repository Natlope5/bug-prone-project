# BugBuddy Website

BugBuddy is a live interactive website for analyzing uploaded source code and surfacing bug-prone functions. The deployed website is available at [bugbuddy-app.onrender.com](https://bugbuddy-app.onrender.com/).

BugBuddy is presented as a multilingual HCI prototype that helps users upload supported code files, analyze functions, rank them by bug risk, and receive plain-language guidance through a browser-based experience.

## Live Website

The live deployed version of BugBuddy is hosted here:

- [https://bugbuddy-app.onrender.com/](https://bugbuddy-app.onrender.com/)

The website introduces BugBuddy as a friendly code-risk helper and allows users to upload a supported code file directly in the browser.

## Overview

BugBuddy is designed as an actual website experience rather than just a command-line tool or a small prototype wrapper.

The site includes a guided landing experience, code upload workflow, ranked results area, quick-question buttons, and browser-based explanations that help users understand which functions in an uploaded file may deserve more review attention.

## What the Website Does

BugBuddy allows a user to upload a supported source-code file and then analyzes the functions detected in that file.

The website ranks functions within the uploaded file so users can review the riskiest code first.

The live site explains that:

- Python uploads use the saved trained model when available.
- Other supported languages use rule-based scoring so the interface still works across languages.
- Results are ranked within the uploaded file so the riskiest functions appear first.

## Website Features

### Browser-based upload flow

The website includes a file-upload workflow directly in the page.

Users do not need to run scoring manually through the interface once the site is open; they can upload a supported file and view the results in the browser.

### Ranked function review

BugBuddy analyzes uploaded files at the function level and surfaces bug-prone functions for review.

This makes the site useful as a lightweight code-review aid and educational review interface.

### Multi-language support

The live site describes BugBuddy as a multilingual HCI prototype.

Python uploads can use the saved trained model when available, while other supported languages are still handled through rule-based risk analysis so the site remains usable across more than one language.

### Plain-language guidance

BugBuddy is designed to explain results in a user-friendly way rather than only showing raw code metrics.

The site uses a friendly assistant framing so users can move from upload to interpretation more easily.

### Quick exploration workflow

The live website tells users to upload a supported code file first and then use quick buttons to explore only that uploaded file.

This helps keep the review session focused on the current upload rather than mixing results from unrelated files.

## Current Behavior

According to the live deployed site, BugBuddy currently:

1. Accepts a supported code file upload.
2. Analyzes the functions in that uploaded file.
3. Uses the saved trained model for Python files when available.
4. Uses rule-based risk analysis for other supported languages.
5. Ranks functions so the riskiest ones can be reviewed first.

## Project Positioning

BugBuddy is best described as a live multilingual website and HCI prototype for code-risk review.

It is intended to help users interact with function-level risk analysis through a more understandable browser interface instead of relying only on scripts or raw output files.

## Local Development

If you want to run the project locally from the repository root, start the web app with:

```bash
py "scripts\bugbuddy_app"
```

Then open the local address shown in the terminal, typically:

```text
http://127.0.0.1:5000/
```

## Notes

The strongest trained-model path currently applies to Python uploads when a saved model is available.

Broader language support depends on rule-based scoring so that the website can still analyze and rank uploaded files across multiple supported languages.