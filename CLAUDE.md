# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AiDetective is an Honours Thesis project (Macquarie University) that uses LLMs to detect and classify scam advertisements on Meta's advertising platform (Facebook/Instagram). It compares 5 LLM providers on a binary scam-detection task.

## Setup

```bash
# Activate virtual environment (Windows)
source venv/Scripts/activate

# Install dependencies
pip install -r requirements.txt

# Configure API keys
cp .env.example .env
# Edit .env with keys for: OPENAI_API_KEY, GEMINI_API_KEY, ANTHROPIC_API_KEY, DEEPSEEK_API_KEY, GROQ_API_KEY
# Missing keys are skipped gracefully — only models with valid keys will run
```

## Running the Pipeline

All commands run from `src/`:

```bash
# Single-model detection (GPT-4o-mini by default) → outputs/results.csv
python main.py

# Multi-model comparison across all configured models → outputs/comparison_results.csv + comparison_table.csv
python compare_models.py

# Evaluate single-model results
python evaluator.py

# Evaluate multi-model results (optional: filter by model name)
python evaluator.py --multi
python evaluator.py --multi --model "GPT-4o-mini"
```

There is no test suite (`tests/` is empty).

## Architecture

Two-stage LLM pipeline:
1. **Detection** — classifies each ad as `scam`, `suspicious`, or `legitimate`
2. **Classification** — if scam/suspicious, categorizes into: `phishing`, `investment`, `impersonation`, `health`, `giveaway`, `other`

Both stages call the same LLM and expect JSON responses with `label`, `confidence` (0.0–1.0), and `reasoning` fields.

### Key Files

| File | Role |
|------|------|
| `src/main.py` | Entry point for single-model runs; loads CSV, runs pipeline, writes output |
| `src/models.py` | `BaseDetector` abstract class + 5 concrete implementations (OpenAI, Gemini, Anthropic, DeepSeek, Groq/Llama) |
| `src/compare_models.py` | Instantiates all models with valid API keys and runs comparison |
| `src/evaluator.py` | Reads output CSVs and computes accuracy, precision, recall, F1, false positive rate |
| `src/preprocessor.py` | `clean_text()`: lowercases, masks URLs as `[URL]`, strips special chars, normalizes whitespace |
| `src/llm_detector.py` | Legacy single-model detector (OpenAI only); superseded by `models.py` |

### Data Flow

```
data/sample_ads.csv  →  preprocessor.clean_text()  →  LLM detect()  →  LLM classify()  →  outputs/*.csv
```

Input CSV columns: `ad_id`, `ad_text`, `source_url`, `true_label`, `true_category`

Output CSV adds: `cleaned_text`, `predicted_label`, `detection_confidence`, `detection_reasoning`, `predicted_category`, `classification_confidence`, `classification_reasoning`

### Evaluation Note

"Suspicious" is treated as "scam" during metric computation (conservative binary classification). JSON parse failures produce an `"error"` label and are counted as misclassifications.

## Supported Models

| Name | Provider | API Key |
|------|----------|---------|
| GPT-4o-mini | OpenAI | `OPENAI_API_KEY` |
| Gemini 1.5 Flash | Google | `GEMINI_API_KEY` |
| Claude Haiku 4.5 | Anthropic | `ANTHROPIC_API_KEY` |
| DeepSeek Chat | DeepSeek | `DEEPSEEK_API_KEY` |
| Llama 3.1 8B Instant | Groq | `GROQ_API_KEY` |
