"""
compare_models.py — Multi-model scam detection comparison for AiDetective thesis.

Runs every configured LLM against the labelled dataset, computes per-model
classification metrics, and saves:
  outputs/comparison_results.csv  — raw per-ad predictions from all models
  outputs/comparison_table.csv    — summary metrics table (thesis-ready)

Usage:
    cd src
    python compare_models.py                       # uses outputs/dataset_labelled.csv
    python compare_models.py --input ../data/sample_ads.csv
"""

import argparse
import os
import sys
import time
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

# Allow running from the src/ directory
sys.path.insert(0, os.path.dirname(__file__))

from preprocessor import clean_text
from models import load_detectors

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DEFAULT_INPUT = "../outputs/dataset_labelled.csv"
OUTPUT_RAW    = "../outputs/comparison_results.csv"
OUTPUT_TABLE  = "../outputs/comparison_table.csv"

# Pause between API calls to stay within rate limits (seconds)
API_DELAY = 1.0


# ---------------------------------------------------------------------------
# Normalise label  — map model output to canonical set
# ---------------------------------------------------------------------------

def normalise_label(raw: str) -> str:
    """Collapse model output to 'scam', 'suspicious', or 'legitimate'."""
    raw = str(raw).strip().lower()
    if "scam" in raw:
        return "scam"
    if "suspicious" in raw or "high-risk" in raw or "potentially" in raw:
        return "suspicious"
    if "legitimate" in raw or "legit" in raw:
        return "legitimate"
    return "error"


# ---------------------------------------------------------------------------
# Metrics helper
# ---------------------------------------------------------------------------

def compute_metrics(y_true: list, y_pred: list) -> dict:
    """
    Binary metrics treating 'scam' as the positive class.
    'suspicious' predictions are counted as 'scam' for recall purposes
    (conservative: flag rather than miss).
    """
    # Collapse suspicious → scam for binary evaluation
    def binarise(labels):
        return ["scam" if l in ("scam", "suspicious") else "legitimate" for l in labels]

    y_true_bin = binarise(y_true)
    y_pred_bin = binarise(y_pred)

    valid = [(t, p) for t, p in zip(y_true_bin, y_pred_bin) if p != "error"]
    if not valid:
        return {k: None for k in ("accuracy", "precision", "recall", "f1", "fpr", "errors")}

    yt, yp = zip(*valid)

    tn, fp, fn, tp = confusion_matrix(
        yt, yp, labels=["legitimate", "scam"]
    ).ravel()

    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    errors = sum(1 for p in y_pred_bin if p == "error")

    return {
        "accuracy":  round(accuracy_score(yt, yp), 4),
        "precision": round(precision_score(yt, yp, pos_label="scam", zero_division=0), 4),
        "recall":    round(recall_score(yt, yp, pos_label="scam", zero_division=0), 4),
        "f1":        round(f1_score(yt, yp, pos_label="scam", zero_division=0), 4),
        "fpr":       round(fpr, 4),
        "errors":    errors,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", default=DEFAULT_INPUT,
        help=f"Input CSV (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--all-rows", action="store_true",
        help="Process every row, even unlabelled ones (default: skip rows with no true_label)",
    )
    args = parser.parse_args()

    os.makedirs("../outputs", exist_ok=True)

    # --- Load dataset ---
    if not os.path.exists(args.input):
        print(f"ERROR: {args.input} not found.")
        if args.input == DEFAULT_INPUT:
            print("  Run build_dataset.py and label_dataset.py first, "
                  "or pass --input <path> to use a different CSV.")
        return

    df = pd.read_csv(args.input)
    print(f"\nLoaded {len(df)} rows from {args.input}")

    # Filter to labelled rows unless --all-rows is set
    if "true_label" in df.columns and not args.all_rows:
        before = len(df)
        df = df[df["true_label"].notna() & (df["true_label"].astype(str).str.strip() != "")].reset_index(drop=True)
        if len(df) < before:
            print(f"Filtered to {len(df)} labelled rows ({before - len(df)} unlabelled skipped). "
                  f"Pass --all-rows to include them.")

    if len(df) == 0:
        print("No rows to process. Label some ads first with label_dataset.py.")
        return

    # Ensure ad_id exists (label_dataset.py adds it; older sample_ads.csv has it natively)
    if "ad_id" not in df.columns and "library_id" in df.columns:
        df["ad_id"] = df["library_id"]
    # Ensure label columns exist so the per-ad row writer doesn't KeyError on
    # an unlabelled dataset (--all-rows on outputs/dataset.csv)
    for col in ("true_label", "true_category"):
        if col not in df.columns:
            df[col] = ""

    # Preprocess all ad texts once
    df["cleaned_text"] = df["ad_text"].apply(clean_text)

    # --- Load model detectors ---
    print("\nInitialising models (skipping any with missing API keys):")
    detectors = load_detectors(skip_missing=True)

    if not detectors:
        print("No models could be loaded. Check your .env file.")
        return

    print(f"\nRunning {len(detectors)} model(s) on {len(df)} ads...\n")

    # --- Run each model ---
    raw_rows = []

    for detector in detectors:
        print(f"--- {detector.name} ---")
        for _, row in df.iterrows():
            ad_id = row["ad_id"]
            print(f"  ad {ad_id}...", end=" ", flush=True)

            detection = detector.detect(row["cleaned_text"])
            label = normalise_label(detection.get("label", "error"))
            score = detection.get("evaluationScore", 0.0)

            if detection.get("isScamFlagged") and detection["label"] != "error":
                time.sleep(API_DELAY)
                classification = detector.classify(row["cleaned_text"])
            else:
                classification = {
                    "scamCategory": "N/A",
                    "classificationScore": 0.0,
                    "explanationTrace": "Not classified - ad was not flagged as scam.",
                }

            raw_rows.append({
                "model":            detector.name,
                "ad_id":            ad_id,
                "true_label":       row["true_label"],
                "true_category":    row.get("true_category", "N/A"),
                "predicted_label":  label,
                "confidence":       score,
                "reasoning":        detection.get("reasoningSummary", ""),
                "predicted_category":       classification["scamCategory"],
                "classification_confidence": classification["classificationScore"],
                "classification_reasoning":  classification["explanationTrace"],
            })
            print(f"{label} ({score:.2f}) -> {classification['scamCategory']}")
            time.sleep(API_DELAY)

        print()

    # --- Save raw results ---
    raw_df = pd.DataFrame(raw_rows)
    raw_df.to_csv(OUTPUT_RAW, index=False)
    print(f"Raw results saved to {OUTPUT_RAW}")

    # --- Compute per-model metrics ---
    summary_rows = []
    for detector in detectors:
        model_df = raw_df[raw_df["model"] == detector.name]
        y_true = model_df["true_label"].tolist()
        y_pred = model_df["predicted_label"].tolist()
        avg_conf = round(model_df["confidence"].mean(), 4)

        metrics = compute_metrics(y_true, y_pred)
        metrics["model"] = detector.name
        metrics["avg_confidence"] = avg_conf
        metrics["n_ads"] = len(model_df)
        summary_rows.append(metrics)

    summary_df = pd.DataFrame(summary_rows)

    # Reorder columns for the thesis table
    cols = ["model", "n_ads", "accuracy", "precision", "recall", "f1", "fpr", "avg_confidence", "errors"]
    summary_df = summary_df[cols]
    summary_df.to_csv(OUTPUT_TABLE, index=False)

    # --- Print comparison table ---
    print("\n" + "=" * 80)
    print("MULTI-MODEL SCAM DETECTION COMPARISON TABLE")
    print("=" * 80)
    print(f"{'Model':<22} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6} {'FPR':>6} {'AvgConf':>8} {'Errors':>7}")
    print("-" * 80)
    for _, r in summary_df.iterrows():
        print(
            f"{r['model']:<22} "
            f"{r['accuracy']:>6.4f} "
            f"{r['precision']:>6.4f} "
            f"{r['recall']:>6.4f} "
            f"{r['f1']:>6.4f} "
            f"{r['fpr']:>6.4f} "
            f"{r['avg_confidence']:>8.4f} "
            f"{int(r['errors']):>7}"
        )
    print("=" * 80)
    print(f"\nComparison table saved to {OUTPUT_TABLE}")


if __name__ == "__main__":
    main()
