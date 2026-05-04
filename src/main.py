import argparse
import os
import pandas as pd
from preprocessor import clean_text
from models import GPTDetector

DEFAULT_INPUT = "../outputs/dataset_labelled.csv"
OUTPUT_FILE = "../outputs/results.csv"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=DEFAULT_INPUT,
                        help=f"Input CSV (default: {DEFAULT_INPUT})")
    parser.add_argument("--all-rows", action="store_true",
                        help="Process every row (default: skip unlabelled rows)")
    args = parser.parse_args()

    print("Starting AiDetective pipeline...")

    if not os.path.exists(args.input):
        print(f"ERROR: Input file not found: {args.input}")
        if args.input == DEFAULT_INPUT:
            print("  Run build_dataset.py and label_dataset.py first, "
                  "or pass --input <path>.")
        return

    os.makedirs("../outputs", exist_ok=True)

    df = pd.read_csv(args.input)
    print(f"Loaded {len(df)} rows from {args.input}")

    if "true_label" in df.columns and not args.all_rows:
        before = len(df)
        df = df[df["true_label"].notna() & (df["true_label"].astype(str).str.strip() != "")].reset_index(drop=True)
        if len(df) < before:
            print(f"Filtered to {len(df)} labelled rows ({before - len(df)} unlabelled skipped). "
                  f"Pass --all-rows to include them.")

    if len(df) == 0:
        print("No rows to process. Label some ads first with label_dataset.py.")
        return

    if "ad_id" not in df.columns and "library_id" in df.columns:
        df["ad_id"] = df["library_id"]

    detector = GPTDetector()
    results = []

    for _, row in df.iterrows():
        print(f"Processing ad {row['ad_id']}...")

        original_text = row["ad_text"]
        cleaned_text = clean_text(original_text)

        detection = detector.detect(cleaned_text)

        if detection["isScamFlagged"] and detection["label"] != "error":
            classification = detector.classify(cleaned_text)
        else:
            classification = {
                "scamCategory": "N/A",
                "classificationScore": 0.0,
                "explanationTrace": "Not classified - ad was not flagged as scam.",
            }

        results.append({
            "ad_id": row["ad_id"],
            "original_text": original_text,
            "cleaned_text": cleaned_text,
            "true_label": row.get("true_label", ""),
            "true_category": row.get("true_category", ""),
            "predicted_label": detection["label"],
            "detection_confidence": detection["evaluationScore"],
            "detection_reasoning": detection["reasoningSummary"],
            "predicted_category": classification["scamCategory"],
            "classification_confidence": classification["classificationScore"],
            "classification_reasoning": classification["explanationTrace"],
        })

    output_df = pd.DataFrame(results)
    output_df.to_csv(OUTPUT_FILE, index=False)

    print(f"Done. Results saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
