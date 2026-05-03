import os
import pandas as pd
from preprocessor import clean_text
from models import GPTDetector

INPUT_FILE = "../data/sample_ads.csv"
OUTPUT_FILE = "../outputs/results.csv"


def main():
    print("Starting AiDetective pipeline...")

    if not os.path.exists(INPUT_FILE):
        print(f"ERROR: Input file not found: {INPUT_FILE}")
        return

    os.makedirs("../outputs", exist_ok=True)

    df = pd.read_csv(INPUT_FILE)
    print(f"Loaded {len(df)} ads from {INPUT_FILE}")

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
