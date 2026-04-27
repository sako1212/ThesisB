import os
import pandas as pd
from preprocessor import clean_text
from llm_detector import detect_scam, classify_scam

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

    results = []

    for _, row in df.iterrows():
        print(f"Processing ad {row['ad_id']}...")

        original_text = row["ad_text"]
        cleaned_text = clean_text(original_text)

        detection = detect_scam(cleaned_text)

        scam_category = {
            "category": "N/A",
            "confidence": 0.0,
            "reasoning": "Not classified because ad was not detected as scam."
        }

        if detection.get("label") in ["scam", "suspicious"]:
            scam_category = classify_scam(cleaned_text)

        results.append({
            "ad_id": row["ad_id"],
            "original_text": original_text,
            "cleaned_text": cleaned_text,
            "true_label": row.get("true_label", ""),
            "true_category": row.get("true_category", ""),
            "predicted_label": detection.get("label", "error"),
            "detection_confidence": detection.get("confidence", 0.0),
            "detection_reasoning": detection.get("reasoning", ""),
            "predicted_category": scam_category.get("category", "N/A"),
            "classification_confidence": scam_category.get("confidence", 0.0),
            "classification_reasoning": scam_category.get("reasoning", "")
        })

    output_df = pd.DataFrame(results)
    output_df.to_csv(OUTPUT_FILE, index=False)

    print(f"Done. Results saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()