"""
evaluator.py — Compute and display metrics from a saved comparison CSV.

Supports both single-model output (outputs/results.csv) and
multi-model output (outputs/comparison_results.csv).

Usage:
    cd src
    python evaluator.py                              # single-model results
    python evaluator.py --multi                      # multi-model comparison
    python evaluator.py --multi --model "GPT-4o-mini"  # one model from multi file
"""

import argparse
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

SINGLE_FILE = "../outputs/results.csv"
MULTI_FILE  = "../outputs/comparison_results.csv"


def binarise(labels):
    """Treat 'suspicious' the same as 'scam' for binary evaluation."""
    return ["scam" if l in ("scam", "suspicious") else "legitimate" for l in labels]


def print_metrics(model_name: str, y_true: list, y_pred: list, avg_conf: float = None):
    y_true_bin = binarise(y_true)
    y_pred_bin = binarise(y_pred)

    valid = [(t, p) for t, p in zip(y_true_bin, y_pred_bin) if p not in ("error",)]
    if not valid:
        print(f"  {model_name}: no valid predictions to evaluate.")
        return

    yt, yp = zip(*valid)
    tn, fp, fn, tp = confusion_matrix(yt, yp, labels=["legitimate", "scam"]).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    errors = len(y_pred) - len(valid)

    print(f"\n{'='*60}")
    print(f"Model: {model_name}")
    print(f"{'='*60}")
    print(f"  Ads evaluated : {len(valid)}  |  Parse errors: {errors}")
    print(f"  Accuracy      : {accuracy_score(yt, yp):.4f}")
    print(f"  Precision     : {precision_score(yt, yp, pos_label='scam', zero_division=0):.4f}")
    print(f"  Recall        : {recall_score(yt, yp, pos_label='scam', zero_division=0):.4f}")
    print(f"  F1-Score      : {f1_score(yt, yp, pos_label='scam', zero_division=0):.4f}")
    print(f"  False Pos Rate: {fpr:.4f}")
    if avg_conf is not None:
        print(f"  Avg Confidence: {avg_conf:.4f}")

    print("\n  Confusion Matrix (rows=true, cols=pred):")
    print("                Pred Legitimate  Pred Scam")
    print(f"  True Legit         {tn:>5}          {fp:>5}")
    print(f"  True Scam          {fn:>5}          {tp:>5}")


def run_single():
    try:
        df = pd.read_csv(SINGLE_FILE)
    except FileNotFoundError:
        print(f"ERROR: {SINGLE_FILE} not found. Run main.py first.")
        return

    print_metrics(
        model_name="GPT-4o-mini (single run)",
        y_true=df["true_label"].tolist(),
        y_pred=df["predicted_label"].tolist(),
        avg_conf=df["detection_confidence"].mean() if "detection_confidence" in df else None,
    )


def run_multi(filter_model: str = None):
    try:
        df = pd.read_csv(MULTI_FILE)
    except FileNotFoundError:
        print(f"ERROR: {MULTI_FILE} not found. Run compare_models.py first.")
        return

    models = df["model"].unique()
    if filter_model:
        models = [m for m in models if filter_model.lower() in m.lower()]
        if not models:
            print(f"No model matching '{filter_model}' found in results.")
            return

    for model in models:
        mdf = df[df["model"] == model]
        print_metrics(
            model_name=model,
            y_true=mdf["true_label"].tolist(),
            y_pred=mdf["predicted_label"].tolist(),
            avg_conf=mdf["confidence"].mean(),
        )

    # Summary comparison table
    if not filter_model and len(models) > 1:
        print(f"\n{'='*80}")
        print("SUMMARY COMPARISON TABLE")
        print(f"{'='*80}")
        print(f"{'Model':<22} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6} {'FPR':>6} {'AvgConf':>8}")
        print("-" * 80)

        for model in models:
            mdf = df[df["model"] == model]
            yt_bin = binarise(mdf["true_label"].tolist())
            yp_bin = binarise(mdf["predicted_label"].tolist())
            valid = [(t, p) for t, p in zip(yt_bin, yp_bin) if p != "error"]
            if not valid:
                continue
            yt, yp = zip(*valid)
            tn, fp, _, _ = confusion_matrix(yt, yp, labels=["legitimate", "scam"]).ravel()
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            print(
                f"{model:<22} "
                f"{accuracy_score(yt, yp):>6.4f} "
                f"{precision_score(yt, yp, pos_label='scam', zero_division=0):>6.4f} "
                f"{recall_score(yt, yp, pos_label='scam', zero_division=0):>6.4f} "
                f"{f1_score(yt, yp, pos_label='scam', zero_division=0):>6.4f} "
                f"{fpr:>6.4f} "
                f"{mdf['confidence'].mean():>8.4f}"
            )
        print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate AiDetective model results.")
    parser.add_argument("--multi", action="store_true", help="Evaluate multi-model comparison results.")
    parser.add_argument("--model", type=str, default=None, help="Filter to a specific model name.")
    args = parser.parse_args()

    if args.multi:
        run_multi(filter_model=args.model)
    else:
        run_single()
