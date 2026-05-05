import io
import os
import sys
import textwrap
import webbrowser
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

INPUT  = "../outputs/dataset.csv"
OUTPUT = "../outputs/dataset_labelled.csv"

LABELS = {
    "s": "scam",
    "?": "suspicious",
    "l": "legitimate",
}
CATEGORIES = {
    "p": "phishing",
    "i": "investment",
    "m": "impersonation",
    "h": "health",
    "g": "giveaway",
    "o": "other",
}


def load_state() -> pd.DataFrame:
    if os.path.exists(OUTPUT):
        df = pd.read_csv(OUTPUT)
        print(f"Resuming from {OUTPUT}")
    else:
        df = pd.read_csv(INPUT)
        df["ad_id"] = df["library_id"]
        df["true_label"] = ""
        df["true_category"] = ""
        df["notes"] = ""
        print(f"Starting fresh from {INPUT}")
    for col in ("ad_id", "true_label", "true_category", "notes"):
        if col not in df.columns:
            df[col] = ""
    df["ad_id"] = df["library_id"]
    df = df.fillna({"true_label": "", "true_category": "", "notes": ""})
    return df


def save(df: pd.DataFrame):
    tmp = OUTPUT + ".tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, OUTPUT)


def show_stats(df: pd.DataFrame):
    done = (df["true_label"] != "").sum()
    print(f"\n--- Progress: {done}/{len(df)} labelled ({done/len(df)*100:.0f}%) ---")
    if done:
        print("Label counts: " + dict(df[df["true_label"] != ""]["true_label"].value_counts()).__repr__())
        cats = df[df["true_category"].isin(CATEGORIES.values())]["true_category"].value_counts()
        if len(cats):
            print("Category counts: " + dict(cats).__repr__())
    print()


def show_ad(idx: int, total: int, row) -> None:
    print("\n" + "=" * 80)
    print(f"  [{idx + 1}/{total}]  search='{row['search_term']}'  country={row['country']}  id={row['library_id']}")
    print(f"  url: {row['ad_url']}")
    print("-" * 80)
    print(textwrap.fill(str(row["ad_text"]), width=78, initial_indent="  ", subsequent_indent="  "))
    print("=" * 80)


def prompt(msg: str) -> str:
    try:
        return input(msg).strip().lower()
    except (EOFError, KeyboardInterrupt):
        return "q"


def label_one(row, df, idx) -> str:
    while True:
        choice = prompt("  [s]cam  [?]suspicious  [l]egitimate  [k]skip  [b]ack  [o]pen-url  [?]stats  [q]uit  > ")

        if choice == "q":
            return "quit"
        if choice == "b":
            return "back"
        if choice == "k":
            return "next"
        if choice == "?":
            show_stats(df)
            continue
        if choice == "o":
            url = row.get("ad_url")
            if url:
                webbrowser.open(url)
                print(f"  opened {url}")
            else:
                print("  no URL on this ad")
            continue

        if choice not in LABELS:
            print("  invalid key, try again")
            continue

        label = LABELS[choice]
        df.at[idx, "true_label"] = label

        if label == "legitimate":
            df.at[idx, "true_category"] = "N/A"
        else:
            cat = ""
            while not cat:
                ck = prompt(
                    "    category: [p]hishing [i]nvestment [m]impersonation "
                    "[h]ealth [g]iveaway [o]ther > "
                )
                if ck in CATEGORIES:
                    cat = CATEGORIES[ck]
                else:
                    print("    invalid category key")
            df.at[idx, "true_category"] = cat

        note = prompt("    notes (optional, Enter to skip): ")
        if note:
            df.at[idx, "notes"] = note

        save(df)
        return "next"


def main():
    if not os.path.exists(INPUT):
        print(f"ERROR: {INPUT} not found — run build_dataset.py first")
        return

    df = load_state()
    show_stats(df)

    pending = df.index[df["true_label"] == ""].tolist()
    if not pending:
        print("All ads already labelled. Nothing to do.")
        return

    print(f"\n{len(pending)} ads to label. Ctrl+C or 'q' to save and exit.\n")
    history: list[int] = []

    cursor = 0
    while cursor < len(pending):
        idx = pending[cursor]
        show_ad(cursor, len(pending), df.loc[idx])
        action = label_one(df.loc[idx], df, idx)

        if action == "quit":
            print("\nSaved. Resume any time.")
            break
        if action == "back":
            if not history:
                print("  no previous label to go back to")
                continue
            prev_idx = history.pop()
            df.at[prev_idx, "true_label"] = ""
            df.at[prev_idx, "true_category"] = ""
            df.at[prev_idx, "notes"] = ""
            save(df)
            pending.insert(cursor, prev_idx)
            continue
        history.append(idx)
        cursor += 1

    show_stats(df)
    print(f"Saved to {OUTPUT}")


if __name__ == "__main__":
    main()
