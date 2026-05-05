import json
import os
import sys
import time
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from meta_scraper import scrape_ad_library

OUTPUT          = "../outputs/dataset.csv"
TARGET_UNIQUE   = 200
PER_SEARCH_LIMIT = 25
PER_SEARCH_MAX_SCROLLS = 30

SEARCH_TERMS = [
    # phishing / impersonation
    "account verification",
    "secure your account",
    "facebook security alert",
    "official store",
    "ray ban sale",
    # investment
    "crypto investment",
    "bitcoin trading",
    "forex signals",
    "passive income",
    "trading bot",
    "make money online",
    # health
    "weight loss",
    "miracle cure",
    "anti aging",
    "muscle supplement",
    # giveaway
    "free iphone",
    "giveaway",
    "win prize",
    "free gift card",
    # neutral / legit-leaning
    "online course",
    "fitness app",
    "meal prep",
    "travel deal",
    "real estate",
    "credit card",
]
COUNTRIES = ["AU", "US"]


def _safe(s: str, n: int = 80) -> str:
    s = (s or "").replace("\n", " ")[:n]
    return s.encode("ascii", "replace").decode("ascii")


def _save(rows: list[dict]):
    df = pd.DataFrame(rows)
    if "image_urls" in df.columns:
        df["image_urls"] = df["image_urls"].apply(json.dumps)
    df.to_csv(OUTPUT, index=False)


def main():
    os.makedirs("../outputs", exist_ok=True)

    seen_lib: set[str] = set()
    seen_text: set[str] = set()
    rows: list[dict] = []

    pairs = [(t, c) for c in COUNTRIES for t in SEARCH_TERMS]
    total = len(pairs)
    t0 = time.time()

    for i, (term, country) in enumerate(pairs, 1):
        if len(rows) >= TARGET_UNIQUE:
            print(f"\nReached target ({TARGET_UNIQUE}). Stopping early.")
            break

        print(f"\n[{i:>2}/{total}] {country} :: {term!r}  "
              f"(have {len(rows)}/{TARGET_UNIQUE} unique)")
        try:
            ads = scrape_ad_library(
                term,
                country=country,
                limit=PER_SEARCH_LIMIT,
                max_scrolls=PER_SEARCH_MAX_SCROLLS,
                english_only=True,
                dedup_by_text=True,
                headless=True,
            )
        except Exception as e:
            print(f"  ERROR: {type(e).__name__}: {e}")
            continue

        added = 0
        for ad in ads:
            lib = ad.get("library_id")
            txt = ad.get("ad_text") or ""
            if lib and lib in seen_lib:
                continue
            if txt in seen_text:
                continue
            seen_lib.add(lib or f"_no_id_{len(rows)}")
            seen_text.add(txt)
            rows.append({"search_term": term, "country": country, **ad})
            added += 1
            print(f"   + [id={lib}] {_safe(txt, 90)}")
            if len(rows) >= TARGET_UNIQUE:
                break

        kept_pct = (added / max(len(ads), 1)) * 100
        print(f"   scraped {len(ads)}, added {added} unique ({kept_pct:.0f}% new)")

        _save(rows)

    elapsed = time.time() - t0
    print(f"\nDone. {len(rows)} unique ads in {elapsed/60:.1f} min")
    print(f"Saved to {OUTPUT}")

    if rows:
        df = pd.DataFrame(rows)
        print(f"\nBy country: {dict(df['country'].value_counts())}")
        with_img = sum(1 for r in rows if r.get("image_urls"))
        print(f"With ad-creative images: {with_img}/{len(rows)}")


if __name__ == "__main__":
    main()
