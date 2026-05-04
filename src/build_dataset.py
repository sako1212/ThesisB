"""
build_dataset.py — Scrape the Meta Ad Library across many search terms and
countries to build a deduplicated thesis dataset.

Strategy
    - Run ~25 scam-relevant search terms across AU and US
    - Each search drops Polish/Spanish/etc ads (english_only)
    - Each search dedupes within itself by ad_text (dedup_by_text)
    - Globally dedupe across the whole run by library_id AND ad_text
    - Save incrementally after every search so a crash never loses progress
    - Stop early once TARGET_UNIQUE unique ads have been collected

Output
    outputs/dataset.csv  - one row per unique ad
"""

import json
import os
import sys
import time
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from meta_scraper import scrape_ad_library

OUTPUT          = "../outputs/dataset.csv"
TARGET_UNIQUE   = 200
PER_SEARCH_LIMIT = 25         # raw rows requested per (term, country)
PER_SEARCH_MAX_SCROLLS = 30   # generous; scroll-until-stable will short-circuit

# Categories chosen to span the 6 scam types in the thesis classifier
# (phishing / investment / impersonation / health / giveaway / other),
# plus a few neutral/legit-leaning queries for the negative class.
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
    """Trim + ASCII-escape a string for Windows console printing."""
    s = (s or "").replace("\n", " ")[:n]
    return s.encode("ascii", "replace").decode("ascii")


def _save(rows: list[dict]):
    df = pd.DataFrame(rows)
    # image_urls is a list-of-dicts; serialise to JSON so CSV survives reload
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

        # Persist after every search so crashes don't lose progress
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
