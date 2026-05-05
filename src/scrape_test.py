import os
import sys
import time
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from meta_scraper import scrape_ad_library

SEARCHES = [
    "crypto investment",
    "free iphone giveaway",
    "weight loss miracle",
]

COUNTRY = "AU"
LIMIT_PER_TERM = 50
OUTPUT = "../outputs/scraped_test.csv"


def main():
    os.makedirs("../outputs", exist_ok=True)
    all_rows = []

    for term in SEARCHES:
        print(f"\n--- Scraping: {term!r} ({COUNTRY}, max {LIMIT_PER_TERM}) ---")
        t0 = time.time()
        try:
            ads = scrape_ad_library(term, country=COUNTRY, limit=LIMIT_PER_TERM, headless=True)
        except Exception as e:
            print(f"  ERROR: {type(e).__name__}: {e}")
            continue
        dt = time.time() - t0
        print(f"  Got {len(ads)} ads in {dt:.1f}s")
        for i, ad in enumerate(ads, 1):
            preview = ad["ad_text"][:100].replace("\n", " ")
            safe = (preview + ("..." if len(ad["ad_text"]) > 100 else "")).encode("ascii", "replace").decode("ascii")
            n_imgs = len(ad.get("image_urls") or [])
            lib = ad.get("library_id") or "?"
            print(f"  {i:>2}. [id={lib} imgs={n_imgs}] {safe}")
            all_rows.append({"search_term": term, "country": COUNTRY, **ad})

    if not all_rows:
        print("\nNo ads scraped. Check Playwright install or network.")
        return

    df = pd.DataFrame(all_rows)
    df.to_csv(OUTPUT, index=False)
    print(f"\nSaved {len(df)} ads across {df['search_term'].nunique()} searches to {OUTPUT}")


if __name__ == "__main__":
    main()
