"""
meta_scraper.py — Scrapes public ads from the Facebook Ad Library using Playwright.

Key finding: ad body text lands in classless <span> leaf nodes; all UI / metadata
text has hashed CSS class names. Selecting spans where className == "" gives clean
ad copy with no extra filtering needed beyond a minimum length check.
"""

import re
from urllib.parse import quote

_UI_STRINGS = re.compile(
    r"^(see more|see less|see summary details|sponsored|learn more|shop now|"
    r"sign up|contact us|get offer|apply now|watch more|book now|donate now|"
    r"send message|follow|like|share|comment|about this ad|"
    r"this ad is not active|started running|active since|"
    r"all results|no results|filters|sort by|reset).*",
    re.IGNORECASE | re.DOTALL,
)


def _is_ad_text(text: str) -> bool:
    return len(text) >= 30 and not _UI_STRINGS.match(text)


def scrape_ad_library(
    search_term: str,
    country: str = "AU",
    limit: int = 20,
    headless: bool = True,
) -> list[dict]:
    """
    Scrape public ads from the Facebook Ad Library.
    Returns list of dicts: {"ad_text": str, "source_url": str}
    """
    from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout

    search_url = (
        "https://www.facebook.com/ads/library/"
        f"?active_status=all&ad_type=all&country={country}"
        f"&q={quote(search_term)}&search_type=keyword_unordered"
    )

    with sync_playwright() as pw:
        browser = pw.chromium.launch(
            headless=headless,
            args=["--disable-blink-features=AutomationControlled", "--no-sandbox"],
        )
        context = browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            viewport={"width": 1280, "height": 900},
            locale="en-US",
        )
        page = context.new_page()
        page.add_init_script(
            "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
        )

        try:
            page.goto(search_url, wait_until="domcontentloaded", timeout=30000)
        except PWTimeout:
            pass

        page.wait_for_timeout(4000)

        # Dismiss cookie / consent dialogs
        for sel in [
            '[data-cookiebanner="accept_only_essential_button"]',
            'button:has-text("Only allow essential cookies")',
            'button:has-text("Allow essential and optional cookies")',
            'button:has-text("Accept All")',
            'button:has-text("Decline optional cookies")',
        ]:
            try:
                if page.is_visible(sel, timeout=1200):
                    page.click(sel)
                    page.wait_for_timeout(800)
                    break
            except Exception:
                continue

        page.wait_for_timeout(2000)

        def _expand_see_more():
            for btn in page.query_selector_all('div[role="button"]:has-text("See More")'):
                try:
                    btn.scroll_into_view_if_needed()
                    btn.click(timeout=500)
                    page.wait_for_timeout(120)
                except Exception:
                    pass

        _expand_see_more()

        # Scroll to lazy-load more ads
        for _ in range(6):
            page.evaluate("window.scrollBy(0, 1200)")
            page.wait_for_timeout(1300)
            _expand_see_more()

        # ── Extract classless <span> leaf nodes (ad body text) ───────────────
        texts: list[str] = page.evaluate(
            """() =>
                Array.from(document.querySelectorAll('span'))
                    .filter(el => el.children.length === 0
                                  && el.className.trim() === ''
                                  && el.textContent.trim().length >= 30)
                    .map(el => el.textContent.trim())
            """
        )

        seen: set[str] = set()
        results: list[dict] = []
        for text in texts:
            if _is_ad_text(text) and text not in seen:
                seen.add(text)
                results.append({"ad_text": text, "source_url": search_url})
                if len(results) >= limit:
                    break

        browser.close()

    return results
