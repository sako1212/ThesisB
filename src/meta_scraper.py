import re
from urllib.parse import quote

_UI_STRINGS = re.compile(
    r"^(see more|see less|see summary details|sponsored|learn more|shop now|"
    r"sign up|contact us|get offer|apply now|watch more|book now|donate now|"
    r"send message|follow|like|share|comment|about this ad|"
    r"this ad is not active|started running|active since|library id|"
    r"all results|no results|filters|sort by|reset).*",
    re.IGNORECASE | re.DOTALL,
)

_MIN_IMAGE_DIM = 100


def _is_ad_text(text: str) -> bool:
    return len(text) >= 30 and not _UI_STRINGS.match(text)


def _is_english(text: str) -> bool:
    try:
        from langdetect import detect, DetectorFactory
        DetectorFactory.seed = 0
        return detect(text[:600]) == "en"
    except Exception:
        return True


def scrape_ad_library(
    search_term: str,
    country: str = "AU",
    limit: int = 20,
    headless: bool = True,
    max_scrolls: int = 25,
    english_only: bool = False,
    dedup_by_text: bool = False,
) -> list[dict]:
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

        def _count_ids() -> int:
            return page.evaluate(
                r"() => (document.body.innerText.match(/Library ID:?\s*\d+/g) || []).length"
            )

        _expand_see_more()

        prev = -1
        stable = 0
        for i in range(max_scrolls):
            page.evaluate("window.scrollBy(0, 1500)")
            page.wait_for_timeout(1300)
            _expand_see_more()
            cur = _count_ids()
            if cur >= limit + 5:
                break
            if cur == prev:
                stable += 1
                if stable >= 3:
                    break
            else:
                stable = 0
            prev = cur

        _expand_see_more()
        page.wait_for_timeout(800)

        cards = page.evaluate(
            r"""() => {
                const re = /Library ID:?\s*(\d+)/;
                const allEls = Array.from(document.querySelectorAll('*'));
                // Leaf text nodes that mention Library ID
                const idEls = allEls.filter(el =>
                    el.children.length === 0
                    && re.test(el.textContent || '')
                );

                const seen = new Set();
                const out = [];

                for (const idEl of idEls) {
                    let card = idEl;
                    for (let i = 0; i < 7 && card.parentElement; i++) {
                        card = card.parentElement;
                    }
                    if (seen.has(card)) continue;
                    seen.add(card);

                    const m = (idEl.textContent || '').match(re);
                    const libraryId = m ? m[1] : null;

                    const bodySpans = Array.from(card.querySelectorAll('span'))
                        .filter(s => s.children.length === 0
                                     && s.className.trim() === ''
                                     && s.textContent.trim().length >= 30);
                    const seenText = new Set();
                    const bodyParts = [];
                    for (const s of bodySpans) {
                        const t = s.textContent.trim();
                        if (!seenText.has(t)) {
                            seenText.add(t);
                            bodyParts.push(t);
                        }
                    }
                    const adText = bodyParts.join('\n');

                    const imgs = Array.from(card.querySelectorAll('img'))
                        .map(i => ({
                            src: i.src,
                            width: i.naturalWidth || i.width || 0,
                            height: i.naturalHeight || i.height || 0,
                        }))
                        .filter(o => o.src && (o.src.includes('scontent') || o.src.includes('fbcdn')));

                    out.push({
                        library_id: libraryId,
                        ad_text: adText,
                        image_urls: imgs,
                    });
                }
                return out;
            }"""
        )

        results: list[dict] = []
        seen_text: set[str] = set()
        for c in cards:
            text = (c.get("ad_text") or "").strip()
            if not _is_ad_text(text):
                continue
            if dedup_by_text and text in seen_text:
                continue
            if english_only and not _is_english(text):
                continue

            imgs = [
                img for img in (c.get("image_urls") or [])
                if (img.get("width") or 0) >= _MIN_IMAGE_DIM
                and (img.get("height") or 0) >= _MIN_IMAGE_DIM
            ]

            lib_id = c.get("library_id")
            ad_url = (
                f"https://www.facebook.com/ads/library/?id={lib_id}"
                if lib_id else None
            )
            results.append({
                "library_id":  lib_id,
                "ad_url":      ad_url,
                "ad_text":     text,
                "image_urls":  imgs,
                "search_url":  search_url,
                "source_url":  search_url,
            })
            seen_text.add(text)
            if len(results) >= limit:
                break

        browser.close()

    return results
