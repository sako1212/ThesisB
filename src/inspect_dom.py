"""
inspect_dom.py — One-off probe to confirm the ad card structure on the
Facebook Ad Library page. Prints structural hints so we can build a robust
selector.
"""

from urllib.parse import quote
from playwright.sync_api import sync_playwright

URL = (
    "https://www.facebook.com/ads/library/"
    "?active_status=all&ad_type=all&country=AU"
    f"&q={quote('crypto investment')}&search_type=keyword_unordered"
)

with sync_playwright() as pw:
    browser = pw.chromium.launch(headless=True, args=["--no-sandbox"])
    page = browser.new_context(viewport={"width": 1280, "height": 900}).new_page()
    page.goto(URL, wait_until="domcontentloaded", timeout=30000)
    page.wait_for_timeout(5000)

    # Dismiss cookie
    for sel in [
        '[data-cookiebanner="accept_only_essential_button"]',
        'button:has-text("Only allow essential cookies")',
        'button:has-text("Decline optional cookies")',
    ]:
        try:
            if page.is_visible(sel, timeout=1000):
                page.click(sel)
                break
        except Exception:
            pass

    page.wait_for_timeout(3000)
    for _ in range(3):
        page.evaluate("window.scrollBy(0, 1000)")
        page.wait_for_timeout(1500)

    info = page.evaluate(
        r"""() => {
            const out = {};
            // 1. Articles?
            out.role_article = document.querySelectorAll('div[role="article"]').length;
            // 2. How many "Library ID" mentions?
            const libraryIdRe = /Library ID:?\s*(\d+)/g;
            const matches = [...document.body.innerText.matchAll(libraryIdRe)];
            out.library_id_count = matches.length;
            out.library_id_first5 = matches.slice(0, 5).map(m => m[1]);
            // 3. Find ancestor of a Library ID text node, walk up levels
            const allEls = Array.from(document.querySelectorAll('*'));
            const idEl = allEls.find(el => el.children.length === 0
                                            && /Library ID/i.test(el.textContent || ''));
            if (idEl) {
                const ancestors = [];
                let cur = idEl;
                for (let i = 0; i < 12; i++) {
                    if (!cur.parentElement) break;
                    cur = cur.parentElement;
                    ancestors.push({
                        level: i + 1,
                        tag: cur.tagName.toLowerCase(),
                        role: cur.getAttribute('role') || '',
                        textLen: (cur.textContent || '').length,
                        imgs: cur.querySelectorAll('img').length,
                        children: cur.children.length,
                    });
                }
                out.ancestors = ancestors;
            }
            // 4. Image hints
            out.imgs_total = document.querySelectorAll('img').length;
            out.imgs_scontent = [...document.querySelectorAll('img')]
                .map(i => i.src)
                .filter(s => s.includes('scontent') || s.includes('fbcdn'))
                .length;
            return out;
        }"""
    )

    print("DOM probe results:")
    for k, v in info.items():
        print(f"  {k}: {v}")

    browser.close()
