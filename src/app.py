import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from preprocessor import clean_text
from models import GPTDetector, GeminiDetector, ClaudeDetector, DeepSeekDetector
from scraper import scrape_url, extract_text_from_image_bytes, fetch_image_from_url, is_image_url
from meta_scraper import scrape_ad_library

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

DETECTORS = {
    "GPT-4o-mini": GPTDetector,
    "Gemini 1.5 Flash": GeminiDetector,
    "Claude Haiku": ClaudeDetector,
    "DeepSeek Chat": DeepSeekDetector,
}

COUNTRIES = {
    "Australia (AU)": "AU",
    "United States (US)": "US",
    "United Kingdom (GB)": "GB",
    "Canada (CA)": "CA",
    "India (IN)": "IN",
    "Singapore (SG)": "SG",
    "Global (ALL)": "ALL",
}

st.set_page_config(page_title="AiDetective", page_icon="🔍", layout="centered")
st.title("🔍 AiDetective")
st.caption("Meta Ad Scam Detector — Honours Thesis, Macquarie University")
st.divider()


def show_result(result: dict):
    label = result.get("label", "error")
    confidence = result.get("confidence", 0.0)
    reasoning = result.get("reasoning", "")

    st.divider()
    if label == "scam":
        st.error(f"**SCAM** — {confidence * 100:.0f}% confidence")
    elif label == "suspicious":
        st.warning(f"**SUSPICIOUS** — {confidence * 100:.0f}% confidence")
    elif label == "legitimate":
        st.success(f"**LEGITIMATE** — {confidence * 100:.0f}% confidence")
    else:
        st.info(f"**{label.upper()}** — {confidence * 100:.0f}% confidence")

    st.text_area("Reasoning", value=reasoning, height=80, disabled=True)


def run_detector(model_name: str, text: str):
    cleaned = clean_text(text)
    try:
        detector = DETECTORS[model_name]()
        result = detector.detect(cleaned)
    except EnvironmentError as e:
        st.error(f"API key missing: {e}")
        st.stop()
    show_result(result)
    with st.expander("Cleaned text (sent to model)"):
        st.code(cleaned, language=None)


col1, _ = st.columns([2, 1])
with col1:
    model_name = st.selectbox("Model", list(DETECTORS.keys()))

tab_text, tab_url, tab_image, tab_scrape = st.tabs(["Text", "URL", "Image", "Scrape"])

# ── Tab 1: plain text ────────────────────────────────────────────────────────
with tab_text:
    ad_text = st.text_area(
        "Ad Text",
        placeholder="Paste your Meta (Facebook/Instagram) ad text here...",
        height=160,
        label_visibility="collapsed",
    )
    if st.button("Analyze", type="primary", key="btn_text"):
        if not ad_text.strip():
            st.warning("Please enter some ad text.")
        else:
            with st.spinner(f"Analyzing with {model_name}..."):
                run_detector(model_name, ad_text)

# ── Tab 2: URL ───────────────────────────────────────────────────────────────
with tab_url:
    st.caption(
        "Paste any public webpage URL. "
        "Note: Facebook/Instagram posts require login and cannot be scraped directly — "
        "use the [Facebook Ad Library](https://www.facebook.com/ads/library) URL or the Scrape tab instead."
    )
    ad_url = st.text_input("Ad URL", placeholder="https://www.facebook.com/ads/library/?id=...")
    if st.button("Scrape & Analyze", type="primary", key="btn_url"):
        if not ad_url.strip():
            st.warning("Please enter a URL.")
        else:
            with st.spinner("Fetching page..."):
                try:
                    if is_image_url(ad_url):
                        image_bytes, mime = fetch_image_from_url(ad_url)
                        with st.spinner("Extracting text from image..."):
                            extracted = extract_text_from_image_bytes(image_bytes, mime)
                        st.text_area("Extracted text", value=extracted, height=120, disabled=True)
                        with st.spinner(f"Analyzing with {model_name}..."):
                            run_detector(model_name, extracted)
                    else:
                        page_text = scrape_url(ad_url)
                        st.text_area("Scraped text (preview)", value=page_text[:500] + "...", height=100, disabled=True)
                        with st.spinner(f"Analyzing with {model_name}..."):
                            run_detector(model_name, page_text)
                except Exception as e:
                    st.error(f"Could not fetch URL: {e}")

# ── Tab 3: image upload ──────────────────────────────────────────────────────
with tab_image:
    st.caption("Upload a screenshot or photo of an ad. Text is extracted via GPT-4o-mini vision, then analyzed by the selected model.")
    uploaded = st.file_uploader(
        "Upload ad image",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed",
    )
    if uploaded:
        st.image(uploaded, use_container_width=True)
        if st.button("Analyze Image", type="primary", key="btn_image"):
            with st.spinner("Extracting text from image..."):
                try:
                    mime = uploaded.type or "image/jpeg"
                    extracted = extract_text_from_image_bytes(uploaded.read(), mime)
                except EnvironmentError as e:
                    st.error(str(e))
                    st.stop()
                except Exception as e:
                    st.error(f"Image extraction failed: {e}")
                    st.stop()

            st.text_area("Extracted text", value=extracted, height=100, disabled=True)
            with st.spinner(f"Analyzing with {model_name}..."):
                run_detector(model_name, extracted)

# ── Tab 4: Meta Ad Library scraper ──────────────────────────────────────────
with tab_scrape:
    st.caption(
        "Scrapes the public [Facebook Ad Library](https://www.facebook.com/ads/library) "
        "— no login required. First scrape ads, then analyze them all at once."
    )

    col_q, col_country, col_n = st.columns([3, 2, 1])
    with col_q:
        search_term = st.text_input("Search term", placeholder="e.g. crypto investment, free iPhone")
    with col_country:
        country_label = st.selectbox("Country", list(COUNTRIES.keys()))
    with col_n:
        limit = st.number_input("Max ads", min_value=5, max_value=50, value=20, step=5)

    if st.button("Scrape Ads", type="primary", key="btn_scrape"):
        if not search_term.strip():
            st.warning("Please enter a search term.")
        else:
            with st.spinner(f"Opening browser and scraping '{search_term}'… (this takes ~20 s)"):
                try:
                    ads = scrape_ad_library(
                        search_term,
                        country=COUNTRIES[country_label],
                        limit=int(limit),
                    )
                except Exception as e:
                    if "Executable doesn't exist" in str(e) or "playwright install" in str(e).lower():
                        st.error(
                            "Playwright browser not found. Run this once in your terminal:\n\n"
                            "```\nplaywright install chromium\n```"
                        )
                    else:
                        st.error(f"Scraping failed: {e}")
                    st.stop()

            if not ads:
                st.warning("No ads found. Try a different search term or country.")
            else:
                st.session_state["scraped_ads"] = ads
                st.success(f"Scraped {len(ads)} ads.")

    if "scraped_ads" in st.session_state:
        ads = st.session_state["scraped_ads"]

        with st.expander(f"Preview scraped ads ({len(ads)})"):
            for i, ad in enumerate(ads, 1):
                st.markdown(f"**{i}.** {ad['ad_text'][:200]}{'…' if len(ad['ad_text']) > 200 else ''}")
                st.divider()

        if st.button(f"Analyze All {len(ads)} Ads with {model_name}", type="primary", key="btn_analyze_all"):
            try:
                detector = DETECTORS[model_name]()
            except EnvironmentError as e:
                st.error(f"API key missing: {e}")
                st.stop()

            rows = []
            progress = st.progress(0, text="Analyzing…")
            for i, ad in enumerate(ads):
                cleaned = clean_text(ad["ad_text"])
                result = detector.detect(cleaned)
                rows.append({
                    "Ad Text": ad["ad_text"][:120] + ("…" if len(ad["ad_text"]) > 120 else ""),
                    "Label": result.get("label", "error").upper(),
                    "Confidence": f"{result.get('confidence', 0) * 100:.0f}%",
                    "Reasoning": result.get("reasoning", ""),
                })
                progress.progress((i + 1) / len(ads), text=f"Analyzed {i + 1}/{len(ads)}")

            df = pd.DataFrame(rows)

            # Color-code labels
            def color_label(val):
                colors = {"SCAM": "background-color:#f8d7da", "SUSPICIOUS": "background-color:#fff3cd", "LEGITIMATE": "background-color:#d1e7dd"}
                return colors.get(val, "")

            st.dataframe(df.style.map(color_label, subset=["Label"]), use_container_width=True)

            # Save results
            os.makedirs("../outputs", exist_ok=True)
            out_path = "../outputs/scraped_results.csv"
            df.to_csv(out_path, index=False)
            st.caption(f"Results saved to `{out_path}`")
