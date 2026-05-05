import os
import json
from dotenv import load_dotenv

load_dotenv()

PROVIDERS = ["OPENAI_API_KEY", "GEMINI_API_KEY", "ANTHROPIC_API_KEY", "DEEPSEEK_API_KEY"]


def report(name, ok, detail=""):
    status = "PASS" if ok else "FAIL"
    safe = str(detail).encode("ascii", "replace").decode("ascii")
    print(f"[{status}] {name}: {safe}")


def test_openai():
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        return report("OpenAI GPT-4o-mini", False, "OPENAI_API_KEY missing")
    try:
        from openai import OpenAI
        client = OpenAI(api_key=key)
        r = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=5,
        )
        report("OpenAI GPT-4o-mini", True, r.choices[0].message.content[:30])
    except Exception as e:
        report("OpenAI GPT-4o-mini", False, f"{type(e).__name__}: {e}")


def test_gemini():
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        return report("Gemini 2.5 Flash", False, "GEMINI_API_KEY missing")
    try:
        import google.generativeai as genai
        genai.configure(api_key=key)
        model = genai.GenerativeModel("gemini-2.5-flash")
        r = model.generate_content("say ok in one word", generation_config={"max_output_tokens": 200})
        report("Gemini 2.5 Flash", True, (r.text or "")[:30])
    except Exception as e:
        report("Gemini 2.5 Flash", False, f"{type(e).__name__}: {e}")


def test_anthropic():
    key = os.getenv("ANTHROPIC_API_KEY")
    if not key:
        return report("Claude Haiku 4.5", False, "ANTHROPIC_API_KEY missing")
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=key)
        r = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=10,
            messages=[{"role": "user", "content": "ping"}],
        )
        report("Claude Haiku 4.5", True, r.content[0].text[:30])
    except Exception as e:
        report("Claude Haiku 4.5", False, f"{type(e).__name__}: {e}")


def test_deepseek():
    key = os.getenv("DEEPSEEK_API_KEY")
    if not key:
        return report("DeepSeek Chat", False, "DEEPSEEK_API_KEY missing")
    try:
        from openai import OpenAI
        client = OpenAI(api_key=key, base_url="https://api.deepseek.com")
        r = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": "Reply with the single word: ok"}],
            max_tokens=5,
        )
        report("DeepSeek Chat", True, r.choices[0].message.content[:30])
    except Exception as e:
        report("DeepSeek Chat", False, f"{type(e).__name__}: {e}")


def test_deepseek_pipeline():
    print("\n--- DeepSeek end-to-end pipeline test ---")
    try:
        from models import DeepSeekDetector
        det = DeepSeekDetector()
    except Exception as e:
        report("DeepSeek init", False, f"{type(e).__name__}: {e}")
        return

    ad = "Earn $5,000/week guaranteed - DM me 'START' to join. Limited spots!"

    detection = det.detect(ad)
    label_ok = detection.get("label") in {"scam", "suspicious", "legitimate"}
    has_score = isinstance(detection.get("evaluationScore"), float)
    has_reason = bool(detection.get("reasoningSummary"))
    report(
        "DeepSeek detect()",
        label_ok and has_score and has_reason,
        f"label={detection.get('label')!r} score={detection.get('evaluationScore')} "
        f"flagged={detection.get('isScamFlagged')}",
    )

    if not detection.get("isScamFlagged"):
        print("  (skipping classify - ad was not flagged)")
        return

    classification = det.classify(ad)
    cat_ok = classification.get("scamCategory") in {
        "phishing", "investment", "impersonation", "health", "giveaway", "other",
    }
    report(
        "DeepSeek classify()",
        cat_ok,
        f"category={classification.get('scamCategory')!r} score={classification.get('classificationScore')}",
    )


if __name__ == "__main__":
    print("Loaded keys (first 12 chars):")
    for k in PROVIDERS:
        v = os.getenv(k)
        print(f"  {k}: {v[:12] + '...' if v else '<missing>'}")
    print()

    test_openai()
    test_gemini()
    test_anthropic()
    test_deepseek()
    test_deepseek_pipeline()
