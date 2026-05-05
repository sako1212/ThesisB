import os
import json
import re
from abc import ABC, abstractmethod
from dotenv import load_dotenv

load_dotenv()

SYSTEM_PROMPT_DETECT = (
    "You are an advanced AI content analyst specialising in identifying and "
    "classifying scam advertisements on Meta platforms (Facebook, Instagram). "
    "Your role is to determine whether an advertisement is legitimate, suspicious, "
    "or a scam based on linguistic cues, manipulative intent, and deceptive patterns."
)

SYSTEM_PROMPT_CLASSIFY = (
    "You are a scam-typology analyst. An upstream detector has already flagged the "
    "advertisement as scam or suspicious; your job is to assign the most likely "
    "scam category based on the primary deception vector."
)

DETECTION_PROMPT = """You are an expert scam-detector AI analysing Meta advertisements (Facebook/Instagram).

Look for these indicators of deception:
- Urgency / pressure ("act now", countdowns, scarcity)
- Unrealistic promises (guaranteed returns, miracle cures, free money)
- Suspicious CTAs (DM-only contact, off-platform links, "click here to verify")
- Impersonation (posing as a brand, celebrity, government, or platform)
- Emotional manipulation / social engineering (fear, shame, fake authority, fabricated stories)

Crucial nuance - analyse intent, not keywords:
1. Do NOT flag a post solely because it contains words like "free", "earn", "giveaway", "crypto", "phishing", or "$SCAM". Judge what the post is *trying to do*.
2. Sarcasm, jokes, and parody are NOT scams. "Stop reporting my phishing sites :(" is a joke. A post defining "Fee = Scam" is informational.
3. News reports, opinions, and questions are NOT scams. "Massive online leak reported" or "What should I invest in?" are discussion, not deception.

Examples:

Ad: "Earn $5,000/week guaranteed - DM me 'START' to join. Limited spots!"
Output: {{"label": "scam", "evaluationScore": 0.95, "reasoningSummary": "Guaranteed unrealistic income with urgency and DM-only CTA."}}

Ad: "Get 20% off running shoes this weekend at our official Nike store."
Output: {{"label": "legitimate", "evaluationScore": 0.92, "reasoningSummary": "Standard retail promotion from a named brand, no deceptive indicators."}}

Ad: "lol another 'crypto millionaire' scam in my feed, when will Meta actually do something"
Output: {{"label": "legitimate", "evaluationScore": 0.88, "reasoningSummary": "User is complaining about scams, not perpetrating one - opinion, not deception."}}

Ad: "Your account will be disabled in 24h. Verify your password here: bit.ly/fb-secure"
Output: {{"label": "scam", "evaluationScore": 0.97, "reasoningSummary": "Impersonates the platform with urgency and an off-platform credential capture link."}}

Now analyse this advertisement:

\"\"\"{ad_text}\"\"\"

Return ONLY valid JSON, no markdown, no code fences:
{{
  "label": "scam | suspicious | legitimate",
  "evaluationScore": 0.0,
  "reasoningSummary": "1-2 sentence explanation"
}}"""

CLASSIFICATION_PROMPT = """You are categorising a Meta advertisement that an upstream detector has already flagged as scam or suspicious. Assign the most likely scam *type*.

Categories:
- phishing      - credential theft, fake login/verify pages, "account disabled" scares
- investment    - guaranteed returns, crypto/forex/trading "systems", get-rich-quick
- impersonation - posing as a brand, celebrity, government agency, or platform
- health        - miracle cures, fake supplements, weight-loss / anti-aging fraud
- giveaway      - fake prizes, lotteries, "you've won" claims requiring fees or info
- other         - clearly a scam but doesn't fit the categories above

Rules:
1. Pick the *primary* deception vector. A phishing page posing as PayPal is "phishing", not "impersonation" - the credential theft is the goal; impersonation is just the dressing.
2. Use "other" rather than forcing a poor fit. Honest "other" beats low-confidence guessing.
3. Categorise on intent and mechanics, not keywords. An ad mentioning "Bitcoin" isn't automatically "investment" - it could be phishing for a wallet.

Examples:

Ad: "Your Facebook account will be disabled in 24h. Verify password here: bit.ly/fb-secure"
Output: {{"scamCategory": "phishing", "classificationScore": 0.96, "explanationTrace": "Impersonates Meta and drives to off-platform credential capture - phishing is the operative mechanic."}}

Ad: "Elon Musk is giving away 5,000 BTC! Send 0.1 BTC to verify your wallet and receive 1 BTC back."
Output: {{"scamCategory": "giveaway", "classificationScore": 0.93, "explanationTrace": "Fake celebrity giveaway requiring up-front payment - the giveaway pretext is the core lure."}}

Ad: "I made $40k in 3 weeks with this AI trading bot - DM for access."
Output: {{"scamCategory": "investment", "classificationScore": 0.94, "explanationTrace": "Guaranteed unrealistic trading returns with DM-only access - investment-fraud pattern."}}

Now categorise this advertisement:

\"\"\"{ad_text}\"\"\"

Return ONLY valid JSON, no markdown, no code fences:
{{
  "scamCategory": "phishing | investment | impersonation | health | giveaway | other",
  "classificationScore": 0.0,
  "explanationTrace": "1-2 sentence explanation"
}}"""


VALID_LABELS = {"scam", "suspicious", "legitimate"}
VALID_CATEGORIES = {"phishing", "investment", "impersonation", "health", "giveaway", "other"}


class BaseDetector(ABC):
    name: str = "base"

    @abstractmethod
    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        ...

    def detect(self, ad_text: str) -> dict:
        try:
            content = self._call_llm(SYSTEM_PROMPT_DETECT, DETECTION_PROMPT.format(ad_text=ad_text))
        except Exception as e:
            return self._detection_error(str(e))

        parsed = self._parse_json(content)
        if parsed is None:
            return self._detection_error(f"Failed to parse: {content[:200]}")

        label = str(parsed.get("label", "")).strip().lower()
        if label not in VALID_LABELS:
            return self._detection_error(f"Invalid label '{label}' in: {content[:200]}")

        return {
            "label": label,
            "isScamFlagged": label != "legitimate",
            "evaluationScore": float(parsed.get("evaluationScore", 0.0)),
            "reasoningSummary": str(parsed.get("reasoningSummary", "")),
        }

    def classify(self, ad_text: str) -> dict:
        try:
            content = self._call_llm(SYSTEM_PROMPT_CLASSIFY, CLASSIFICATION_PROMPT.format(ad_text=ad_text))
        except Exception as e:
            return self._classification_error(str(e))

        parsed = self._parse_json(content)
        if parsed is None:
            return self._classification_error(f"Failed to parse: {content[:200]}")

        category = str(parsed.get("scamCategory", "")).strip().lower()
        if category not in VALID_CATEGORIES:
            return self._classification_error(f"Invalid category '{category}' in: {content[:200]}")

        return {
            "scamCategory": category,
            "classificationScore": float(parsed.get("classificationScore", 0.0)),
            "explanationTrace": str(parsed.get("explanationTrace", "")),
        }

    @staticmethod
    def _parse_json(content: str):
        content = re.sub(r"```(?:json)?\s*", "", content).strip().rstrip("`").strip()
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", content, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    return None
        return None

    @staticmethod
    def _detection_error(msg: str) -> dict:
        return {
            "label": "error",
            "isScamFlagged": False,
            "evaluationScore": 0.0,
            "reasoningSummary": msg,
        }

    @staticmethod
    def _classification_error(msg: str) -> dict:
        return {
            "scamCategory": "error",
            "classificationScore": 0.0,
            "explanationTrace": msg,
        }


class GPTDetector(BaseDetector):
    name = "GPT-4o-mini"

    def __init__(self):
        from openai import OpenAI
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY not set in .env")
        self.client = OpenAI(api_key=api_key)

    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
            max_tokens=2048,
        )
        return response.choices[0].message.content


class GeminiDetector(BaseDetector):
    name = "Gemini 2.5 Flash"

    def __init__(self):
        import google.generativeai as genai
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise EnvironmentError("GEMINI_API_KEY not set in .env")
        genai.configure(api_key=api_key)
        self._genai = genai

    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        model = self._genai.GenerativeModel(
            model_name="gemini-2.5-flash",
            system_instruction=system_prompt,
        )
        response = model.generate_content(
            user_prompt,
            generation_config={"temperature": 0, "max_output_tokens": 2048},
        )
        return response.text


class ClaudeDetector(BaseDetector):
    name = "Claude Haiku"

    def __init__(self):
        import anthropic
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise EnvironmentError("ANTHROPIC_API_KEY not set in .env")
        self.client = anthropic.Anthropic(api_key=api_key)

    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        response = self.client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=2048,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return response.content[0].text


class DeepSeekDetector(BaseDetector):
    name = "DeepSeek Chat"

    def __init__(self):
        from openai import OpenAI
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise EnvironmentError("DEEPSEEK_API_KEY not set in .env")
        self.client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        response = self.client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
            max_tokens=2048,
        )
        return response.choices[0].message.content


ALL_DETECTORS = [
    GPTDetector,
    GeminiDetector,
    ClaudeDetector,
    DeepSeekDetector,
]


def load_detectors(skip_missing: bool = True) -> list:
    detectors = []
    for cls in ALL_DETECTORS:
        try:
            detectors.append(cls())
            print(f"  [OK] {cls.name}")
        except (EnvironmentError, ModuleNotFoundError, ImportError) as e:
            if skip_missing:
                print(f"  [SKIP] {cls.name}: {e}")
            else:
                raise
    return detectors
