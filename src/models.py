"""
models.py — Unified LLM detector wrappers for multi-model scam detection comparison.

Each detector exposes a single method:
    detect(ad_text: str) -> dict
        {"label": "scam|suspicious|legitimate", "confidence": float, "reasoning": str}

All models use the same prompt so results are directly comparable.
"""

import os
import json
import re
from abc import ABC, abstractmethod
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Shared prompt (based on thesis Appendix A)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are an advanced AI content analyst specialising in identifying and "
    "classifying scam advertisements on Meta platforms (Facebook, Instagram). "
    "Your role is to determine whether an advertisement is legitimate, suspicious, "
    "or a scam based on linguistic cues, manipulative intent, and deceptive patterns."
)

DETECTION_PROMPT = """Analyse the following Meta advertisement text and classify it.

Advertisement Text:
\"\"\"{ad_text}\"\"\"

Instructions:
1. Identify indicators of deception such as: unrealistic financial promises, urgency tactics, impersonation of trusted figures, requests for personal/financial information, or emotionally manipulative language.
2. Classify the advertisement as exactly ONE of:
   - legitimate  (no scam indicators)
   - suspicious  (potentially misleading or ambiguous)
   - scam        (confirmed fraudulent intent)
3. Provide a confidence score between 0.0 (no confidence) and 1.0 (high confidence).
4. Include a brief reasoning statement (1-2 sentences).

Return ONLY valid JSON with no markdown, no code blocks, no extra text:
{{
  "label": "legitimate | suspicious | scam",
  "confidence": 0.0,
  "reasoning": "brief explanation"
}}"""


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class BaseDetector(ABC):
    name: str = "base"

    @abstractmethod
    def detect(self, ad_text: str) -> dict:
        pass

    def _build_prompt(self, ad_text: str) -> str:
        return DETECTION_PROMPT.format(ad_text=ad_text)

    def _parse_response(self, content: str) -> dict:
        """Extract JSON from model response, handling markdown code fences."""
        # Strip markdown code fences if present
        content = re.sub(r"```(?:json)?\s*", "", content).strip().rstrip("`").strip()

        try:
            return json.loads(content)
        except json.JSONDecodeError:
            # Try to find a JSON object anywhere in the response
            match = re.search(r"\{.*?\}", content, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    pass

        return {
            "label": "error",
            "confidence": 0.0,
            "reasoning": f"Failed to parse model response: {content[:200]}",
        }


# ---------------------------------------------------------------------------
# 1. OpenAI GPT-4o-mini
# ---------------------------------------------------------------------------

class GPTDetector(BaseDetector):
    name = "GPT-4o-mini"

    def __init__(self):
        from openai import OpenAI
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY not set in .env")
        self.client = OpenAI(api_key=api_key)

    def detect(self, ad_text: str) -> dict:
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": self._build_prompt(ad_text)},
                ],
                temperature=0,
                max_tokens=256,
            )
            return self._parse_response(response.choices[0].message.content)
        except Exception as e:
            return {"label": "error", "confidence": 0.0, "reasoning": str(e)}


# ---------------------------------------------------------------------------
# 2. Google Gemini 1.5 Flash
# ---------------------------------------------------------------------------

class GeminiDetector(BaseDetector):
    name = "Gemini 1.5 Flash"

    def __init__(self):
        import google.generativeai as genai
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise EnvironmentError("GEMINI_API_KEY not set in .env")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            system_instruction=SYSTEM_PROMPT,
        )

    def detect(self, ad_text: str) -> dict:
        try:
            response = self.model.generate_content(
                self._build_prompt(ad_text),
                generation_config={"temperature": 0, "max_output_tokens": 256},
            )
            return self._parse_response(response.text)
        except Exception as e:
            return {"label": "error", "confidence": 0.0, "reasoning": str(e)}


# ---------------------------------------------------------------------------
# 3. Anthropic Claude Haiku
# ---------------------------------------------------------------------------

class ClaudeDetector(BaseDetector):
    name = "Claude Haiku"

    def __init__(self):
        import anthropic
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise EnvironmentError("ANTHROPIC_API_KEY not set in .env")
        self.client = anthropic.Anthropic(api_key=api_key)

    def detect(self, ad_text: str) -> dict:
        try:
            response = self.client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=256,
                system=SYSTEM_PROMPT,
                messages=[
                    {"role": "user", "content": self._build_prompt(ad_text)},
                ],
            )
            return self._parse_response(response.content[0].text)
        except Exception as e:
            return {"label": "error", "confidence": 0.0, "reasoning": str(e)}


# ---------------------------------------------------------------------------
# 4. DeepSeek Chat (OpenAI-compatible endpoint)
# ---------------------------------------------------------------------------

class DeepSeekDetector(BaseDetector):
    name = "DeepSeek Chat"

    def __init__(self):
        from openai import OpenAI
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise EnvironmentError("DEEPSEEK_API_KEY not set in .env")
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com",
        )

    def detect(self, ad_text: str) -> dict:
        try:
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": self._build_prompt(ad_text)},
                ],
                temperature=0,
                max_tokens=256,
            )
            return self._parse_response(response.choices[0].message.content)
        except Exception as e:
            return {"label": "error", "confidence": 0.0, "reasoning": str(e)}


# ---------------------------------------------------------------------------
# 5. Groq — Llama 3.1 8B Instant (free tier available)
# ---------------------------------------------------------------------------

class GroqDetector(BaseDetector):
    name = "Llama 3.1 (Groq)"

    def __init__(self):
        from groq import Groq
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise EnvironmentError("GROQ_API_KEY not set in .env")
        self.client = Groq(api_key=api_key)

    def detect(self, ad_text: str) -> dict:
        try:
            response = self.client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": self._build_prompt(ad_text)},
                ],
                temperature=0,
                max_tokens=256,
            )
            return self._parse_response(response.choices[0].message.content)
        except Exception as e:
            return {"label": "error", "confidence": 0.0, "reasoning": str(e)}


# ---------------------------------------------------------------------------
# Registry — add/remove models here
# ---------------------------------------------------------------------------

ALL_DETECTORS = [
    GPTDetector,
    GeminiDetector,
    ClaudeDetector,
    DeepSeekDetector,
    GroqDetector,
]


def load_detectors(skip_missing: bool = True) -> list:
    """
    Instantiate all detectors. If skip_missing=True, detectors whose API key
    is not set are silently skipped instead of raising an error.
    """
    detectors = []
    for cls in ALL_DETECTORS:
        try:
            detectors.append(cls())
            print(f"  [OK] {cls.name}")
        except EnvironmentError as e:
            if skip_missing:
                print(f"  [SKIP] {cls.name}: {e}")
            else:
                raise
    return detectors
