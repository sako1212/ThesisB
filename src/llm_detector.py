import os
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def detect_scam(ad_text: str) -> dict:
    """
    Stage 1: Detect whether a Meta ad is scam or legitimate.
    """

    prompt = f"""
You are an AI scam detection assistant analysing Meta advertisements.

Classify the advertisement as either:
- scam
- legitimate
- suspicious

Return only valid JSON with:
{{
  "label": "scam | legitimate | suspicious",
  "confidence": 0.0,
  "reasoning": "short explanation"
}}

Advertisement:
{ad_text}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You detect scam advertisements on Meta platforms."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    content = response.choices[0].message.content

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return {
            "label": "error",
            "confidence": 0.0,
            "reasoning": content
        }


def classify_scam(ad_text: str) -> dict:
    """
    Stage 2: Classify scam type if the ad is detected as scam.
    """

    prompt = f"""
You are classifying scam advertisements on Meta Ads.

Classify the scam into one of:
- phishing
- investment
- impersonation
- health
- giveaway
- other

Return only valid JSON:
{{
  "category": "phishing | investment | impersonation | health | giveaway | other",
  "confidence": 0.0,
  "reasoning": "short explanation"
}}

Advertisement:
{ad_text}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You classify scam types in Meta advertisements."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    content = response.choices[0].message.content

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return {
            "category": "error",
            "confidence": 0.0,
            "reasoning": content
        }