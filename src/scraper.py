import re
import base64
import os
import requests
from bs4 import BeautifulSoup

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp")


def scrape_url(url: str, timeout: int = 10) -> str:
    """Fetch a URL and return visible text content (max 4000 chars)."""
    resp = requests.get(url, headers=HEADERS, timeout=timeout)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()

    text = soup.get_text(separator=" ", strip=True)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:4000]


def extract_text_from_image_bytes(image_bytes: bytes, mime_type: str = "image/jpeg") -> str:
    """Use GPT-4o-mini vision to extract all text from an ad image."""
    from openai import OpenAI

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY not set — needed for image analysis")

    client = OpenAI(api_key=api_key)
    b64 = base64.b64encode(image_bytes).decode()

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Extract all visible text from this advertisement image. "
                            "Return only the raw text content exactly as it appears, "
                            "no analysis or commentary."
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime_type};base64,{b64}"},
                    },
                ],
            }
        ],
        max_tokens=512,
    )
    return response.choices[0].message.content.strip()


def fetch_image_from_url(url: str) -> tuple[bytes, str]:
    """Download an image URL and return (bytes, mime_type)."""
    resp = requests.get(url, headers=HEADERS, timeout=10)
    resp.raise_for_status()
    mime = resp.headers.get("Content-Type", "image/jpeg").split(";")[0]
    return resp.content, mime


def is_image_url(url: str) -> bool:
    return url.lower().split("?")[0].endswith(IMAGE_EXTENSIONS)
