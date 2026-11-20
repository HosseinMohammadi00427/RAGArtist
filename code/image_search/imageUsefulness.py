"""
visual_usefulness_screen.py

Use GPT-4o for direct visual evaluation of images via URLs.
Scores images for usefulness as GenAI base references.
"""

import os
import time
import json
import requests
from typing import List, Dict, Optional
import openai

# OpenAI Connection
openai.api_key = os.environ.get("OPENAI_API_KEY")
gpt_client = openai.OpenAI()

def call_gpt_vision(image_url: str, concept:str,
                     target_caption: str, timeout=60) -> Dict:
    """
    Sends the image URL + text instruction to GPT-4o for evaluation.
    Returns JSON {score, rationale}.
    """
    messages = [
        {
            "role": "system",
            "content": "You are an expert visual analyst evaluating image composition for generative AI training.",
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        f"Evaluate this image for its usefulness as a visual reference for concept {concept},"
                        f"being used to generate an image described as:\n\n'{target_caption}'\n\n"
                        "Focus on spatial layout, posture, camera angle, realism, and absence of distortions or filters. "
                        "Give a usefulness score (1–10) and a one-sentence rationale. "
                        "Return strictly JSON with keys: score, rationale."
                    ),
                },
                {"type": "image_url", "image_url": {"url": image_url}},
            ],
        },
    ]


    r = gpt_client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        response_format={"type": "text"},
        temperature=0 # for less randomness
    )
    output = r.choices

    # Parse JSON
    try:
        start, end = output.find("{"), output.rfind("}")
        parsed = json.loads(output[start:end + 1])
        return {
            "score": parsed.get("score"),
            "rationale": parsed.get("rationale"),
            "raw_output": output,
        }
    except Exception:
        return {"score": None, "rationale": f"Unparsable output: {output}", "raw_output": output}


def screen_image_urls(urls: List[str], concept: str,
                      target_caption: str, top_k: int = 5) -> List[Dict]:
    results = []
    for url in urls:
        print(f"🔍 Evaluating: {url}")
        result = call_gpt_vision(url, concept, target_caption)
        result["url"] = url
        results.append(result)
        time.sleep(0.5)

    # filter + sort by score
    scored = [r for r in results if isinstance(r.get("score"), (int, float))]
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


# -------------------- Example --------------------

if __name__ == "__main__":
    target_caption = "A cinematic photo of a man riding a bicycle beside a lake at sunrise."
    concept = 'bicycle at sunrise'
    urls = [
"https://images.pexels.com/photos/5901979/pexels-photo-5901979.jpeg?cs=srgb&dl=pexels-lysenkow-5901979.jpg&fm=jpg",
        "https://static.vecteezy.com/system/resources/thumbnails/052/393/397/small_2x/male-bike-rider-is-standing-next-to-his-bike-near-the-river-at-sunset-silhouette-of-a-biker-in-helmet-who-is-having-a-rest-on-the-orange-water-background-evening-landscape-and-a-cyclist-video.jpg",
        "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSOECc5AZccZaXoEV2RNpGMqDGxJ11jbWUFnQ&s",
    ]

    top_images = screen_image_urls(urls, concept, target_caption, top_k=2)
    print(json.dumps(top_images, indent=2, ensure_ascii=False))
