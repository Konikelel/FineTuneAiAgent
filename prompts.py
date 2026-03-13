"""
pipeline/prompts.py
───────────────────
All VLM prompts in one place.
Import or override from .env without touching pipeline logic.
"""

# ── Stage 1: Quality filter ────────────────────────────────────────────

FILTER_SYSTEM_PROMPT: str = (
    "You are a strict curator selecting high-quality images for training "
    "a travel image generation model. Answer ONLY with YES or NO."
)

FILTER_USER_PROMPT: str = (
    "Is this a HIGH QUALITY travel photo worth including in an image generation training dataset?\n"
    "Answer YES only if ALL of these are true:\n"
    "- The main subject is a SCENE: landscape, architecture, street, nature, interior space, "
    "or a person naturally situated within a travel environment\n"
    "- The image has clear focus, good lighting, and meaningful composition\n"
    "- It looks like a genuine travel photograph\n"
    "FIRST, check if the image is a COMPOSITE: if the image contains 2, 3, 4, or more "
    "separate photos arranged in a grid, quadrant, collage, side-by-side, or stitched layout, "
    "answer NO immediately.\n"
    "Answer NO if ANY of these apply:\n"
    "- Is a composite image: multiple distinct photos tiled, arranged in quadrants, "
    "placed side-by-side, or stitched together in any grid/collage layout\n"
    "- Shows objects/items/products as the main subject (food close-ups, souvenirs, "
    "clothing items, accessories, bags, cosmetics, packaged goods)\n"
    "- Is dominated by text, infographics, banners, or UI elements\n"
    "- Is an advertisement, promotional content, or heavily watermarked\n"
    "- Is a screenshot of a phone, app, map, or website\n"
    "- Is blurry, severely under/overexposed, or very low resolution\n"
    "Answer:"
)

# ── Stage 2: Structured labelling ─────────────────────────────────────

LABEL_SYSTEM_PROMPT: str = (
    "You are an expert travel-image tagger. "
    "Return ONLY valid JSON with no markdown, no code fences, no extra text."
)

LABEL_USER_PROMPT: str = """Analyze this travel/lifestyle image and classify it.
Return ONLY valid JSON with these fields:
{
  "category": "scenic|food|hotel|people|itinerary|template|pricing|lifestyle|shopping|transport",
  "subcategory": "more specific type, e.g. 'beach', 'street_food', 'boutique_hotel'",
  "description": "one sentence describing the image content",
  "landmark": "name of landmark if recognizable, otherwise null",
  "city": "city name if identifiable, otherwise null",
  "mood": "warm|cool|vibrant|serene|adventurous|luxurious|casual|romantic",
  "is_professional": true or false,
  "has_text_overlay": true or false
}
Rules:
- category MUST be one of the listed options
- Return ONLY the JSON object, no markdown, no explanation"""
