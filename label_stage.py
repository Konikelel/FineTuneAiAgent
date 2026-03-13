"""
pipeline/label_stage.py
───────────────────────
Stage 2 — Structured labelling.

Input  : image (Path / PIL / bytes) + optional pre-loaded bytes from Parquet
Output : dict with all LABEL_OUTPUT_SCHEMA label fields

Handles:
  • Markdown fences around JSON (```json … ```)
  • Leading / trailing whitespace
  • Partial JSON (regex fallback extraction)
  • Completely invalid JSON (returns error record)
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, Union

from PIL import Image

from pipeline.prompts import LABEL_SYSTEM_PROMPT, LABEL_USER_PROMPT
from services.vlm_service import VLMService

logger = logging.getLogger(__name__)

_REQUIRED_FIELDS = {
    "category", "subcategory", "description",
    "landmark", "city", "mood", "is_professional", "has_text_overlay",
}
_VALID_CATEGORIES = {
    "scenic", "food", "hotel", "people", "itinerary",
    "template", "pricing", "lifestyle", "shopping", "transport",
}
_VALID_MOODS = {
    "warm", "cool", "vibrant", "serene",
    "adventurous", "luxurious", "casual", "romantic",
}


class LabelStage:
    """
    Wraps the VLM labelling call and JSON parsing.

    Parameters
    ----------
    vlm : VLMService
    system_prompt / user_prompt : str, optional
    """

    def __init__(
        self,
        vlm: VLMService,
        *,
        system_prompt: str = LABEL_SYSTEM_PROMPT,
        user_prompt: str   = LABEL_USER_PROMPT,
    ) -> None:
        self.vlm           = vlm
        self.system_prompt = system_prompt
        self.user_prompt   = user_prompt

    def run(self, image: Union[Path, str, bytes, Image.Image]) -> Dict[str, Any]:
        """
        Label *image* and return a flat dict of parsed label fields.

        Every key from ``_REQUIRED_FIELDS`` is guaranteed to be present
        (missing ones default to None). An ``error`` key is populated
        if parsing failed; ``label_json`` holds the raw VLM response.
        """
        name = Path(image).name if isinstance(image, (str, Path)) else "<bytes>"

        try:
            raw: str = self.vlm.generate(
                system_prompt=self.system_prompt,
                user_prompt=self.user_prompt,
                images=[image],
            )
        except Exception as exc:
            logger.error("LabelStage VLM call failed for %s: %s", name, exc)
            return self._error_record(str(exc))

        return self._parse(raw, name)

    # ── Private ────────────────────────────────────────────────────────

    def _parse(self, raw: str, name: str) -> Dict[str, Any]:
        cleaned = self._strip_fences(raw)

        try:
            data: Dict[str, Any] = json.loads(cleaned)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group())
                    logger.warning("label: partial JSON extracted for %s", name)
                except json.JSONDecodeError:
                    logger.error("label: unparseable JSON for %s — raw: %.200s", name, raw)
                    return self._error_record(f"JSONDecodeError: {raw[:200]}")
            else:
                logger.error("label: no JSON object found for %s — raw: %.200s", name, raw)
                return self._error_record(f"No JSON object found: {raw[:200]}")

        return self._normalise(data, raw)

    @staticmethod
    def _strip_fences(text: str) -> str:
        text = text.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
        return text.strip()

    @staticmethod
    def _normalise(data: Dict[str, Any], raw: str) -> Dict[str, Any]:
        category = data.get("category", "")
        if category not in _VALID_CATEGORIES:
            logger.warning("label: unknown category '%s', keeping as-is", category)

        mood = data.get("mood", "")
        if mood not in _VALID_MOODS:
            logger.warning("label: unknown mood '%s', keeping as-is", mood)

        for bool_field in ("is_professional", "has_text_overlay"):
            val = data.get(bool_field)
            if isinstance(val, str):
                data[bool_field] = val.lower() in ("true", "yes", "1")

        return {
            "label_json":       raw,
            "category":         data.get("category"),
            "subcategory":      data.get("subcategory"),
            "description":      data.get("description"),
            "landmark":         data.get("landmark"),
            "city":             data.get("city"),
            "mood":             data.get("mood"),
            "is_professional":  data.get("is_professional"),
            "has_text_overlay": data.get("has_text_overlay"),
            "error":            None,
        }

    @staticmethod
    def _error_record(msg: str) -> Dict[str, Any]:
        return {
            "label_json":       None,
            "category":         None,
            "subcategory":      None,
            "description":      None,
            "landmark":         None,
            "city":             None,
            "mood":             None,
            "is_professional":  None,
            "has_text_overlay": None,
            "error":            msg,
        }
