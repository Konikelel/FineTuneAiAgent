"""
pipeline/filter_stage.py
────────────────────────
Stage 1 — Binary quality filter.

Input  : Raw image (Path / PIL / bytes)
Output : Boolean (True = keep / YES, False = discard / NO)

This module handles VLM inference ONLY. Reading and writing to Parquet
is strictly handled by the pipeline orchestrator loop in main.py.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Union

from PIL import Image

from pipeline.prompts import FILTER_SYSTEM_PROMPT, FILTER_USER_PROMPT
from services.vlm_service import VLMService

logger = logging.getLogger(__name__)


class FilterStage:
    """
    Wraps the VLM quality-filter call.

    Parameters
    ----------
    vlm : VLMService
        A fully configured VLMService instance.
    system_prompt / user_prompt : str, optional
        Override defaults from pipeline/prompts.py.
    """

    def __init__(
        self,
        vlm: VLMService,
        *,
        system_prompt: str = FILTER_SYSTEM_PROMPT,
        user_prompt: str = FILTER_USER_PROMPT,
    ) -> None:
        self.vlm = vlm
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt

    def run(self, image: Union[Path, str, bytes, Image.Image]) -> bool:
        """
        Ask the Vision-Language Model whether the image meets quality standards.

        Parameters
        ----------
        image : Path, str, bytes, or PIL.Image.Image
            The image to evaluate.

        Returns
        -------
        bool
            True if the model responds starting with "YES" (keep the image).
            False if the model responds starting with "NO" (discard the image).

        Raises
        ------
        Exception
            If the VLM Service fails to generate a response after all retry attempts.
        """
        try:
            answer: str = self.vlm.generate(
                system_prompt=self.system_prompt,
                user_prompt=self.user_prompt,
                images=[image],
            )
            keep = answer.strip().upper().startswith("YES")
            label = Path(image).name if isinstance(image, (str, Path)) else "<bytes>"
            logger.debug("filter: %s → %s", label, "KEEP" if keep else "SKIP")
            return keep

        except Exception as exc:
            logger.error("FilterStage error: %s", exc)
            raise
