"""
services/vlm_service.py
───────────────────────
Reusable Vision-Language-Model service.

Supports any combination of:
  • PIL Images
  • Raw bytes
  • File paths (images)
  • Plain text (text-only, no image required)

Features:
  • Lazy model loading (first generate() call triggers download)
  • Exponential back-off retry on RuntimeError / CUDA OOM
  • Fully stateless between calls — safe to reuse across pipeline stages
  • Tested with Qwen/Qwen3-VL-8B-Instruct (trust_remote_code=True)

Standalone usage
────────────────
    from services.vlm_service import VLMService
    from pathlib import Path

    vlm = VLMService("Qwen/Qwen3-VL-8B-Instruct")

    # Vision + text
    answer = vlm.generate(
        system_prompt="Answer concisely.",
        user_prompt="What city is shown?",
        images=[Path("photo.jpg")],
    )

    # Text-only (no image)
    answer = vlm.generate(
        system_prompt="You are a helpful assistant.",
        user_prompt="Translate 'hello' to Japanese.",
    )

    vlm.unload()  # free GPU memory when done
"""

from __future__ import annotations

import io
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
from PIL import Image

logger = logging.getLogger(__name__)

_DEFAULT_MAX_ATTEMPTS = 3
_DEFAULT_BACKOFF_BASE = 4.0   # seconds
_DEFAULT_BACKOFF_MAX  = 60.0  # seconds


class VLMService:
    """
    Thin, reusable wrapper around a HuggingFace Vision-Language Model.

    Parameters
    ----------
    model_id : str
        HuggingFace model ID, e.g. "Qwen/Qwen3-VL-8B-Instruct".
    device : str, optional
        Full device string. Accepted formats:
          - ``None``       → auto-detect (uses "cuda:0" if available, else "cpu")
          - ``"cpu"``      → force CPU
          - ``"cuda"``     → first available GPU (equivalent to "cuda:0")
          - ``"cuda:0"``   → explicitly GPU 0
          - ``"cuda:1"``   → explicitly GPU 1
          - ``"cuda:2,3"`` → multi-GPU spanning devices 2 and 3 via device_map="auto"
        When a specific single GPU is given (e.g. "cuda:1"), the model is
        loaded onto that device and ``device_map`` is NOT used so that the
        model stays on the requested GPU only.
    torch_dtype : torch.dtype, optional
        Defaults to float16 on GPU, float32 on CPU.
    max_new_tokens : int
        Maximum tokens the model may generate per call.
    cache_dir : str, optional
        Local directory for cached model weights
        (defaults to ~/.cache/huggingface).
    hf_token : str, optional
        HuggingFace access token for gated models.
    max_attempts : int
        How many times to retry on transient errors.
    backoff_base : float
        Base seconds for exponential back-off.
    backoff_max : float
        Upper cap on back-off delay in seconds.
    """

    def __init__(
            self,
            model_id: str,
            *,
            device: Optional[str] = None,
            torch_dtype: Optional[torch.dtype] = None,
            max_new_tokens: int = 512,
            cache_dir: Optional[str] = None,
            hf_token: Optional[str] = None,
            max_attempts: int = _DEFAULT_MAX_ATTEMPTS,
            backoff_base: float = _DEFAULT_BACKOFF_BASE,
            backoff_max: float = _DEFAULT_BACKOFF_MAX,
    ) -> None:
        self.model_id       = model_id
        self.max_new_tokens = max_new_tokens
        self.cache_dir      = cache_dir
        self.hf_token       = hf_token
        self.max_attempts   = max_attempts
        self.backoff_base   = backoff_base
        self.backoff_max    = backoff_max

        self.device, self._use_device_map = self._resolve_device(device)
        self.torch_dtype: torch.dtype = torch_dtype or (
            torch.float16 if self.device.startswith("cuda") else torch.float32
        )

        self._model     = None
        self._processor = None

    # ── Device resolution ──────────────────────────────────────────────

    @staticmethod
    def _resolve_device(device: Optional[str]) -> tuple[str, bool]:
        """
        Parse the device string and decide whether to use device_map="auto".

        Returns
        -------
        (device_str, use_device_map)

        Rules:
          None / "cuda"       → ("cuda:0", True)   — auto-spread across all GPUs
          "cuda:N"            → ("cuda:N", False)  — pin to one specific GPU
          "cuda:N,M,…"        → ("cuda:N", True)   — multi-GPU, device_map="auto",
                                                      restricted to listed devices
          "cpu"               → ("cpu", False)
        """
        if device is None:
            if torch.cuda.is_available():
                return "cuda:0", True
            return "cpu", False

        device = device.strip()

        if device == "cpu":
            return "cpu", False

        if device == "cuda":
            # bare "cuda" → use GPU 0 as primary, spread via device_map
            return "cuda:0", True

        if device.startswith("cuda:"):
            remainder = device[len("cuda:"):]
            if "," in remainder:
                # Multi-GPU: "cuda:0,1" → primary = cuda:0, device_map=auto
                primary_idx = remainder.split(",")[0].strip()
                return f"cuda:{primary_idx}", True
            # Single specific GPU: "cuda:1" → pin, no device_map
            return device, False

        raise ValueError(
            f"Unrecognised device string: {device!r}. "
            "Use 'cpu', 'cuda', 'cuda:0', 'cuda:1', or 'cuda:0,1' etc."
        )

    # ── Public API ─────────────────────────────────────────────────────

    def generate(
            self,
            system_prompt: str,
            user_prompt: str,
            *,
            images: Optional[List[Union[Image.Image, bytes, Path, str]]] = None,
            texts: Optional[List[str]] = None,
    ) -> str:
        """
        Run inference and return the model's text response.

        Parameters
        ----------
        system_prompt : str
            High-level instruction injected as the system turn.
        user_prompt : str
            The actual query (placed last in user content).
        images : list, optional
            Zero or more images as PIL Image, raw bytes, or file paths.
            Images appear *before* user_prompt in the message.
            Omit or pass [] for text-only inference.
        texts : list of str, optional
            Extra text snippets inserted between images and user_prompt.

        Returns
        -------
        str
            Model response, stripped of special tokens and whitespace.

        Raises
        ------
        RuntimeError
            If all retry attempts are exhausted.
        """
        self._ensure_loaded()

        pil_images, user_content = self._build_user_content(
            images or [], texts or [], user_prompt
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_content},
        ]

        inputs = self._prepare_inputs(messages, pil_images)

        try:
            return self._generate_with_retry(inputs)
        finally:
            del inputs
            self._maybe_clear_cuda_cache()

    def unload(self) -> None:
        """Release model weights from (GPU) memory."""
        self._model     = None
        self._processor = None
        self._maybe_clear_cuda_cache()
        logger.info("VLMService: model unloaded")

    # ── Model loading ──────────────────────────────────────────────────

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return

        logger.info(
            "Loading model '%s'  |  device=%s  device_map=%s  dtype=%s",
            self.model_id,
            self.device,
            "auto" if self._use_device_map else "none",
            self.torch_dtype,
        )
        t0 = time.perf_counter()

        from transformers import AutoProcessor
        self._processor = AutoProcessor.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            cache_dir=self.cache_dir,
            token=self.hf_token,
        )
        self._model = self._load_model()

        elapsed = time.perf_counter() - t0
        logger.info("Model loaded in %.1f s", elapsed)

    def _load_model(self):
        """
        Try Qwen-specific class first; fall back to AutoModelForVision2Seq.

        device_map="auto" is used only when self._use_device_map is True
        (i.e. bare "cuda", multi-GPU "cuda:0,1", or auto-detected).
        For a specific single GPU (e.g. "cuda:1") device_map is skipped
        so the model is pinned to that GPU only.
        """
        common_kwargs: Dict = dict(
            torch_dtype=self.torch_dtype,
            trust_remote_code=True,
            cache_dir=self.cache_dir,
            token=self.hf_token,
        )
        if self._use_device_map:
            common_kwargs["device_map"] = "auto"

        try:
            from transformers import Qwen2_5_VLForConditionalGeneration
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_id, **common_kwargs
            )
            logger.debug("Loaded via Qwen2_5_VLForConditionalGeneration")
        except (ImportError, OSError, ValueError):
            from transformers import AutoModelForVision2Seq
            model = AutoModelForVision2Seq.from_pretrained(
                self.model_id, **common_kwargs
            )
            logger.debug("Loaded via AutoModelForVision2Seq (fallback)")

        # When not using device_map, move model to the exact requested device
        if not self._use_device_map:
            model = model.to(self.device)

        model.eval()
        return model

    # ── Input preparation ──────────────────────────────────────────────

    def _build_user_content(
            self,
            images: List,
            texts: List[str],
            user_prompt: str,
    ) -> Tuple[List[Image.Image], List[Dict]]:
        """Convert raw inputs into a HF-compatible content list."""
        pil_images: List[Image.Image] = []
        content: List[Dict] = []

        for img in images:
            pil = self._to_pil(img)
            pil_images.append(pil)
            content.append({"type": "image", "image": pil})

        for text in texts:
            content.append({"type": "text", "text": text})

        content.append({"type": "text", "text": user_prompt})
        return pil_images, content

    def _prepare_inputs(self, messages: List[Dict], pil_images: List[Image.Image]):
        """Tokenize messages and move tensors to the target device."""
        text_prompt: str = self._processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self._processor(
            text=[text_prompt],
            images=pil_images if pil_images else None,
            padding=True,
            return_tensors="pt",
        )

        # When device_map="auto" the model may span multiple GPUs —
        # always move inputs to the embedding layer's device.
        # For single-GPU or CPU, self.device is already the right target.
        if self._use_device_map:
            target_device = next(self._model.parameters()).device
        else:
            target_device = self.device

        return inputs.to(target_device)

    # ── Generation + retry ─────────────────────────────────────────────

    def _generate_with_retry(self, inputs) -> str:
        last_exc: Optional[Exception] = None

        for attempt in range(1, self.max_attempts + 1):
            try:
                return self._run_generation(inputs)
            except RuntimeError as exc:
                last_exc = exc
                is_oom = "out of memory" in str(exc).lower()

                if is_oom:
                    self._maybe_clear_cuda_cache()
                    logger.warning(
                        "CUDA OOM on attempt %d/%d — cache cleared",
                        attempt, self.max_attempts,
                    )
                else:
                    logger.warning(
                        "RuntimeError on attempt %d/%d: %s",
                        attempt, self.max_attempts, exc,
                    )

                if attempt < self.max_attempts:
                    delay = min(
                        self.backoff_base * (2 ** (attempt - 1)),
                        self.backoff_max,
                        )
                    logger.info("Retrying in %.1f s …", delay)
                    time.sleep(delay)

        logger.error(
            "Generation failed after %d attempts: %s", self.max_attempts, last_exc
        )
        raise RuntimeError(
            f"VLMService: generation failed after {self.max_attempts} attempts"
        ) from last_exc

    @torch.no_grad()
    def _run_generation(self, inputs) -> str:
        output_ids = self._model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
        )
        # Strip prompt tokens from each output sequence
        new_ids = [
            out[len(inp):]
            for inp, out in zip(inputs["input_ids"], output_ids)
        ]
        decoded = self._processor.batch_decode(
            new_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return decoded[0].strip()

    # ── Utilities ──────────────────────────────────────────────────────

    @staticmethod
    def _to_pil(img: Union[Image.Image, bytes, bytearray, Path, str]) -> Image.Image:
        if isinstance(img, Image.Image):
            return img.convert("RGB")
        if isinstance(img, (bytes, bytearray)):
            return Image.open(io.BytesIO(img)).convert("RGB")
        if isinstance(img, (Path, str)):
            return Image.open(img).convert("RGB")
        raise TypeError(f"Unsupported image type: {type(img)}")

    def _maybe_clear_cuda_cache(self) -> None:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def __repr__(self) -> str:
        loaded = self._model is not None
        return (
            f"VLMService(model_id={self.model_id!r}, "
            f"device={self.device!r}, "
            f"device_map={'auto' if self._use_device_map else 'none'}, "
            f"loaded={loaded})"
        )