from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


class SpeechToText:
    def __init__(self, model_size: str, device: str, compute_type: str, language: str) -> None:
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self._model = None
        logger.debug(
            "SpeechToText initialized model=%s device=%s compute_type=%s language=%s",
            model_size,
            device,
            compute_type,
            language,
        )

    def _ensure_model(self) -> None:
        if self._model is not None:
            logger.debug("Faster-Whisper model already loaded")
            return
        try:
            from faster_whisper import WhisperModel
        except ModuleNotFoundError as exc:  # pragma: no cover
            raise RuntimeError("faster-whisper is not installed") from exc

        logger.info(
            "Loading Faster-Whisper model=%s device=%s compute_type=%s",
            self.model_size,
            self.device,
            self.compute_type,
        )
        self._model = WhisperModel(self.model_size, device=self.device, compute_type=self.compute_type)

    def transcribe(self, samples: np.ndarray, sample_rate: int) -> str:
        logger.debug("Transcription requested sample_rate=%d samples=%d", sample_rate, len(samples))
        self._ensure_model()
        assert self._model is not None
        segments, _ = self._model.transcribe(samples, language=self.language)
        text = " ".join(segment.text.strip() for segment in segments).strip()
        print(text)
        logger.info("STT text length=%d", len(text))
        return text
