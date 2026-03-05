from __future__ import annotations

import logging
import threading

import numpy as np

from voxykey.app.config import AppConfig
from voxykey.core.audio import AudioRecorder, detect_voice_activity
from voxykey.core.inject import TextInjector
from voxykey.core.stt import SpeechToText
from voxykey.core.translate import Translator

logger = logging.getLogger(__name__)


class Pipeline:
    def __init__(self, cfg: AppConfig) -> None:
        self.cfg = cfg
        self._lock = threading.Lock()
        logger.debug(
            "Initializing pipeline stt_language=%s translation=%s->%s",
            cfg.stt.language,
            cfg.translation.source_lang,
            cfg.translation.target_lang,
        )
        self.audio = AudioRecorder(sample_rate=cfg.audio.sample_rate, channels=cfg.audio.channels)
        self.stt = SpeechToText(
            model_size=cfg.stt.model_size,
            device=cfg.stt.device,
            compute_type=cfg.stt.compute_type,
            language=cfg.stt.language,
        )
        self.translator = Translator(
            source_lang=cfg.translation.source_lang,
            target_lang=cfg.translation.target_lang,
            models_dir=cfg.translation.models_dir,
            device=cfg.translation.device,
            compute_type=cfg.translation.compute_type,
        )
        self.translator.prepare_models()
        self.injector = TextInjector()
        self._recording = False
        logger.info("Pipeline initialized")

    def run_once(self) -> None:
        logger.debug("Pipeline run_once requested")
        if not self._lock.acquire(blocking=False):
            logger.info("Pipeline run already in progress")
            return

        try:
            chunk = self.audio.record(self.cfg.audio.record_seconds)
            self._process_chunk(chunk.samples, chunk.sample_rate)
            logger.info("Pipeline completed")
        except Exception:
            logger.exception("Pipeline failed")
        finally:
            self._lock.release()

    def begin_capture(self) -> None:
        logger.debug("Pipeline begin_capture requested")
        if not self._lock.acquire(blocking=False):
            logger.info("Pipeline busy, ignoring capture start")
            return
        try:
            if self._recording:
                logger.debug("Pipeline already recording")
                return
            self.audio.start()
            self._recording = True
            logger.info("Push-to-talk capture started")
        except Exception:
            self._lock.release()
            logger.exception("Failed to start capture")

    def end_capture(self) -> None:
        logger.debug("Pipeline end_capture requested")
        if not self._recording:
            logger.debug("No active recording to stop")
            return
        try:
            chunk = self.audio.stop()
            seconds = len(chunk.samples) / max(chunk.sample_rate, 1)
            if seconds < self.cfg.audio.min_seconds:
                logger.info("Capture ignored: too short (%.2fs)", seconds)
                return
            self._process_chunk(chunk.samples, chunk.sample_rate)
            logger.info("Push-to-talk pipeline completed")
        except Exception:
            logger.exception("Failed to finish capture")
        finally:
            self._recording = False
            self._lock.release()

    def set_target_language(self, target_lang: str) -> None:
        logger.info("Pipeline target language update: %s", target_lang)
        self.cfg.translation.target_lang = target_lang
        self.translator.set_target_language(target_lang)

    def _process_chunk(self, samples: np.ndarray, sample_rate: int) -> None:
        logger.debug("Processing audio chunk samples=%d sample_rate=%d", len(samples), sample_rate)
        has_voice, voiced_ratio = detect_voice_activity(
            samples=samples,
            sample_rate=sample_rate,
            frame_ms=self.cfg.audio.vad_frame_ms,
            rms_threshold=self.cfg.audio.vad_rms_threshold,
            min_voiced_ratio=self.cfg.audio.vad_min_voiced_ratio,
        )
        if not has_voice:
            logger.info(
                "Capture ignored: no voice detected (voiced_ratio=%.3f threshold=%.3f)",
                voiced_ratio,
                self.cfg.audio.vad_min_voiced_ratio,
            )
            return
        text = self.stt.transcribe(samples, sample_rate=sample_rate)
        if not text.strip():
            logger.info("Capture ignored: empty transcription")
            return
        logger.debug("STT output text: %s", text)
        translated = self.translator.translate(text)
        logger.debug("Translated output text=%s", translated)
        self.injector.inject_text(translated)
