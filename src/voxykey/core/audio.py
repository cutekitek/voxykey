from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class AudioChunk:
    samples: np.ndarray
    sample_rate: int


class AudioRecorder:
    def __init__(self, sample_rate: int, channels: int) -> None:
        self.sample_rate = sample_rate
        self.channels = channels
        self._stream = None
        self._chunks: list[np.ndarray] = []
        logger.debug("AudioRecorder initialized sample_rate=%d channels=%d", sample_rate, channels)

    def record(self, seconds: float) -> AudioChunk:
        try:
            import sounddevice as sd
        except ModuleNotFoundError as exc:  # pragma: no cover
            raise RuntimeError("sounddevice is not installed") from exc

        logger.info("Recording for %.1f seconds", seconds)
        frames = int(seconds * self.sample_rate)
        logger.debug("Fixed recording frames=%d", frames)
        audio = sd.rec(frames, samplerate=self.sample_rate, channels=self.channels, dtype="float32")
        sd.wait()
        mono = audio[:, 0] if audio.ndim > 1 else audio
        logger.debug("Fixed recording complete samples=%d", len(mono))
        return AudioChunk(samples=np.asarray(mono), sample_rate=self.sample_rate)

    def start(self) -> None:
        try:
            import sounddevice as sd
        except ModuleNotFoundError as exc:  # pragma: no cover
            raise RuntimeError("sounddevice is not installed") from exc

        if self._stream is not None:
            logger.debug("Audio stream already running; start ignored")
            return
        self._chunks = []

        def _callback(indata: np.ndarray, _frames: int, _time: object, status: object) -> None:
            if status:
                logger.debug("Audio status: %s", status)
            self._chunks.append(indata.copy())

        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype="float32",
            callback=_callback,
        )
        self._stream.start()
        logger.info("Audio stream started")

    def stop(self) -> AudioChunk:
        if self._stream is None:
            logger.debug("Audio stop called with no active stream")
            return AudioChunk(samples=np.asarray([], dtype=np.float32), sample_rate=self.sample_rate)

        self._stream.stop()
        self._stream.close()
        self._stream = None

        if not self._chunks:
            logger.debug("Audio stream stopped with no chunks captured")
            return AudioChunk(samples=np.asarray([], dtype=np.float32), sample_rate=self.sample_rate)

        audio = np.concatenate(self._chunks, axis=0)
        mono = audio[:, 0] if audio.ndim > 1 else audio
        logger.info("Audio stream stopped, captured %.2fs", len(mono) / self.sample_rate)
        return AudioChunk(samples=np.asarray(mono), sample_rate=self.sample_rate)


def detect_voice_activity(
    samples: np.ndarray,
    sample_rate: int,
    frame_ms: int,
    rms_threshold: float,
    min_voiced_ratio: float,
) -> tuple[bool, float]:
    if sample_rate <= 0 or len(samples) == 0:
        return False, 0.0

    frame_samples = max(1, int(sample_rate * frame_ms / 1000))
    usable = (len(samples) // frame_samples) * frame_samples
    if usable == 0:
        rms = float(np.sqrt(np.mean(np.square(samples, dtype=np.float32))))
        voiced_ratio = 1.0 if rms >= rms_threshold else 0.0
        return voiced_ratio >= min_voiced_ratio, voiced_ratio

    framed = samples[:usable].reshape(-1, frame_samples)
    rms_per_frame = np.sqrt(np.mean(np.square(framed, dtype=np.float32), axis=1))
    voiced_ratio = float(np.mean(rms_per_frame >= rms_threshold))
    return voiced_ratio >= min_voiced_ratio, voiced_ratio
