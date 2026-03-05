from __future__ import annotations

from dataclasses import asdict, dataclass, field
import logging
from pathlib import Path
from typing import Any

from platformdirs import user_config_dir

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    tomllib = None  # type: ignore[assignment]

import tomli_w

APP_NAME = "voxykey"
logger = logging.getLogger(__name__)


@dataclass(slots=True)
class AudioConfig:
    sample_rate: int = 16_000
    channels: int = 1
    record_seconds: float = 6.0
    min_seconds: float = 0.2
    vad_frame_ms: int = 30
    vad_rms_threshold: float = 0.012
    vad_min_voiced_ratio: float = 0.15


@dataclass(slots=True)
class STTConfig:
    model_size: str = "small"
    device: str = "cpu"
    compute_type: str = "int8"
    language: str = "en"


@dataclass(slots=True)
class TranslationConfig:
    source_lang: str = "ru"
    target_lang: str = "en"
    models_dir: str = "models"


@dataclass(slots=True)
class HotkeyConfig:
    trigger: str = "<ctrl>+<shift>+<space>"
    push_to_talk_key: str = "f8"


@dataclass(slots=True)
class OutputConfig:
    paste_with_clipboard: bool = True


@dataclass(slots=True)
class AppConfig:
    enabled: bool = True
    audio: AudioConfig = field(default_factory=AudioConfig)
    stt: STTConfig = field(default_factory=STTConfig)
    translation: TranslationConfig = field(default_factory=TranslationConfig)
    hotkey: HotkeyConfig = field(default_factory=HotkeyConfig)
    output: OutputConfig = field(default_factory=OutputConfig)


def config_path() -> Path:
    path = Path(user_config_dir(APP_NAME, roaming=True)) / "config.toml"
    logger.debug("Resolved config path: %s", path)
    return path


def _merge(default: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    logger.debug("Merging config dictionaries")
    out = dict(default)
    for key, value in incoming.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config() -> AppConfig:
    path = config_path()
    logger.info("Loading config from %s", path)
    if not path.exists():
        logger.info("Config file missing, writing defaults")
        cfg = AppConfig()
        save_config(cfg)
        return cfg

    raw: dict[str, Any] = {}
    if tomllib is not None:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
        logger.debug("Config loaded with top-level keys: %s", sorted(raw.keys()))

    merged = _merge(asdict(AppConfig()), raw)
    cfg = AppConfig(
        enabled=bool(merged["enabled"]),
        audio=AudioConfig(**merged["audio"]),
        stt=STTConfig(**merged["stt"]),
        translation=TranslationConfig(**merged["translation"]),
        hotkey=HotkeyConfig(**merged["hotkey"]),
        output=OutputConfig(**merged["output"]),
    )
    normalized = _normalize_hotkey(cfg.hotkey.trigger)
    if normalized != cfg.hotkey.trigger:
        logger.info("Normalized hotkey trigger from '%s' to '%s'", cfg.hotkey.trigger, normalized)
        cfg.hotkey.trigger = normalized
        save_config(cfg)
    logger.debug("Config loaded: %s", cfg)
    return cfg


def save_config(cfg: AppConfig) -> None:
    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Saving config to %s", path)
    path.write_text(tomli_w.dumps(asdict(cfg)), encoding="utf-8")
    logger.debug("Config saved")


def _normalize_hotkey(hotkey: str) -> str:
    # Pynput expects special keys wrapped in angle brackets.
    token_map = {
        "space": "<space>",
        "enter": "<enter>",
        "tab": "<tab>",
        "esc": "<esc>",
        "escape": "<esc>",
    }
    parts = [part.strip() for part in hotkey.split("+")]
    out: list[str] = []
    for part in parts:
        lower = part.lower()
        if lower in token_map:
            out.append(token_map[lower])
        else:
            out.append(part)
    normalized = "+".join(out)
    logger.debug("Hotkey normalization '%s' -> '%s'", hotkey, normalized)
    return normalized
