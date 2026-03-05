from __future__ import annotations

import logging
import os
import platform
from typing import Callable

logger = logging.getLogger(__name__)


class HotkeyService:
    def __init__(
        self,
        hotkey: str,
        push_to_talk_key: str,
        on_trigger: Callable[[], None],
        on_ptt_start: Callable[[], None],
        on_ptt_end: Callable[[], None],
    ) -> None:
        self.hotkey = hotkey
        self.push_to_talk_key = push_to_talk_key.lower()
        self.on_trigger = on_trigger
        self.on_ptt_start = on_ptt_start
        self.on_ptt_end = on_ptt_end
        self._listener = None
        self._global_hotkeys = None
        self._ptt_pressed = False
        logger.debug(
            "HotkeyService initialized trigger=%s ptt=%s",
            self.hotkey,
            self.push_to_talk_key,
        )

    def start(self) -> None:
        self._configure_backend_for_session()
        try:
            from pynput import keyboard
            from pynput.keyboard import GlobalHotKeys
        except ModuleNotFoundError as exc:  # pragma: no cover
            raise RuntimeError("pynput is not installed") from exc
        except ImportError as exc:
            raise RuntimeError(
                "Keyboard backend is unavailable. On Linux Wayland, set "
                "PYNPUT_BACKEND_KEYBOARD=uinput and ensure access to /dev/input."
            ) from exc

        hotkey = self._normalize_hotkey(self.hotkey)
        try:
            self._global_hotkeys = GlobalHotKeys({hotkey: self._trigger})
        except ValueError:
            logger.exception("Invalid one-shot hotkey '%s'; disabling it", self.hotkey)
            self._global_hotkeys = None

        try:
            if self._global_hotkeys is not None:
                self._global_hotkeys.start()
            self._listener = keyboard.Listener(on_press=self._on_press, on_release=self._on_release)
            self._listener.start()
        except Exception as exc:
            self.stop()
            raise RuntimeError(
                "Failed to start keyboard listener. On Linux Wayland, "
                "PYNPUT_BACKEND_KEYBOARD=uinput may be required."
            ) from exc
        logger.info("Hotkey listener started trigger=%s ptt=%s", hotkey, self.push_to_talk_key)

    def stop(self) -> None:
        logger.debug("Stopping hotkey service")
        stopped = False
        if self._listener is not None:
            self._listener.stop()
            self._listener = None
            stopped = True
        if self._global_hotkeys is not None:
            self._global_hotkeys.stop()
            self._global_hotkeys = None
            stopped = True
        self._ptt_pressed = False
        if stopped:
            logger.info("Hotkey listener stopped")

    def _trigger(self) -> None:
        logger.info("Hotkey triggered")
        self.on_trigger()

    def _on_press(self, key: object) -> None:
        key_name = self._key_name(key)
        logger.debug("Key press detected: %s", key_name)
        if key_name == self.push_to_talk_key and not self._ptt_pressed:
            self._ptt_pressed = True
            logger.info("Push-to-talk start")
            self.on_ptt_start()

    def _on_release(self, key: object) -> None:
        key_name = self._key_name(key)
        logger.debug("Key release detected: %s", key_name)
        if key_name == self.push_to_talk_key and self._ptt_pressed:
            self._ptt_pressed = False
            logger.info("Push-to-talk end")
            self.on_ptt_end()

    @staticmethod
    def _key_name(key: object) -> str:
        try:
            char = getattr(key, "char", None)
            if char:
                return str(char).lower()
            name = getattr(key, "name", None)
            if name:
                return str(name).lower()
            text = str(key).lower()
            if text.startswith("key."):
                return text.split(".", 1)[1]
            return text
        except Exception:
            return ""

    @staticmethod
    def _normalize_hotkey(hotkey: str) -> str:
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
            out.append(token_map.get(lower, part))
        normalized = "+".join(out)
        logger.debug("Normalized hotkey '%s' -> '%s'", hotkey, normalized)
        return normalized

    @staticmethod
    def _configure_backend_for_session() -> None:
        if platform.system().lower() != "linux":
            return
        os.environ["PYNPUT_BACKEND_KEYBOARD"] = "uinput"
        os.environ["PYNPUT_BACKEND_MOUSE"] = "dummy"
        logger.info(
            "Wayland session detected; defaulting keyboard backend to uinput "
            "(override with PYNPUT_BACKEND_KEYBOARD)."
        )
