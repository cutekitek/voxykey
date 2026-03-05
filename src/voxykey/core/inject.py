from __future__ import annotations

import logging
import platform

logger = logging.getLogger(__name__)


class TextInjector:
    def __init__(self) -> None:
        self.system = platform.system().lower()
        logger.debug("TextInjector initialized system=%s", self.system)

    def inject_text(self, text: str) -> None:
        if not text:
            logger.debug("Inject skipped for empty text")
            return
        try:
            import pyperclip
            from pynput.keyboard import Controller, Key
        except ModuleNotFoundError as exc:  # pragma: no cover
            raise RuntimeError("pyperclip and pynput are required for text injection") from exc

        keyboard = Controller()
        previous = pyperclip.paste()
        logger.debug("Clipboard captured previous_length=%d", len(previous))

        try:
            pyperclip.copy(text)
            mod = Key.cmd if self.system == "darwin" else Key.ctrl
            logger.debug("Injecting text length=%d using modifier=%s", len(text), mod)
            with keyboard.pressed(mod):
                keyboard.press("v")
                keyboard.release("v")
            logger.info("Injected text using clipboard paste")
        finally:
            pyperclip.copy(previous)
            logger.debug("Clipboard restored")
