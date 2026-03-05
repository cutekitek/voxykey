from __future__ import annotations

import logging
import threading
from typing import Callable

from PIL import Image, ImageDraw
import pystray

logger = logging.getLogger(__name__)


class TrayApp:
    TARGET_LANGUAGES = {
        "en": "English",
        "es": "Spanish",
        "fr": "French",
        "de": "German",
        "ru": "Russian",
    }

    def __init__(
        self,
        on_capture: Callable[[], None],
        on_reload: Callable[[], None],
        on_quit: Callable[[], None],
        on_set_target_language: Callable[[str], None],
        get_target_language: Callable[[], str],
    ) -> None:
        logger.debug("Initializing tray app")
        self.on_capture = on_capture
        self.on_reload = on_reload
        self.on_quit = on_quit
        self.on_set_target_language = on_set_target_language
        self.get_target_language = get_target_language
        self._icon = pystray.Icon(
            "voxykey",
            icon=self._create_icon(),
            title="VoxyKey",
            menu=self._build_menu(),
        )
        logger.debug("Tray app initialized")

    def run(self, on_started: Callable[[], None] | None = None) -> bool:
        logger.info("Starting tray icon on main thread")
        try:
            if on_started is None:
                self._icon.run()
            else:
                self._icon.run(setup=self._build_setup_callback(on_started))
            return True
        except Exception:
            logger.exception("Tray icon failed to start")
            return False

    def stop(self) -> None:
        logger.info("Stopping tray icon")
        self._icon.stop()

    def _capture_now(self, _icon: object, _item: object) -> None:
        logger.info("Tray action: capture now")
        threading.Thread(target=self.on_capture, daemon=True).start()

    def _reload_config(self, _icon: object, _item: object) -> None:
        logger.info("Tray action: reload config")
        self.on_reload()

    def _quit(self, _icon: object, _item: object) -> None:
        logger.info("Tray action: quit")
        self.on_quit()
        self._icon.stop()

    def _build_menu(self) -> pystray.Menu:
        logger.debug("Building tray menu")
        return pystray.Menu(
            pystray.MenuItem("Capture now", self._capture_now),
            pystray.MenuItem(
                "Target language",
                pystray.Menu(
                    *[self._lang_item(code, label) for code, label in self.TARGET_LANGUAGES.items()]
                ),
            ),
            pystray.MenuItem("Reload config", self._reload_config),
            pystray.MenuItem("Quit", self._quit),
        )

    def _lang_item(self, code: str, label: str) -> pystray.MenuItem:
        def _set_lang(_icon: object, _item: object) -> None:
            logger.info("Tray action: set target language=%s", code)
            self.on_set_target_language(code)
            self._icon.update_menu()

        return pystray.MenuItem(
            label, _set_lang, checked=lambda _item: self.get_target_language() == code
        )

    def _build_setup_callback(self, on_started: Callable[[], None]) -> Callable[[object], None]:
        def _setup(_icon: object) -> None:
            try:
                on_started()
            except Exception:
                logger.exception("Startup callback failed; continuing without hotkeys")

        return _setup

    @staticmethod
    def _create_icon() -> Image.Image:
        logger.debug("Creating tray icon image")
        size = 64
        image = Image.new("RGB", (size, size), color=(24, 39, 57))
        draw = ImageDraw.Draw(image)
        draw.ellipse((8, 8, 56, 56), fill=(60, 172, 98))
        draw.rectangle((28, 18, 36, 44), fill=(10, 20, 10))
        return image
