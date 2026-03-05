from __future__ import annotations

import logging
import os
import threading
import time

from voxykey.app.config import load_config, save_config
from voxykey.app.logging_setup import setup_logging
from voxykey.core.hotkeys import HotkeyService
from voxykey.core.pipeline import Pipeline

logger = logging.getLogger(__name__)


class Application:
    def __init__(self) -> None:
        logger.debug("Application initialization started")
        self._stop_event = threading.Event()
        self._is_wayland = self._detect_wayland_session()
        self.cfg = load_config()
        logger.debug("Config loaded in application init")
        self.pipeline = Pipeline(self.cfg)
        self.hotkeys = HotkeyService(
            hotkey=self.cfg.hotkey.trigger,
            push_to_talk_key=self.cfg.hotkey.push_to_talk_key,
            on_trigger=self.capture,
            on_ptt_start=self.begin_capture,
            on_ptt_end=self.end_capture,
        )
        self.tray = None
        if self._is_wayland:
            logger.info("Wayland session detected; tray initialization is disabled")
        else:
            try:
                from voxykey.app.tray import TrayApp

                self.tray = TrayApp(
                    on_capture=self.capture,
                    on_reload=self.reload_config,
                    on_quit=self.stop,
                    on_set_target_language=self.set_target_language,
                    get_target_language=lambda: self.cfg.translation.target_lang,
                )
            except:
                logger.warning("failed to initialize tray icon")
        logger.info("Application initialized")

    def capture(self) -> None:
        logger.debug("Starting one-shot capture thread")
        threading.Thread(target=self.pipeline.run_once, daemon=True).start()

    def begin_capture(self) -> None:
        logger.debug("Begin push-to-talk requested")
        self.pipeline.begin_capture()

    def end_capture(self) -> None:
        logger.debug("End push-to-talk requested")
        threading.Thread(target=self.pipeline.end_capture, daemon=True).start()

    def set_target_language(self, target_lang: str) -> None:
        logger.info("Setting target language: %s", target_lang)
        self.cfg.translation.target_lang = target_lang
        save_config(self.cfg)
        self.pipeline.set_target_language(target_lang)

    def reload_config(self) -> None:
        logger.info("Reloading config")
        self.hotkeys.stop()
        self.cfg = load_config()
        self.pipeline = Pipeline(self.cfg)
        self.hotkeys = HotkeyService(
            hotkey=self.cfg.hotkey.trigger,
            push_to_talk_key=self.cfg.hotkey.push_to_talk_key,
            on_trigger=self.capture,
            on_ptt_start=self.begin_capture,
            on_ptt_end=self.end_capture,
        )
        self.hotkeys.start()
        logger.info("Config reload complete")

    def run(self) -> None:
        logger.info("Application run loop starting")
        self.hotkeys.start()
        if self.tray is None:
            started = False
        else:
            started = self.tray.run()
        if started:
            logger.info("Tray loop exited normally")
            return
        logger.warning("Running without tray icon; hotkeys remain active")
        try:
            while not self._stop_event.is_set():
                time.sleep(0.5)
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
            self.stop()

    def stop(self) -> None:
        logger.info("Stopping application")
        self._stop_event.set()
        self.hotkeys.stop()
        logger.info("Application stopped")

    @staticmethod
    def _detect_wayland_session() -> bool:
        session_type = os.environ.get("XDG_SESSION_TYPE", "").lower()
        return session_type == "wayland" or bool(os.environ.get("WAYLAND_DISPLAY"))


def run() -> None:
    setup_logging()
    logger.debug("Entrypoint run() called")
    app = Application()
    app.run()
