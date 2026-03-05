from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class Translator:
    def __init__(self, source_lang: str, target_lang: str, models_dir: str) -> None:
        self.source_lang = source_lang.lower()
        self.target_lang = target_lang.lower()
        self.models_dir = Path(models_dir)
        self._repo_exists_cache: dict[str, bool] = {}
        self._pipeline_repos: list[str] = []
        self._models: list[object] = []
        self._tokenizers: list[object] = []
        logger.debug(
            "Translator initialized source=%s target=%s models_dir=%s",
            self.source_lang,
            self.target_lang,
            self.models_dir,
        )

    def prepare_models(self) -> None:
        logger.debug("Preparing translation models")
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self._pipeline_repos = self._decide_repositories(self.source_lang, self.target_lang)
        logger.info("Translation model plan: %s", " -> ".join(self._pipeline_repos) or "identity")

        self._models = []
        self._tokenizers = []
        for repo_id in self._pipeline_repos:
            model_path = self._ensure_model_downloaded(repo_id)
            tokenizer, model = self._load_local_model(model_path)
            self._tokenizers.append(tokenizer)
            self._models.append(model)
        logger.debug("Prepared %d translation model(s)", len(self._models))

    def set_target_language(self, target_lang: str) -> None:
        logger.info("Translator target language change: %s -> %s", self.target_lang, target_lang.lower())
        self.target_lang = target_lang.lower()
        self.prepare_models()

    def translate(self, text: str) -> str:
        logger.debug("Translation requested text_length=%d", len(text))
        if not text:
            logger.debug("Translation skipped for empty text")
            return ""
        if self.source_lang == self.target_lang:
            logger.debug("Translation skipped because source and target are identical")
            return text
        if not self._models:
            self.prepare_models()

        out = text
        for tokenizer, model in zip(self._tokenizers, self._models, strict=True):
            logger.debug("Running translation hop")
            inputs = tokenizer(out, return_tensors="pt", truncation=True)
            output_tokens = model.generate(**inputs, max_length=512)
            out = tokenizer.batch_decode(output_tokens, skip_special_tokens=True)[0]
        logger.info("Translated text length=%d", len(out))
        return out

    def _decide_repositories(self, source_lang: str, target_lang: str) -> list[str]:
        logger.debug("Deciding translation repositories for %s->%s", source_lang, target_lang)
        if source_lang == target_lang:
            return []

        direct = self._repo_for(source_lang, target_lang)
        if self._repo_available(direct):
            return [direct]

        if source_lang != "en" and target_lang != "en":
            to_en = self._repo_for(source_lang, "en")
            from_en = self._repo_for("en", target_lang)
            if self._repo_available(to_en) and self._repo_available(from_en):
                return [to_en, from_en]

        if target_lang == "en":
            to_en = self._repo_for(source_lang, "en")
            if self._repo_available(to_en):
                return [to_en]

        if source_lang == "en":
            from_en = self._repo_for("en", target_lang)
            if self._repo_available(from_en):
                return [from_en]

        raise RuntimeError(
            f"No OPUS-MT model route found for language pair: {source_lang}->{target_lang}"
        )

    @staticmethod
    def _repo_for(source_lang: str, target_lang: str) -> str:
        return f"Helsinki-NLP/opus-mt-{source_lang}-{target_lang}"

    def _repo_available(self, repo_id: str) -> bool:
        local_path = self._model_path(repo_id)
        if (local_path / "config.json").exists():
            logger.debug("Model repo available locally: %s", repo_id)
            return True
        if repo_id in self._repo_exists_cache:
            logger.debug("Model repo availability cache hit: %s=%s", repo_id, self._repo_exists_cache[repo_id])
            return self._repo_exists_cache[repo_id]

        try:
            from huggingface_hub import HfApi
        except ModuleNotFoundError as exc:  # pragma: no cover
            raise RuntimeError("huggingface-hub is required for OPUS-MT downloads") from exc

        try:
            HfApi().model_info(repo_id)
            self._repo_exists_cache[repo_id] = True
            logger.debug("Model repo available on hub: %s", repo_id)
        except Exception:
            self._repo_exists_cache[repo_id] = False
            logger.debug("Model repo unavailable on hub: %s", repo_id)
        return self._repo_exists_cache[repo_id]

    def _ensure_model_downloaded(self, repo_id: str) -> Path:
        model_path = self._model_path(repo_id)
        if (model_path / "config.json").exists():
            logger.debug("Model already downloaded: %s", repo_id)
            return model_path

        logger.info("Downloading model %s into %s", repo_id, model_path)
        try:
            from huggingface_hub import snapshot_download
        except ModuleNotFoundError as exc:  # pragma: no cover
            raise RuntimeError("huggingface-hub is required for OPUS-MT downloads") from exc

        snapshot_download(repo_id=repo_id, local_dir=str(model_path))
        logger.info("Model download complete: %s", repo_id)
        return model_path

    @staticmethod
    def _load_local_model(model_path: Path) -> tuple[object, object]:
        try:
            from transformers import MarianMTModel, MarianTokenizer
        except ModuleNotFoundError as exc:  # pragma: no cover
            raise RuntimeError("transformers is not installed") from exc

        tokenizer = MarianTokenizer.from_pretrained(str(model_path))
        model = MarianMTModel.from_pretrained(str(model_path))
        logger.debug("Loaded local model from %s", model_path)
        return tokenizer, model

    def _model_path(self, repo_id: str) -> Path:
        # Keep one folder per HuggingFace repo under local models directory.
        path = self.models_dir / repo_id.replace("/", "--")
        logger.debug("Resolved model path for %s: %s", repo_id, path)
        return path
