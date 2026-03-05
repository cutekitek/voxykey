from __future__ import annotations

import logging
from pathlib import Path
import shutil
from typing import Any
from urllib.parse import quote, urlencode
from urllib.request import urlopen
import xml.etree.ElementTree as ET
import zipfile

logger = logging.getLogger(__name__)
S3_NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}


class Translator:
    OPUS_MODELS_BASE_URL = "https://object.pouta.csc.fi/OPUS-MT-models"

    def __init__(
        self,
        source_lang: str,
        target_lang: str,
        models_dir: str,
        device: str = "cpu",
        compute_type: str = "int8",
    ) -> None:
        self.source_lang = source_lang.lower()
        self.target_lang = target_lang.lower()
        self.models_dir = Path(models_dir)
        self.device = device
        self.compute_type = compute_type
        self._pair_exists_cache: dict[str, bool] = {}
        self._archive_key_cache: dict[str, str | None] = {}
        self._pipeline_pairs: list[str] = []
        self._translators: list[Any] = []
        self._source_tokenizers: list[Any] = []
        self._target_tokenizers: list[Any] = []
        logger.debug(
            "Translator initialized source=%s target=%s models_dir=%s device=%s compute_type=%s",
            self.source_lang,
            self.target_lang,
            self.models_dir,
            self.device,
            self.compute_type,
        )

    def prepare_models(self) -> None:
        logger.debug("Preparing translation models")
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self._pipeline_pairs = self._decide_pairs(self.source_lang, self.target_lang)
        logger.info("Translation model plan: %s", " -> ".join(self._pipeline_pairs) or "identity")

        self._translators = []
        self._source_tokenizers = []
        self._target_tokenizers = []
        for pair in self._pipeline_pairs:
            model_path, source_spm_path, target_spm_path = self._ensure_model_converted(pair)
            source_tokenizer, target_tokenizer, model = self._load_local_model(
                model_path, source_spm_path, target_spm_path
            )
            self._source_tokenizers.append(source_tokenizer)
            self._target_tokenizers.append(target_tokenizer)
            self._translators.append(model)
        logger.debug("Prepared %d translation model(s)", len(self._translators))

    def set_target_language(self, target_lang: str) -> None:
        logger.info(
            "Translator target language change: %s -> %s", self.target_lang, target_lang.lower()
        )
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
        if not self._translators:
            self.prepare_models()

        out = text
        for source_tokenizer, target_tokenizer, model in zip(
            self._source_tokenizers,
            self._target_tokenizers,
            self._translators,
            strict=True,
        ):
            logger.debug("Running translation hop")
            source_tokens = source_tokenizer.encode(out, out_type=str)
            if not source_tokens:
                logger.debug("Translation hop skipped for empty tokenized input")
                return ""
            result = model.translate_batch(
                [source_tokens],
                beam_size=1,
                max_decoding_length=512,
                disable_unk=True,
            )
            print(result)
            out_tokens = result[0].hypotheses[0]
            out = target_tokenizer.decode(out_tokens).strip()
        logger.info("Translated text length=%d", len(out))
        return out

    def _decide_pairs(self, source_lang: str, target_lang: str) -> list[str]:
        logger.debug("Deciding translation pairs for %s->%s", source_lang, target_lang)
        if source_lang == target_lang:
            return []

        direct = self._pair_for(source_lang, target_lang)
        if self._pair_available(direct):
            return [direct]

        if source_lang != "en" and target_lang != "en":
            to_en = self._pair_for(source_lang, "en")
            from_en = self._pair_for("en", target_lang)
            if self._pair_available(to_en) and self._pair_available(from_en):
                return [to_en, from_en]

        if target_lang == "en":
            to_en = self._pair_for(source_lang, "en")
            if self._pair_available(to_en):
                return [to_en]

        if source_lang == "en":
            from_en = self._pair_for("en", target_lang)
            if self._pair_available(from_en):
                return [from_en]

        raise RuntimeError(
            f"No OPUS-MT model route found for language pair: {source_lang}->{target_lang}"
        )

    @staticmethod
    def _pair_for(source_lang: str, target_lang: str) -> str:
        return f"{source_lang}-{target_lang}"

    def _pair_available(self, pair_id: str) -> bool:
        local_path = self._model_path(pair_id)
        if (local_path / "ct2" / "model.bin").exists():
            logger.debug("Model pair available locally: %s", pair_id)
            return True
        if pair_id in self._pair_exists_cache:
            logger.debug(
                "Model pair availability cache hit: %s=%s",
                pair_id,
                self._pair_exists_cache[pair_id],
            )
            return self._pair_exists_cache[pair_id]

        exists = self._latest_archive_key(pair_id) is not None
        self._pair_exists_cache[pair_id] = exists
        logger.debug("Model pair availability on OPUS bucket: %s=%s", pair_id, exists)
        return exists

    def _latest_archive_key(self, pair_id: str) -> str | None:
        if pair_id in self._archive_key_cache:
            return self._archive_key_cache[pair_id]

        query = urlencode({"prefix": f"{pair_id}/"})
        url = f"{self.OPUS_MODELS_BASE_URL}/?{query}"
        logger.debug("Querying OPUS model listing: %s", url)

        try:
            with urlopen(url, timeout=30) as response:
                payload = response.read()
        except Exception:
            logger.exception("Failed to fetch model listing for pair %s", pair_id)
            self._archive_key_cache[pair_id] = None
            return None

        try:
            xml_root = ET.fromstring(payload)
        except ET.ParseError:
            logger.exception("Failed to parse model listing for pair %s", pair_id)
            self._archive_key_cache[pair_id] = None
            return None

        keys = [
            key_node.text
            for key_node in xml_root.findall("s3:Contents/s3:Key", S3_NS)
            if key_node.text is not None
        ]
        archives = [
            key
            for key in keys
            if key.startswith(f"{pair_id}/opus-")
            and key.endswith(".zip")
            and not key.endswith(".eval.zip")
        ]
        latest = sorted(archives)[-1] if archives else None
        self._archive_key_cache[pair_id] = latest
        logger.debug("Latest OPUS archive for pair %s: %s", pair_id, latest)
        return latest

    def _ensure_model_converted(self, pair_id: str) -> tuple[Path, Path, Path]:
        model_root = self._model_path(pair_id)
        ct2_dir = model_root / "ct2"
        source_spm_path = model_root / "source.spm"
        target_spm_path = model_root / "target.spm"
        if (
            (ct2_dir / "model.bin").exists()
            and source_spm_path.exists()
            and target_spm_path.exists()
        ):
            logger.debug("Model already converted: %s", pair_id)
            return ct2_dir, source_spm_path, target_spm_path

        archive_key = self._latest_archive_key(pair_id)
        if archive_key is None:
            raise RuntimeError(f"No OPUS archive found for language pair: {pair_id}")

        archive_url = f"{self.OPUS_MODELS_BASE_URL}/{quote(archive_key, safe='/')}"
        archive_path = model_root / Path(archive_key).name
        extract_root = model_root / "opus"
        model_root.mkdir(parents=True, exist_ok=True)

        if not archive_path.exists():
            logger.info("Downloading OPUS model archive %s", archive_url)
            self._download_file(archive_url, archive_path)
            logger.info("Model archive downloaded: %s", archive_path)

        if not extract_root.exists() or not any(extract_root.iterdir()):
            logger.info("Extracting OPUS model archive %s", archive_path)
            extract_root.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(archive_path) as archive:
                archive.extractall(extract_root)

        opus_model_dir = self._find_opus_model_dir(extract_root)
        source_spm, target_spm = self._resolve_tokenizers(opus_model_dir)
        if not source_spm_path.exists():
            shutil.copy2(source_spm, source_spm_path)
        if not target_spm_path.exists():
            shutil.copy2(target_spm, target_spm_path)

        if not (ct2_dir / "model.bin").exists():
            logger.info("Converting OPUS model to CTranslate2 format: %s", pair_id)
            try:
                from ctranslate2.converters import OpusMTConverter
            except ModuleNotFoundError as exc:  # pragma: no cover
                raise RuntimeError("ctranslate2 is required for OPUS model conversion") from exc

            converter = OpusMTConverter(str(opus_model_dir))
            converter.convert(str(ct2_dir), quantization="int8")
            logger.info("CTranslate2 conversion complete: %s", ct2_dir)

        return ct2_dir, source_spm_path, target_spm_path

    @staticmethod
    def _download_file(url: str, destination: Path) -> None:
        with urlopen(url, timeout=120) as response:
            with destination.open("wb") as output_file:
                shutil.copyfileobj(response, output_file)

    @staticmethod
    def _find_opus_model_dir(extract_root: Path) -> Path:
        if (extract_root / "decoder.yml").exists():
            return extract_root
        for config in extract_root.rglob("decoder.yml"):
            return config.parent
        raise RuntimeError(f"Could not find OPUS model files under {extract_root}")

    @staticmethod
    def _resolve_tokenizers(model_dir: Path) -> tuple[Path, Path]:
        source = Translator._find_existing_path(
            model_dir,
            ["source.spm", "source.spm.model", "spm.src", "spm.model"],
        )
        target = Translator._find_existing_path(
            model_dir,
            ["target.spm", "target.spm.model", "spm.trg", "spm.model"],
        )

        all_spm = sorted(model_dir.glob("*.spm"))
        if source is None and all_spm:
            source = all_spm[0]
        if target is None:
            if len(all_spm) > 1:
                target = all_spm[1]
            elif source is not None:
                target = source

        if source is None or target is None:
            raise RuntimeError(f"Could not find sentencepiece models in {model_dir}")
        return source, target

    @staticmethod
    def _find_existing_path(root: Path, names: list[str]) -> Path | None:
        for name in names:
            candidate = root / name
            if candidate.exists():
                return candidate
        return None

    def _load_local_model(
        self,
        model_path: Path,
        source_spm_path: Path,
        target_spm_path: Path,
    ) -> tuple[Any, Any, Any]:
        try:
            import ctranslate2
            import sentencepiece as spm
        except ModuleNotFoundError as exc:  # pragma: no cover
            raise RuntimeError(
                "ctranslate2 and sentencepiece are required for translation"
            ) from exc

        source_tokenizer = spm.SentencePieceProcessor()
        source_tokenizer.Load(str(source_spm_path))
        target_tokenizer = spm.SentencePieceProcessor()
        target_tokenizer.Load(str(target_spm_path))
        translator = ctranslate2.Translator(
            str(model_path),
            device=self.device,
            compute_type=self.compute_type,
        )
        logger.debug("Loaded local CTranslate2 model from %s", model_path)
        return source_tokenizer, target_tokenizer, translator

    def _model_path(self, pair_id: str) -> Path:
        path = self.models_dir / f"opus-mt-{pair_id}"
        logger.debug("Resolved model path for %s: %s", pair_id, path)
        return path
