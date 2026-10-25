"""Definição das etapas do pipeline e das opções de extração.

Os imports pesados (Playwright, Ollama, etc.) ficam dentro de cada etapa, então
só é carregado o que a execução realmente usa.
"""
from __future__ import annotations

import logging
from collections.abc import Collection
from dataclasses import dataclass
from pathlib import Path

from promotion_tracker.orchestration.pipeline import Stage

logger = logging.getLogger(__name__)

STAGE_NAMES = ("collect", "extract", "normalize", "consolidate")

STAGE_TITLES = {
    "collect": "Coleta dos Stories no Instagram (scraping)",
    "extract": "Extração das ofertas (Bronze -> Silver)",
    "normalize": "Normalização de marcas e produtos (Silver -> Silver reviewed)",
    "consolidate": "Consolidação (Silver reviewed -> Gold)",
}


@dataclass
class ExtractOptions:
    """Fonte única dos padrões de extração (o argparse usa estes valores)."""

    extractor: str = "tiles"  # "tiles" | "legacy"
    backend: str = "ollama"  # "ollama" | "anthropic"
    gate: str = "flyer"  # "flyer" (todo encarte) | "products" (só com produto monitorado) | "none"
    model: str | None = None
    tile_size: int = 640
    overlap: float = 0.3
    target_long_side: int = 1280
    num_ctx: int = 16384
    num_predict: int = 8192
    num_thread: int | None = None
    num_batch: int | None = None
    reprocess: bool = False
    max_images: int | None = None
    debug_tiles: Path | None = None


def _collect_stage(browser: str) -> Stage:
    def run() -> object:
        from promotion_tracker.ingestion.instagram_client import InstagramClient

        client = InstagramClient(browser=browser)
        try:
            client.connect()
            results = client.get_stories()
            logger.info("Stories coletados: %d", len(results))
            return results
        finally:
            client.close()

    return Stage("collect", f"coletando Stories no Instagram ({browser})", run)


def _extract_stage(options: ExtractOptions) -> Stage:
    if options.extractor == "legacy":

        def run_legacy() -> object:
            from promotion_tracker.extraction.llm_extractor import LLMExtractor

            return LLMExtractor(
                use_ocr_images=False,
                skip_existing=not options.reprocess,
                max_images=options.max_images,
            ).process_bronze_folder()

        return Stage("extract", "extraindo ofertas com OCR e Ollama (imagem inteira)", run_legacy)

    def run_tiles() -> object:
        from promotion_tracker.extraction.tiling.tiled_extractor import TiledExtractor
        from promotion_tracker.extraction.tiling.vision_backends import build_backend

        backend_options: dict[str, object] = {}
        if options.backend == "ollama":
            backend_options = {
                "num_ctx": options.num_ctx,
                "num_predict": options.num_predict,
                "num_thread": options.num_thread,
                "num_batch": options.num_batch,
            }

        from promotion_tracker.common.paths import PROJECT_PATHS

        backend = build_backend(options.backend, options.model, **backend_options)

        gate = None
        if options.gate == "flyer":
            from promotion_tracker.extraction.tiling.gate import FlyerDetector

            gate = FlyerDetector.from_project(vlm=backend)
        elif options.gate == "products":
            from promotion_tracker.extraction.tiling.gate import OcrAnalyzerGate

            gate = OcrAnalyzerGate.from_project()

        extractor = TiledExtractor(
            backend=backend,
            bronze_dir=PROJECT_PATHS.bronze_dir,
            silver_dir=PROJECT_PATHS.silver_dir,
            tile_size=options.tile_size,
            overlap=options.overlap,
            target_long_side=options.target_long_side or None,
            skip_existing=not options.reprocess,
            max_images=options.max_images,
            debug_dir=options.debug_tiles,
            gate=gate,
        )
        return extractor.process_bronze_folder()

    return Stage("extract", f"extraindo ofertas por recortes ({options.backend})", run_tiles)


def _normalize_stage() -> Stage:
    def run() -> object:
        from promotion_tracker.normalization.normalizer import Normalizer

        return Normalizer().normalize()

    return Stage("normalize", "normalizando dados de Silver para Silver reviewed", run)


def _consolidate_stage() -> Stage:
    def run() -> object:
        from promotion_tracker.consolidation.consolidator import Consolidator

        return Consolidator().consolidate()

    return Stage("consolidate", "consolidando dados de Silver reviewed para Gold", run)


def build_stages(
    selected: Collection[str],
    options: ExtractOptions,
    browser: str,
) -> list[Stage]:
    """Monta só as etapas pedidas, sempre na ordem canônica do pipeline."""
    factories = {
        "collect": lambda: _collect_stage(browser),
        "extract": lambda: _extract_stage(options),
        "normalize": _normalize_stage,
        "consolidate": _consolidate_stage,
    }
    return [factories[name]() for name in STAGE_NAMES if name in selected]