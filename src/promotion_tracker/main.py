import argparse
import logging
import sys
from collections.abc import Callable
from pathlib import Path

# from typing import Protocol

logger = logging.getLogger(__name__)


# class InstagramLike(Protocol):
#     def connect(self) -> object: ...
#
#     def get_stories(self) -> list[dict]: ...
#
#     def close(self) -> None: ...


def run_pipeline(
    # instagram: InstagramLike,
    extract: Callable[[], object],
    normalize: Callable[[], object],
    consolidate: Callable[[], object],
) -> None:
    # try:
    #     logger.info("Etapa 1/4: conectando ao Instagram e coletando Stories")
    #     instagram.connect()
    #     collection_results = instagram.get_stories()
    #     logger.info("Stories coletados: %d", len(collection_results))

    logger.info("Etapa 1/3: extraindo ofertas com OCR e Ollama")
    extract()

    logger.info("Etapa 2/3: normalizando dados de Silver para Silver reviewed")
    normalize()

    logger.info("Etapa 3/3: consolidando dados de Silver reviewed para Gold")
    consolidate()
    logger.info("Execução finalizada.")

    # finally:
    #     instagram.close()


def _ensure_src_on_path() -> None:
    source_root = str(Path(__file__).resolve().parents[1])
    if source_root not in sys.path:
        sys.path.insert(0, source_root)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pipeline de promoções")
    # Sem efeito enquanto a coleta do Instagram estiver comentada
    parser.add_argument(
        "--browser",
        choices=("edge", "brave", "firefox", "mozilla"),
        default="edge",
        help="Navegador usado na coleta (default: edge)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()  # noqa: F841 (args.browser só volta a ser usado com o Instagram)
    _ensure_src_on_path()

    from promotion_tracker.common.logging_config import configure_logging

    configure_logging()

    from promotion_tracker.consolidation.consolidator import Consolidator
    from promotion_tracker.extraction.llm_extractor import LLMExtractor

    # from promotion_tracker.ingestion.instagram_client import InstagramClient
    from promotion_tracker.normalization.normalizer import Normalizer

    # instagram = InstagramClient(browser=args.browser)

    try:
        run_pipeline(
            # instagram,
            lambda: LLMExtractor(use_ocr_images=False).process_bronze_folder(),
            lambda: Normalizer().normalize(),
            lambda: Consolidator().consolidate(),
        )
    except Exception:
        logger.exception("A execução foi interrompida por um erro.")
        raise


if __name__ == "__main__":
    main()