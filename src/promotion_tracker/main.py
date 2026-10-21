import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def _ensure_src_on_path() -> None:
    source_root = str(Path(__file__).resolve().parents[1])
    if source_root not in sys.path:
        sys.path.insert(0, source_root)


def main():
    _ensure_src_on_path()

    from promotion_tracker.common.logging_config import configure_logging

    configure_logging()

    from promotion_tracker.consolidation.consolidator import Consolidator
    from promotion_tracker.ingestion.instagram_client import InstagramClient
    from promotion_tracker.normalization.normalizer import Normalizer

    instagram = InstagramClient()

    try:
        # ---------------------------------------------------------
        # ETAPA 1: Coleta de Dados Brutos (Playwright)
        # ---------------------------------------------------------
        logger.info("Etapa 1/5: coleta de Stories não executada nesta versão.")

        # ---------------------------------------------------------
        # ETAPA 2: Melhoria de Imagem (ESPCN 4x + Lanczos 2x)
        # ---------------------------------------------------------
        logger.info("Etapa 2/5: melhoria de imagens não executada nesta versão.")

        # ---------------------------------------------------------
        # ETAPA 3: Extração via IA de Visão (Ollama)
        # ---------------------------------------------------------
        logger.info("Etapa 3/5: extração via IA não executada nesta versão.")

        # ---------------------------------------------------------
        # ETAPA 4: Normalização dos dados (Silver -> Silver reviewed)
        # ---------------------------------------------------------
        logger.info("Etapa 4/5: normalizando dados de Silver para Silver reviewed")
        Normalizer().normalize()

        # ---------------------------------------------------------
        # ETAPA 5: Consolidação de Dados (Silver reviewed -> Gold)
        # ---------------------------------------------------------
        logger.info(
            "Etapa 5/5: consolidando dados de Silver reviewed para Gold"
        )
        consolidator = Consolidator()
        consolidator.consolidate()
        logger.info("Execução finalizada.")
    except Exception:
        logger.exception("A execução foi interrompida por um erro.")
        raise
    finally:
        instagram.close()

if __name__ == "__main__":
    main()