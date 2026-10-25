"""Executa a extração por tiles.

Exemplos (a partir da raiz do projeto):

  python -m poetry run python src/promotion_tracker/extraction/tiling/cli.py ^
      --silver-dir data/silver_tiles --max-images 5 --debug-dir debug_tiles

  python -m poetry run python src/promotion_tracker/extraction/tiling/cli.py ^
      --backend anthropic --silver-dir data/silver_tiles_api --max-images 5
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[3]
ROOT_DIR = SRC_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from promotion_tracker.extraction.tiling.tiled_extractor import TiledExtractor  # noqa: E402
from promotion_tracker.extraction.tiling.vision_backends import build_backend  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Extração de encartes por tiles")
    parser.add_argument("--backend", choices=["ollama", "anthropic"], default="ollama")
    parser.add_argument("--model", default=None)
    parser.add_argument("--bronze-dir", type=Path, default=ROOT_DIR / "data" / "bronze")
    parser.add_argument("--silver-dir", type=Path, default=ROOT_DIR / "data" / "silver_tiles")
    parser.add_argument("--tile-size", type=int, default=1024)
    parser.add_argument("--overlap", type=float, default=0.25)
    parser.add_argument("--target-long-side", type=int, default=1280)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--no-skip", action="store_true", help="reprocessa JSONs existentes")
    parser.add_argument("--debug-dir", type=Path, default=None,
                        help="salva cada tile e a resposta crua do modelo")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    extractor = TiledExtractor(
        backend=build_backend(args.backend, args.model),
        bronze_dir=args.bronze_dir,
        silver_dir=args.silver_dir,
        tile_size=args.tile_size,
        overlap=args.overlap,
        target_long_side=args.target_long_side or None,
        skip_existing=not args.no_skip,
        max_images=args.max_images,
        debug_dir=args.debug_dir,
    )
    extractor.process_bronze_folder()


if __name__ == "__main__":
    main()
