"""Orquestração genérica do pipeline: executa etapas em ordem, com log e tempo."""
from __future__ import annotations

import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Stage:
    name: str  # identificador curto (ex.: "extract")
    label: str  # texto exibido no log
    run: Callable[[], object]


class Pipeline:
    def __init__(self, stages: Sequence[Stage]) -> None:
        if not stages:
            raise ValueError("O pipeline precisa de pelo menos uma etapa.")
        self.stages = list(stages)

    def run(self) -> dict[str, float]:
        """Executa as etapas em ordem e devolve {etapa: segundos}."""
        timings: dict[str, float] = {}
        total = len(self.stages)

        for position, stage in enumerate(self.stages, start=1):
            logger.info("Etapa %d/%d: %s", position, total, stage.label)
            started = time.perf_counter()
            try:
                stage.run()
            except Exception:
                logger.exception("Etapa '%s' falhou.", stage.name)
                raise
            timings[stage.name] = time.perf_counter() - started
            logger.info("Etapa '%s' concluída em %.1fs", stage.name, timings[stage.name])

        logger.info("Execução finalizada em %.1fs", sum(timings.values()))
        return timings