"""Centraliza os caminhos usados pelo pipeline de dados."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    """Resolve diretórios e arquivos do projeto a partir de uma única raiz."""

    root: Path = field(
        default_factory=lambda: Path(__file__).resolve().parents[3]
    )

    @property
    def config_dir(self) -> Path:
        return self.root / "config"

    @property
    def data_dir(self) -> Path:
        return self.root / "data"

    @property
    def bronze_dir(self) -> Path:
        return self.data_dir / "bronze"

    @property
    def silver_dir(self) -> Path:
        return self.data_dir / "silver"

    @property
    def gold_dir(self) -> Path:
        return self.data_dir / "gold"

    @property
    def models_dir(self) -> Path:
        return self.root / "models"

    @property
    def networks_config(self) -> Path:
        return self.config_dir / "networks.yaml"

    @property
    def segments_config(self) -> Path:
        return self.config_dir / "segments.yaml"

    @property
    def edge_profile_dir(self) -> Path:
        return self.root / "edge_ig_profile"


PROJECT_PATHS = ProjectPaths()
