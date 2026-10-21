"""Conexão e criação do schema do banco SQLite das promoções consolidadas."""

import logging
import sqlite3
from pathlib import Path

from ..common.paths import PROJECT_PATHS


logger = logging.getLogger(__name__)


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS encartes (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    perfil              TEXT    NOT NULL,
    arquivo_silver      TEXT    NOT NULL UNIQUE,
    arquivo_origem      TEXT,
    data_processamento  TEXT    NOT NULL,
    validade            TEXT,
    modelo              TEXT,
    gerado_em           TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS ofertas (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    encarte_id            INTEGER NOT NULL REFERENCES encartes(id) ON DELETE CASCADE,
    produto               TEXT    NOT NULL,
    produto_original      TEXT,
    marca                 TEXT,
    marca_original        TEXT,
    quantidade            REAL,
    unidade_medida        TEXT,
    preco_atual           REAL,
    preco_anterior        REAL,
    preco_condicao        REAL,
    condicao              TEXT,
    oferta_compartilhada  INTEGER NOT NULL DEFAULT 0,
    variantes_na_oferta   INTEGER
);

CREATE INDEX IF NOT EXISTS idx_ofertas_produto   ON ofertas(produto);
CREATE INDEX IF NOT EXISTS idx_ofertas_preco     ON ofertas(preco_atual);
CREATE INDEX IF NOT EXISTS idx_encartes_perfil   ON encartes(perfil);
CREATE INDEX IF NOT EXISTS idx_encartes_data     ON encartes(data_processamento);
"""


def connect_database(
    database_path: Path | None = None,
) -> sqlite3.Connection:
    """Abre o banco de promoções e retorna uma conexão SQLite configurada.

    Por padrão, usa ``data/gold/promotions.db``. Cria o diretório Gold caso
    ainda não exista. A conexão fica sob responsabilidade de quem a chamou e
    deve ser encerrada com ``connection.close()``.
    """
    path = Path(database_path) if database_path else PROJECT_PATHS.promotions_db
    path.parent.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    logger.debug("Conectado ao banco SQLite: %s", path)
    return connection


def create_tables(connection: sqlite3.Connection) -> None:
    """Cria tabelas e índices caso ainda não existam.

    Idempotente: usa ``CREATE ... IF NOT EXISTS``, então pode ser chamada
    a cada abertura do banco sem risco de erro nem perda de dados. O
    ``executescript`` faz COMMIT implícito antes de rodar, mas não depois;
    por isso o ``commit()`` explícito ao final.
    """
    connection.executescript(SCHEMA_SQL)
    connection.commit()
    logger.debug("Schema verificado/criado no banco SQLite.")


def initialize_database(database_path: Path | None = None) -> Path:
    """Garante o schema do banco e retorna o caminho utilizado.

    Conveniência para o primeiro uso ou para rodar antes de qualquer
    escrita. Abre a conexão, cria as tabelas se necessário e fecha em
    seguida — não deixa conexão aberta para quem chamou.

    Args:
        database_path: Caminho alternativo do ``.db``. Se omitido, usa
            ``PROJECT_PATHS.promotions_db``.

    Returns:
        Caminho do banco que foi inicializado.
    """
    path = Path(database_path) if database_path else PROJECT_PATHS.promotions_db
    connection = connect_database(path)
    try:
        create_tables(connection)
    finally:
        connection.close()
    logger.info("Banco inicializado: %s", path)
    return path