import sqlite3

from promotion_tracker.common.paths import ProjectPaths
from promotion_tracker.consolidation.database import connect_database


def test_connect_database_uses_gold_promotions_database(tmp_path, monkeypatch):
    paths = ProjectPaths(root=tmp_path)
    monkeypatch.setattr(
        "promotion_tracker.consolidation.database.PROJECT_PATHS",
        paths,
    )

    connection = connect_database()
    try:
        assert paths.promotions_db.is_file()
        assert isinstance(connection, sqlite3.Connection)
        assert connection.row_factory is sqlite3.Row
        assert connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        assert connection.execute("SELECT 1").fetchone()[0] == 1
    finally:
        connection.close()


def test_connect_database_accepts_custom_path(tmp_path):
    database_path = tmp_path / "nested" / "custom.db"

    connection = connect_database(database_path)
    try:
        assert database_path.is_file()
        assert connection.execute("SELECT 1").fetchone()[0] == 1
    finally:
        connection.close()
