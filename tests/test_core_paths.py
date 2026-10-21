import json

from promotion_tracker.consolidation.consolidator import Consolidator
from promotion_tracker.core.json_io import write_json_atomic
from promotion_tracker.core.paths import ProjectPaths


def test_project_paths_resolve_pipeline_directories(tmp_path):
    paths = ProjectPaths(root=tmp_path)

    assert paths.bronze_dir == tmp_path / "data" / "bronze"
    assert paths.silver_dir == tmp_path / "data" / "silver"
    assert paths.gold_dir == tmp_path / "data" / "gold"
    assert paths.networks_config == tmp_path / "config" / "networks.yaml"
    assert paths.segments_config == tmp_path / "config" / "segments.yaml"


def test_consolidator_uses_injected_project_paths(tmp_path):
    paths = ProjectPaths(root=tmp_path)

    consolidator = Consolidator(paths=paths)

    assert consolidator.silver_dir == paths.silver_dir
    assert consolidator.gold_dir == paths.gold_dir
    assert consolidator.segments_path == paths.segments_config


def test_consolidator_reads_silver_and_writes_gold_under_project_root(tmp_path):
    paths = ProjectPaths(root=tmp_path)
    paths.config_dir.mkdir(parents=True)
    paths.segments_config.write_text(
        "produtos:\n  - produto: arroz\n",
        encoding="utf-8",
    )
    silver_file = paths.silver_dir / "2026" / "10" / "21" / "story.json"
    silver_file.parent.mkdir(parents=True)
    silver_file.write_text(
        json.dumps({
            "encarte_valido": True,
            "perfil": "mercado",
            "produtos": [{"produto": "arroz branco"}],
        }),
        encoding="utf-8",
    )

    output_path = Consolidator(paths=paths).consolidate()

    assert output_path.is_file()
    assert output_path.is_relative_to(paths.gold_dir)
    assert json.loads(paths.gold_dir.joinpath("latest.json").read_text(
        encoding="utf-8"
    ))["resumo"]["linhas_produto"] == 1
    assert not output_path.with_suffix(".json.tmp").exists()


def test_write_json_atomic_creates_parent_and_replaces_target(tmp_path):
    output_path = tmp_path / "nested" / "result.json"
    payload = {"produto": "café", "preco": 5.99}

    write_json_atomic(output_path, payload)

    assert json.loads(output_path.read_text(encoding="utf-8")) == payload
    assert not output_path.with_suffix(".json.tmp").exists()
