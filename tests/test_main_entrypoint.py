import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from playwright._impl._errors import TargetClosedError

from promotion_tracker.extraction.llm.schemas import (
    Oferta,
    ResultadoExtracao,
    Variante,
)
from promotion_tracker.extraction.llm_extractor import LLMExtractor
from promotion_tracker.extraction.ocr.analyzer import AnaliseOCR, RegiaoOferta
from promotion_tracker.ingestion.instagram_client import InstagramClient
from promotion_tracker.main import parse_args, run_pipeline


@pytest.mark.parametrize(
    ("browser", "expected_profile"),
    (
        ("edge", "edge_ig_profile"),
        ("brave", "brave_ig_profile"),
        ("firefox", "firefox_ig_profile"),
        ("mozilla", "firefox_ig_profile"),
    ),
)
def test_browser_selects_dedicated_profile(
    browser: str, expected_profile: str
) -> None:
    client = InstagramClient(browser=browser)

    assert client.browser_name == expected_profile.removesuffix("_ig_profile")
    assert client.profile_dir.name == expected_profile


def test_invalid_browser_is_rejected() -> None:
    with pytest.raises(ValueError, match="Navegador inválido"):
        InstagramClient(browser="chrome")


def test_firefox_waits_for_login_in_existing_context() -> None:
    client = InstagramClient(browser="firefox")
    client.page = Mock()

    with (
        patch.object(client, "_start_firefox"),
        patch.object(client, "_is_logged_in", side_effect=[False, False, True]),
        patch.object(client, "_stop_edge") as stop,
        patch.object(client, "_open_manual_login") as open_login,
    ):
        client.page.wait_for_timeout.side_effect = lambda _milliseconds: None
        client.connect()

    stop.assert_not_called()
    open_login.assert_not_called()
    client.page.goto.assert_called_once_with(
        client.LOGIN_URL,
        wait_until="domcontentloaded",
    )


def test_manual_login_does_not_navigate_current_page() -> None:
    client = InstagramClient(browser="brave")
    client.page = Mock()
    client.page.locator.return_value.count.return_value = 1

    with patch.object(client, "_is_logged_in", return_value=True) as check:
        client._wait_for_manual_login(timeout=1)

    client.page.goto.assert_called_once_with(
        client.LOGIN_URL,
        wait_until="domcontentloaded",
    )
    check.assert_called_once_with(timeout=2, navigate=False)


def test_dense_flyer_fallback_uses_ocr_regions(tmp_path: Path) -> None:
    image_path = tmp_path / "encarte.png"
    image_path.touch()
    extractor = object.__new__(LLMExtractor)
    crop_results = [
        ResultadoExtracao(
            encarte_valido=True,
            ofertas=[
                Oferta(
                    variantes=[
                        Variante(
                            produto="Leite",
                            marca=None,
                            quantidade=1,
                            unidade_medida="un",
                        )
                    ],
                    preco_atual=5.99,
                    preco_anterior=None,
                    preco_condicao=None,
                    condicao=None,
                )
            ],
        ),
        ResultadoExtracao(
            encarte_valido=True,
            ofertas=[
                Oferta(
                    variantes=[
                        Variante(
                            produto="Queijo",
                            marca=None,
                            quantidade=1,
                            unidade_medida="un",
                        )
                    ],
                    preco_atual=9.99,
                    preco_anterior=None,
                    preco_condicao=None,
                    condicao=None,
                )
            ],
        ),
    ]
    analysis = AnaliseOCR(
        found_products=["Leite", "Queijo"],
        regions=[
            RegiaoOferta((0, 0, 485, 400), ["Leite"], "test"),
            RegiaoOferta((0, 400, 485, 861), ["Queijo"], "test"),
        ],
        image_size=(485, 861),
        use_full_image=True,
        reason="encarte denso",
        dense=True,
        fallback_regions=[
            RegiaoOferta((0, 0, 485, 400), ["Leite"], "test"),
            RegiaoOferta((0, 400, 485, 861), ["Queijo"], "test"),
        ],
    )
    query_calls: list[tuple[int, int, int]] = []

    def query(_image, _path, number, total, attempt=1):
        query_calls.append((number, total, attempt))
        return crop_results[number - 1]

    extractor._query_model = query
    extractor._dense_slices = Mock(
        side_effect=lambda _image: [Mock(), Mock()]
    )
    extractor._expand_offers = Mock(return_value=[{"produto": "Leite"}, {"produto": "Queijo"}])
    extractor._revisit_segment_products = Mock(return_value=[])
    extractor._save_json = Mock()
    extractor._save_rejected_image = Mock()

    with patch(
        "promotion_tracker.extraction.llm_extractor.Image.open"
    ) as open_image:
        open_image.return_value.__enter__.return_value.convert.return_value = Mock()
        extractor._extract_data(image_path, "perfil", analysis)

    assert query_calls == [(1, 2, 1), (2, 2, 1)]
    extractor._save_rejected_image.assert_not_called()
    assert extractor._save_json.call_count == 1


def test_manual_login_handles_closed_target() -> None:
    client = InstagramClient(browser="brave")
    client.page = Mock()
    client.page.locator.side_effect = TargetClosedError("target closed")

    assert client._is_logged_in(navigate=False) is False


def test_brave_uses_dedicated_cdp_profile(tmp_path: Path) -> None:
    client = InstagramClient(browser="brave")
    client.profile_dir = tmp_path / "brave_ig_profile"

    with (
        patch.object(client, "_find_edge", return_value="/usr/bin/brave"),
        patch.object(client, "_port_open", side_effect=[False, True]),
        patch.object(client, "_connect_edge") as connect,
        patch("promotion_tracker.ingestion.instagram_client.subprocess.Popen")
        as popen,
    ):
        popen.return_value.poll.return_value = None
        client._start_edge()

    assert popen.call_args.args[0][0:2] == [
        "/usr/bin/brave",
        f"--user-data-dir={client.profile_dir}",
    ]
    connect.assert_called_once_with()


def test_pipeline_extracts_original_bronze_images(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, object]] = []

    class Extractor:
        def __init__(self, **kwargs: object) -> None:
            calls.append(kwargs)

        def process_bronze_folder(self) -> None:
            calls.append({"processed": True})

    monkeypatch.setattr(
        "promotion_tracker.extraction.llm_extractor.LLMExtractor",
        Extractor,
    )

    def extract() -> None:
        Extractor(use_ocr_images=False).process_bronze_folder()

    run_pipeline(extract, lambda: None, lambda: None)

    assert calls == [{"use_ocr_images": False}, {"processed": True}]


def test_parse_args_defaults_to_edge(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["promotion-tracker"])

    assert parse_args().browser == "edge"


def test_parse_args_accepts_firefox(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["promotion-tracker", "--browser", "firefox"])

    assert parse_args().browser == "firefox"


def test_main_script_imports_from_project_and_external_working_directories(
    tmp_path: Path,
) -> None:
    project_root = Path(__file__).resolve().parents[1]
    main_path = project_root / "src" / "promotion_tracker" / "main.py"
    probe = (
        "import importlib, runpy, sys; "
        "module = runpy.run_path(sys.argv[1], run_name='import_probe'); "
        "module['_ensure_src_on_path'](); "
        "importlib.import_module('promotion_tracker.consolidation.consolidator'); "
        "importlib.import_module('promotion_tracker.ingestion.instagram_client'); "
        "importlib.import_module('promotion_tracker.normalization.normalizer')"
    )

    for working_directory in (project_root, tmp_path):
        subprocess.run(
            [sys.executable, "-c", probe, str(main_path)],
            cwd=working_directory,
            check=True,
            capture_output=True,
            text=True,
        )


def test_pipeline_runs_every_stage_in_order() -> None:
    calls: list[str] = []

    def record(name: str):
        def step() -> None:
            calls.append(name)

        return step

    run_pipeline(
        record("extract"),
        record("normalize"),
        record("consolidate"),
    )

    assert calls == ["extract", "normalize", "consolidate"]


def test_pipeline_propagates_stage_failure() -> None:
    calls: list[str] = []

    def fail() -> None:
        calls.append("extract")
        raise RuntimeError("falha")

    try:
        run_pipeline(
            fail,
            lambda: calls.append("normalize"),
            lambda: calls.append("consolidate"),
        )
    except RuntimeError:
        pass
    else:
        raise AssertionError("A falha deveria ser propagada")

    assert calls == ["extract"]
