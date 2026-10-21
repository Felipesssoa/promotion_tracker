import subprocess
import sys
from pathlib import Path


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
