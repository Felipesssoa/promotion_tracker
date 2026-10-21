import datetime
from unittest.mock import patch

from promotion_tracker.ingestion.image_storage import ImageStorage


def test_save_image_includes_profile_and_capture_time(tmp_path):
    fixed_time = datetime.datetime(2026, 10, 21, 13, 52, 46)

    class FixedDateTime(datetime.datetime):
        @classmethod
        def now(cls, tz=None):
            return fixed_time

    storage = ImageStorage()
    storage.bronze_dir = tmp_path

    with patch(
        "promotion_tracker.ingestion.image_storage.datetime.datetime",
        FixedDateTime,
    ):
        first_path = storage.save_image("@mercado teste")
        first_path.touch()
        second_path = storage.save_image("@mercado teste")

    assert first_path == (
        tmp_path / "2026" / "10" / "21" / "mercado teste"
        / "mercado teste_13_52_46.png"
    )
    assert second_path.name == "mercado teste_13_52_46_002.png"
    assert second_path.parent == first_path.parent
