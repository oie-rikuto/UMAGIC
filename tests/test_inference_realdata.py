"""`src/umagic/inference.py` の実データ版（`D-184`）。

`data/umagic.duckdb` に対して実行する。CIから除外し、`pytest -m realdata`
でのみ手動実行する（`D-053`と同じ規律）。
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pytest

from umagic.inference import build_overlay
from umagic.sources.base import ParsedShutuba

pytestmark = pytest.mark.realdata

DB_PATH = Path(__file__).parent.parent / "data" / "umagic.duckdb"


def _dummy_shutuba(race_id: int, race_date) -> ParsedShutuba:
    race = {
        "race_id": race_id, "date": race_date, "course": "東京", "race_number": 1,
        "title": "テスト", "grade": None, "surface": "芝", "direction": "左",
        "distance": 2000, "weather": "晴", "track_condition": "良", "post_time": "10:00",
        "race_class": "オープン", "weight_rule": "馬齢", "meeting_no": 1, "meeting_day": 1,
        "n_entries": 1,
    }
    entries = [{
        "number": 1, "frame": 1, "horse_source_key": "9999999999", "horse_name": "テスト馬",
        "sex": "牡", "age": 3, "weight_carried": 55.0,
        "jockey_source_key": "00000", "jockey_name": "テスト騎手",
        "trainer_source_key": "00000", "trainer_name": "テスト調教師",
        "affiliation": "東", "horse_weight": 480, "weight_diff": 0,
    }]
    return ParsedShutuba(race=race, entries=entries)


def test_build_overlay_rejects_race_id_already_in_prod():
    """`D-184`: 対象レースが既に本番DBに実在する場合、重複によるUNION ALLの
    破綻（実測で数時間かかる異常な結合コストの原因になった）を防ぐため、
    明示的に `ValueError` で拒否する。"""
    if not DB_PATH.exists():
        pytest.skip(f"{DB_PATH} が無い（P-0 の取り込みが未実行）")
    prod = duckdb.connect(str(DB_PATH), read_only=True)
    try:
        row = prod.execute("SELECT race_id, date FROM races ORDER BY race_id DESC LIMIT 1").fetchone()
    finally:
        prod.close()
    existing_race_id, existing_date = row

    shutuba = _dummy_shutuba(existing_race_id, existing_date)
    conn = duckdb.connect(":memory:")
    try:
        with pytest.raises(ValueError, match="既に本番DB"):
            build_overlay(conn, shutuba)
    finally:
        conn.close()


def test_build_overlay_accepts_race_id_not_in_prod():
    """本番DBに存在しない`race_id`なら正常に重ね合わせられる。"""
    if not DB_PATH.exists():
        pytest.skip(f"{DB_PATH} が無い（P-0 の取り込みが未実行）")
    prod = duckdb.connect(str(DB_PATH), read_only=True)
    try:
        max_id, max_date = prod.execute("SELECT MAX(race_id), MAX(date) FROM races").fetchone()
    finally:
        prod.close()
    new_race_id = max_id + 1  # 本番DBに絶対に存在しない値

    shutuba = _dummy_shutuba(new_race_id, max_date)
    conn = duckdb.connect(":memory:")
    try:
        rid = build_overlay(conn, shutuba)
        assert rid == new_race_id
        assert conn.execute("SELECT COUNT(*) FROM races WHERE race_id = ?", [new_race_id]).fetchone()[0] == 1
    finally:
        conn.close()
