"""`F-805` 出走馬の基礎情報（`docs/spec/003-features.md` / `D-131`）。"""

from __future__ import annotations

from datetime import date, datetime, timezone

import polars as pl

from umagic.features.f805 import compute_f805

NOW = datetime(2026, 8, 18, tzinfo=timezone.utc)


def _race(conn, race_id, race_date, race_number=1, n=2):
    conn.execute(
        "INSERT INTO races (race_id, date, course, race_number, distance, surface, "
        "n_entries, n_starters, source, fetched_at) "
        "VALUES (?, ?, '東京', ?, 2000, '芝', ?, ?, 'netkeiba_jra', ?)",
        [race_id, race_date, race_number, n, n, NOW],
    )


def _runner(conn, race_id, horse_id, *, age, sex, wc, status="出走"):
    if not conn.execute("SELECT 1 FROM horses WHERE horse_id=?", [horse_id]).fetchone():
        conn.execute(
            "INSERT INTO horses VALUES (?, 'h', NULL, NULL, NULL, NULL, 'netkeiba_jra', ?)",
            [horse_id, NOW],
        )
    conn.execute(
        "INSERT INTO runners (race_id, horse_id, number, status, age, sex, weight_carried, "
        "source, fetched_at) VALUES (?, ?, ?, ?, ?, ?, ?, 'netkeiba_jra', ?)",
        [race_id, horse_id, horse_id % 100 + 1, status, age, sex, wc, NOW],
    )


def test_returns_three_columns(conn):
    _race(conn, 1, date(2020, 3, 1), n=2)
    _runner(conn, 1, 10, age=4, sex="牡", wc=57.0)
    _runner(conn, 1, 11, age=3, sex="牝", wc=54.0)

    out = compute_f805(conn, pl.DataFrame({"race_id": [1, 1], "horse_id": [10, 11]}),
                       as_of=date(2025, 1, 1)).sort("horse_id")
    assert out["f805_age"].to_list() == [4.0, 3.0]
    assert out["f805_sex"].to_list() == ["牡", "牝"]
    assert out["f805_weight_carried"].to_list() == [57.0, 54.0]


def test_sex_is_kept_as_string_for_category_rounding(conn):
    """`f805_sex` は文字列のまま出す（`D-092` の丸めに載せるため）。"""
    _race(conn, 1, date(2020, 3, 1), n=1)
    _runner(conn, 1, 10, age=5, sex="セ", wc=58.0)

    out = compute_f805(conn, pl.DataFrame({"race_id": [1], "horse_id": [10]}),
                       as_of=date(2025, 1, 1))
    assert out.schema["f805_sex"] == pl.Utf8
    assert out.row(0, named=True)["f805_sex"] == "セ"


def test_scratched_runner_is_not_returned(conn):
    """`出走取消` は基底集合に含めない（`D-109` と同じ母集団）。"""
    _race(conn, 1, date(2020, 3, 1), n=1)
    _runner(conn, 1, 10, age=4, sex="牡", wc=57.0, status="出走取消")

    out = compute_f805(conn, pl.DataFrame({"race_id": [1], "horse_id": [10]}),
                       as_of=date(2025, 1, 1))
    row = out.row(0, named=True)
    assert row["f805_age"] is None
    assert row["f805_weight_carried"] is None


def test_empty_base_returns_schema(conn):
    out = compute_f805(conn, pl.DataFrame({"race_id": [], "horse_id": []}),
                       as_of=date(2025, 1, 1))
    assert out.is_empty()
    for c in ("f805_age", "f805_sex", "f805_weight_carried"):
        assert c in out.columns
