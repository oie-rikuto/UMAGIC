"""`F-701` 騎手の実力（`docs/spec/003-features.md` / `D-002` / `Q-006`）。"""

from __future__ import annotations

from datetime import date, datetime, timezone

import polars as pl

from umagic.features.f701 import DEFAULT_K, compute_f701
from umagic.features.shrinkage import shrink

NOW = datetime(2026, 8, 18, tzinfo=timezone.utc)


def _race(conn, race_id, race_date, n_starters=4):
    conn.execute(
        "INSERT INTO races (race_id, date, course, race_number, distance, surface, "
        "n_entries, n_starters, source, fetched_at) "
        "VALUES (?, ?, '東京', ?, 2000, '芝', ?, ?, 'netkeiba_jra', ?)",
        [race_id, race_date, race_id, n_starters, n_starters, NOW],
    )


def _runner(conn, race_id, horse_id, popularity, finish_pos, jockey_id):
    if not conn.execute("SELECT 1 FROM horses WHERE horse_id=?", [horse_id]).fetchone():
        conn.execute(
            "INSERT INTO horses VALUES (?, 'h', NULL, NULL, NULL, NULL, 'netkeiba_jra', ?)",
            [horse_id, NOW],
        )
    conn.execute(
        "INSERT INTO runners (race_id, horse_id, number, status, finish_pos, time_sec, "
        "popularity, jockey_id, source, fetched_at) VALUES (?, ?, ?, '出走', ?, 100.0, "
        "?, ?, 'netkeiba_jra', ?)",
        [race_id, horse_id, horse_id % 100 + 1, finish_pos, popularity, jockey_id, NOW],
    )


J1, J2 = 701, 702


def test_shrink_toward_global_residual_mean(conn):
    # 母集団の種: E[finish|popularity=1] の基準を作る
    _race(conn, 1, date(2020, 1, 1))
    _runner(conn, 1, 10, popularity=1, finish_pos=2, jockey_id=709)

    # J1の唯一の騎乗: 人気1番なのに4着（期待より悪い）
    _race(conn, 2, date(2020, 2, 1))
    _runner(conn, 2, 11, popularity=1, finish_pos=4, jockey_id=J1)

    # J2の唯一の騎乗: 人気1番で1着（期待より良い）
    _race(conn, 3, date(2020, 3, 1))
    _runner(conn, 3, 12, popularity=1, finish_pos=1, jockey_id=J2)

    # 対象レース: J1騎乗
    _race(conn, 4, date(2020, 4, 1))
    _runner(conn, 4, 20, popularity=None, finish_pos=None, jockey_id=J1)

    base = pl.DataFrame({"race_id": [4], "horse_id": [20]})
    out = compute_f701(conn, base, as_of=date(2025, 1, 1))
    row = out.filter(pl.col("horse_id") == 20)

    # E[finish|pop=1] (R2の直前) = 2.0 → residual(R2) = 4-2.0 = 2.0（J1の唯一の実績）
    # μ_global (対象日より前の全残差の平均) = (2.0 + (1-3.0)) / 2 = (2.0 + -2.0)/2 = 0.0
    expected = shrink(1, 2.0, k=DEFAULT_K, mu_global=0.0)
    assert abs(row["f701"].to_list()[0] - expected) < 1e-9
    assert row["f701_unavailable"].to_list()[0] == 0


def test_null_jockey_is_unavailable(conn):
    _race(conn, 10, date(2020, 1, 1))
    _runner(conn, 10, 30, popularity=None, finish_pos=None, jockey_id=None)

    base = pl.DataFrame({"race_id": [10], "horse_id": [30]})
    out = compute_f701(conn, base, as_of=date(2025, 1, 1))
    row = out.filter(pl.col("horse_id") == 30)
    assert row["f701"].to_list()[0] is None
    assert row["f701_unavailable"].to_list()[0] == 1


def test_rookie_jockey_falls_back_to_mu_global(conn):
    _race(conn, 20, date(2020, 1, 1))
    _runner(conn, 20, 40, popularity=1, finish_pos=1, jockey_id=709)

    _race(conn, 21, date(2020, 2, 1))
    _runner(conn, 21, 41, popularity=None, finish_pos=None, jockey_id=999)  # 初騎乗の騎手

    base = pl.DataFrame({"race_id": [21], "horse_id": [41]})
    out = compute_f701(conn, base, as_of=date(2025, 1, 1))
    row = out.filter(pl.col("horse_id") == 41)
    assert row["f701_unavailable"].to_list()[0] == 0
    assert row["f701"].to_list()[0] is not None


def test_integration_with_build_features(conn):
    _race(conn, 30, date(2020, 1, 1))
    _runner(conn, 30, 50, popularity=None, finish_pos=None, jockey_id=709)

    from umagic.features.build import build_features

    df = build_features(conn, as_of=date(2025, 1, 1), race_ids=[30], feature_fns=[compute_f701])
    assert "f701" in df.columns
