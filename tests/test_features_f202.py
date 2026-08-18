"""`F-202` 種牡馬の条件別成績（`docs/spec/003-features.md` / `D-066` / `D-067` / `D-068`）。"""

from __future__ import annotations

from datetime import date, datetime, timezone

import polars as pl

from umagic.features.f202 import DEFAULT_K, compute_f202
from umagic.features.shrinkage import shrink

NOW = datetime(2026, 8, 18, tzinfo=timezone.utc)

SIRE1 = 901
SIRE2 = 902


def _race(conn, race_id, race_date, surface, distance, n_starters=4):
    conn.execute(
        "INSERT INTO races (race_id, date, course, race_number, distance, surface, "
        "n_entries, n_starters, source, fetched_at) "
        "VALUES (?, ?, '東京', ?, ?, ?, ?, ?, 'netkeiba_jra', ?)",
        [race_id, race_date, race_id, distance, surface, n_starters, n_starters, NOW],
    )


def _horse(conn, horse_id, sire_id, damsire_id=None):
    if not conn.execute("SELECT 1 FROM horses WHERE horse_id=?", [horse_id]).fetchone():
        conn.execute(
            "INSERT INTO horses VALUES (?, 'h', NULL, ?, NULL, ?, 'netkeiba_jra', ?)",
            [horse_id, sire_id, damsire_id, NOW],
        )


def _runner(conn, race_id, horse_id, finish_pos):
    conn.execute(
        "INSERT INTO runners (race_id, horse_id, number, status, finish_pos, time_sec, "
        "source, fetched_at) VALUES (?, ?, ?, '出走', ?, 100.0, 'netkeiba_jra', ?)",
        [race_id, horse_id, horse_id % 100 + 1, finish_pos, NOW],
    )


def test_shrink_toward_condition_bucket_mu_global(conn):
    # S1 の産駒2頭: いずれも1着（成績が良い）
    _horse(conn, 10, SIRE1)
    _race(conn, 1, date(2020, 1, 1), "芝", 2000)
    _runner(conn, 1, 10, 1)  # perf=1/4=0.25

    _horse(conn, 11, SIRE1)
    _race(conn, 2, date(2020, 2, 1), "芝", 2000)
    _runner(conn, 2, 11, 1)  # perf=1/4=0.25

    # S2 の産駒1頭: 4着（成績が悪い）
    _horse(conn, 12, SIRE2)
    _race(conn, 3, date(2020, 1, 15), "芝", 2000)
    _runner(conn, 3, 12, 4)  # perf=4/4=1.0

    # 対象馬 H: sire=S1, damsire=NULL
    _horse(conn, 20, SIRE1, damsire_id=None)
    _race(conn, 4, date(2020, 3, 1), "芝", 2000)
    _runner(conn, 4, 20, None)

    base = pl.DataFrame({"race_id": [4], "horse_id": [20]})
    out = compute_f202(conn, base, as_of=date(2025, 1, 1))
    row = out.filter(pl.col("horse_id") == 20)

    # μ_global,c(芝,中距離) = (0.25+0.25+1.0)/3 = 0.5
    mu = 0.5
    expected = shrink(2, 0.25, k=DEFAULT_K, mu_global=mu)
    assert abs(row["f202_sire"].to_list()[0] - expected) < 1e-9
    assert row["f202_sire_unavailable"].to_list()[0] == 0
    # damsire は NULL → 構造的欠損
    assert row["f202_damsire"].to_list()[0] is None
    assert row["f202_damsire_unavailable"].to_list()[0] == 1


def test_null_sire_is_structural(conn):
    _horse(conn, 30, sire_id=None)
    _race(conn, 10, date(2020, 1, 1), "芝", 2000)
    _runner(conn, 10, 30, None)

    base = pl.DataFrame({"race_id": [10], "horse_id": [30]})
    out = compute_f202(conn, base, as_of=date(2025, 1, 1))
    row = out.filter(pl.col("horse_id") == 30)
    assert row["f202_sire"].to_list()[0] is None
    assert row["f202_sire_unavailable"].to_list()[0] == 1


def test_no_offspring_falls_back_to_mu_global(conn):
    """条件バケツに該当産駒がまだ0頭 → shrink() のn=0の極限（mu_globalそのもの）。"""
    _horse(conn, 40, sire_id=SIRE1)  # まだ誰も出走していない種牡馬
    _race(conn, 20, date(2020, 1, 1), "芝", 2000)
    _runner(conn, 20, 40, None)

    base = pl.DataFrame({"race_id": [20], "horse_id": [40]})
    out = compute_f202(conn, base, as_of=date(2025, 1, 1))
    row = out.filter(pl.col("horse_id") == 40)
    # 母集団が全く無い（FALLBACK_MU_GLOBAL）
    from umagic.features.f202 import FALLBACK_MU_GLOBAL
    assert abs(row["f202_sire"].to_list()[0] - FALLBACK_MU_GLOBAL) < 1e-9
    assert row["f202_sire_unavailable"].to_list()[0] == 0


def test_distance_band_boundary(conn):
    """D-066: distance<=1400が短距離、1401以上はマイル。"""
    _horse(conn, 50, sire_id=SIRE1)
    _race(conn, 30, date(2020, 1, 1), "芝", 1400)
    _runner(conn, 30, 50, 1)  # 短距離バケツ

    _horse(conn, 51, sire_id=SIRE1)
    _race(conn, 31, date(2020, 2, 1), "芝", 1401)
    _runner(conn, 31, 51, 1)  # マイルバケツ

    # 対象: マイル(1401m)の対象レース。短距離バケツの実績は混ざらない
    _horse(conn, 60, sire_id=SIRE1)
    _race(conn, 32, date(2020, 3, 1), "芝", 1600)
    _runner(conn, 32, 60, None)

    base = pl.DataFrame({"race_id": [32], "horse_id": [60]})
    out = compute_f202(conn, base, as_of=date(2025, 1, 1))
    row = out.filter(pl.col("horse_id") == 60)
    # マイルバケツには horse 51 の1件のみ（n=1）。短距離の horse 50 は混ざらない
    assert row["f202_sire_unavailable"].to_list()[0] == 0


def test_integration_with_build_features(conn):
    _horse(conn, 70, sire_id=None)
    _race(conn, 40, date(2020, 1, 1), "芝", 2000)
    _runner(conn, 40, 70, None)

    from umagic.features.build import build_features

    df = build_features(conn, as_of=date(2025, 1, 1), race_ids=[40], feature_fns=[compute_f202])
    assert "f202_sire" in df.columns and "f202_damsire" in df.columns
