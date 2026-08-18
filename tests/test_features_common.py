"""`003-features.md` 共通規約の単体テスト（`D-051` `D-021` `D-058` `D-059`）。"""

from __future__ import annotations

from datetime import date, datetime, timezone

import polars as pl
import pytest

from umagic.features.missing import with_unavailable_indicator
from umagic.features.relative import relativize
from umagic.features.shrinkage import per_row_stat_before, shrink
from umagic.features.window import full_and_recent

NOW = datetime(2026, 8, 18, tzinfo=timezone.utc)


# --- F-902 縮約（D-051） -----------------------------------------------------

def test_shrink_small_n_pulls_toward_mu_global():
    theta = shrink(1, 10.0, k=5.0, mu_global=3.0)
    # n=1, k=5 なので mu_global (3.0) 寄り
    assert abs(theta - ((1 * 10.0 + 5.0 * 3.0) / 6.0)) < 1e-9
    assert abs(theta - 3.0) < abs(10.0 - 3.0)  # x̄ より mu_global に近い


def test_shrink_large_n_approaches_x_bar():
    theta = shrink(100, 10.0, k=5.0, mu_global=3.0)
    assert abs(theta - 10.0) < 0.5  # n=100, k=5 なのでほぼ x̄


def test_shrink_zero_samples_returns_mu_global():
    assert shrink(0, None, k=5.0, mu_global=3.0) == 3.0
    assert shrink(None, None, k=5.0, mu_global=3.0) == 3.0


def test_per_row_stat_before_uses_each_rows_own_date(conn):
    """D-054 の追記事項: 対象行ごとにその行自身の日付で切る。"""
    _seed_races_for_stat(conn)
    base = pl.DataFrame({"race_id": [1, 2], "horse_id": [10, 20]})
    race_dates = {1: date(2023, 1, 1), 2: date(2023, 6, 1)}

    result = per_row_stat_before(
        conn, base, race_dates=race_dates,
        stat_sql="SELECT AVG(ru.time_sec) FROM runners ru JOIN races r USING (race_id) "
                "WHERE r.date < ?",
    )
    # race_id=1（2023-01-01より前）と race_id=2（2023-06-01より前）で
    # 母集団が異なり、値も異なるはず
    vals = dict(zip(result["race_id"].to_list(), result["stat"].to_list()))
    assert vals[1] != vals[2]


def _seed_races_for_stat(conn):
    conn.execute(
        "INSERT INTO races (race_id, date, course, race_number, distance, surface, "
        "n_entries, n_starters, source, fetched_at) VALUES "
        "(100, '2022-01-01', '東京', 1, 2000, '芝', 1, 1, 'netkeiba_jra', ?)", [NOW],
    )
    conn.execute(
        "INSERT INTO races (race_id, date, course, race_number, distance, surface, "
        "n_entries, n_starters, source, fetched_at) VALUES "
        "(200, '2023-03-01', '東京', 1, 2000, '芝', 1, 1, 'netkeiba_jra', ?)", [NOW],
    )
    for hid, rid, t in [(1, 100, 90.0), (2, 200, 110.0)]:
        conn.execute("INSERT INTO horses VALUES (?, 'h', NULL, NULL, NULL, NULL, "
                    "'netkeiba_jra', ?)", [hid, NOW])
        conn.execute(
            "INSERT INTO runners (race_id, horse_id, number, status, finish_pos, "
            "time_sec, source, fetched_at) VALUES (?, ?, 1, '出走', 1, ?, "
            "'netkeiba_jra', ?)", [rid, hid, t, NOW],
        )


# --- F-901 レース内相対化（D-021） -------------------------------------------

def test_relativize_z_and_rank():
    df = pl.DataFrame({
        "race_id": [1, 1, 1],
        "n_starters": [3, 3, 3],
        "val": [10.0, 20.0, 30.0],
    })
    out = relativize(df, "val")
    assert "val_z" in out.columns and "val_rank" in out.columns
    z = out["val_z"].to_list()
    assert z[0] < z[1] < z[2]  # 順序が保たれる
    ranks = out["val_rank"].to_list()
    assert ranks == [1 / 3, 2 / 3, 1.0]  # 昇順rank / n_starters


def test_relativize_zero_std_gives_null_z():
    """race_level=True 相当（全馬同値）を渡すと z が NaN になる。"""
    df = pl.DataFrame({"race_id": [1, 1], "n_starters": [2, 2], "val": [5.0, 5.0]})
    out = relativize(df, "val")
    assert out["val_z"].is_null().all()


def test_relativize_across_races_independent():
    df = pl.DataFrame({
        "race_id": [1, 1, 2, 2],
        "n_starters": [2, 2, 2, 2],
        "val": [1.0, 2.0, 100.0, 200.0],
    })
    out = relativize(df, "val")
    # レースをまたいだ絶対値の差がz-scoreに影響しない
    z1 = out.filter(pl.col("race_id") == 1)["val_z"].to_list()
    z2 = out.filter(pl.col("race_id") == 2)["val_z"].to_list()
    assert z1 == pytest.approx(z2)


# --- 欠損の表現（D-058） -----------------------------------------------------

def test_indicator_structural_null_is_1():
    df = pl.DataFrame({"val": [None], "is_straight": [True]})
    out = with_unavailable_indicator(df, "val", is_structural=pl.col("is_straight"))
    assert out["val_unavailable"].to_list() == [1]


def test_indicator_non_structural_null_is_0():
    """取得漏れ（構造的でない NULL）は 0。過去走が無いケースなど。"""
    df = pl.DataFrame({"val": [None], "is_straight": [False]})
    out = with_unavailable_indicator(df, "val", is_structural=pl.col("is_straight"))
    assert out["val_unavailable"].to_list() == [0]


def test_indicator_value_present_is_0():
    df = pl.DataFrame({"val": [1.5], "is_straight": [True]})
    out = with_unavailable_indicator(df, "val", is_structural=pl.col("is_straight"))
    assert out["val_unavailable"].to_list() == [0]


# --- 全履歴・直近N走の窓（D-059） --------------------------------------------

def test_full_and_recent_window():
    history = pl.DataFrame({
        "horse_id": [1, 1, 1, 1, 2],
        "date": [date(2022, 1, 1), date(2022, 6, 1), date(2022, 9, 1), date(2023, 1, 1),
                date(2023, 1, 1)],
        "val": [10.0, 20.0, 30.0, 40.0, 5.0],
    })
    out = full_and_recent(history, group_col="horse_id", date_col="date", value_col="val", n=2)
    row1 = out.filter(pl.col("horse_id") == 1)
    assert row1["n_all"].to_list() == [4]
    assert row1["val_all"].to_list() == [(10 + 20 + 30 + 40) / 4]
    # 直近2件は新しい順（2023-01-01, 2022-09-01）= 40, 30
    assert row1["n_recent"].to_list() == [2]
    assert row1["val_recent"].to_list() == [(40 + 30) / 2]


def test_full_and_recent_window_fewer_than_n():
    """標本数が N 未満なら全履歴と直近が一致する。"""
    history = pl.DataFrame({
        "horse_id": [1],
        "date": [date(2023, 1, 1)],
        "val": [7.0],
    })
    out = full_and_recent(history, group_col="horse_id", date_col="date", value_col="val", n=5)
    assert out["n_all"].to_list() == [1]
    assert out["n_recent"].to_list() == [1]
    assert out["val_all"].to_list() == out["val_recent"].to_list()
