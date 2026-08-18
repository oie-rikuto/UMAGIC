"""`F-103` ペース適性（`docs/spec/003-features.md` / `D-061` / `D-065`）。"""

from __future__ import annotations

from datetime import date, datetime, timezone

import polars as pl

from umagic.features.f103 import (
    DEFAULT_K,
    FALLBACK_MU_GLOBAL,
    _mu_global_daily,
    _mu_global_for_dates,
    _ols_slope,
    compute_f103,
)
from umagic.features.shrinkage import shrink

NOW = datetime(2026, 8, 18, tzinfo=timezone.utc)


def _race(conn, race_id, race_date, corner_nos, n_starters):
    conn.execute(
        "INSERT INTO races (race_id, date, course, race_number, distance, surface, "
        "n_entries, n_starters, corner_nos, source, fetched_at) "
        "VALUES (?, ?, '東京', ?, 2000, '芝', ?, ?, ?, 'netkeiba_jra', ?)",
        [race_id, race_date, race_id, n_starters, n_starters, corner_nos, NOW],
    )


def _runner(conn, race_id, horse_id, corners, margin=None):
    if not conn.execute("SELECT 1 FROM horses WHERE horse_id=?", [horse_id]).fetchone():
        conn.execute(
            "INSERT INTO horses VALUES (?, 'h', NULL, NULL, NULL, NULL, 'netkeiba_jra', ?)",
            [horse_id, NOW],
        )
    conn.execute(
        "INSERT INTO runners (race_id, horse_id, number, status, finish_pos, time_sec, "
        "corners, margin, source, fetched_at) VALUES (?, ?, ?, '出走', 1, 100.0, ?, ?, "
        "'netkeiba_jra', ?)",
        [race_id, horse_id, horse_id % 100 + 1, corners, margin, NOW],
    )


# --- 単体: OLS の傾き ---------------------------------------------------------

def test_ols_slope_needs_two_points():
    assert _ols_slope([1.0], [2.0]) is None
    assert _ols_slope([], []) is None


def test_ols_slope_zero_variance_is_none():
    assert _ols_slope([0.5, 0.5, 0.5], [1.0, 2.0, 3.0]) is None


def test_ols_slope_known_value():
    # d = 2 + 3p を厳密に満たす2点 → 傾き3.0
    assert _ols_slope([0.0, 1.0], [2.0, 5.0]) == 3.0


# --- 単体: μ_global の日付累積 -----------------------------------------------

def test_mu_global_daily_and_lookup_excludes_same_date_and_future():
    beta_dated = pl.DataFrame({
        "race_id": [1, 2, 3],
        "horse_id": [10, 20, 30],
        "beta": [2.0, 4.0, 10.0],
        "date": [date(2023, 1, 1), date(2023, 1, 1), date(2023, 3, 1)],
    })
    daily = _mu_global_daily(beta_dated)
    assert daily["date"].to_list() == [date(2023, 1, 1), date(2023, 3, 1)]
    assert daily["cum_sum"].to_list() == [6.0, 16.0]
    assert daily["cum_n"].to_list() == [2, 3]

    targets = pl.DataFrame({
        "race_id": [100, 101, 102],
        "horse_id": [1, 2, 3],
        "date": [date(2023, 1, 1), date(2023, 2, 15), date(2023, 3, 1)],
    })
    mu = _mu_global_for_dates(daily, targets).sort("race_id")
    values = dict(zip(mu["race_id"].to_list(), mu["mu_global"].to_list()))
    assert values[100] is None  # 同日以前にデータが無い
    assert values[101] == 3.0  # 2023-01-01 の累積のみ（6.0/2）
    assert values[102] == 3.0  # 自分自身(2023-03-01)の寄与は除く（D-054）


def test_mu_global_daily_ignores_null_beta():
    beta_dated = pl.DataFrame({
        "race_id": [1], "horse_id": [10], "beta": [None], "date": [date(2023, 1, 1)],
    })
    daily = _mu_global_daily(beta_dated)
    assert daily.is_empty()


# --- 結合: compute_f103 -------------------------------------------------------

def test_beta_and_shrink_with_self_exclusion(conn):
    """`D-061`/`D-065`: 自馬を除いた他馬F-101から p_i を計算し、d_i(着差)に回帰する。"""
    # X1・X2 自身の「先行度」を作るための単独の過去走
    _race(conn, 900, date(2019, 1, 1), [1, 2], 2)
    _runner(conn, 900, 901, corners=[1, 1])  # X1: pos=1/2=0.5
    _runner(conn, 900, 902, corners=[2, 2])  # X2: pos=2/2=1.0

    # H の過去走1（R1）: 他馬 X1(f101=0.5), X2(f101=1.0), X3(履歴無しで除外)
    _race(conn, 901, date(2020, 2, 1), [1, 2], 4)
    _runner(conn, 901, 10, corners=[2, 1], margin="ハナ")  # H, d=0.05
    _runner(conn, 901, 901, corners=[1, 2])
    _runner(conn, 901, 902, corners=[2, 2])
    _runner(conn, 901, 903, corners=[4, 4])  # X3: 履歴なし

    # H の過去走2（R2）: 他馬 X1 のみ（R1参戦により f101 が更新されている）
    _race(conn, 902, date(2020, 3, 1), [1, 2], 3)
    _runner(conn, 902, 10, corners=[1, 2], margin="1.1/4")  # H, d=1.25
    _runner(conn, 902, 901, corners=[1, 2])
    _runner(conn, 902, 904, corners=[4, 4])  # X4: 履歴なし

    # 対象レース（本命の予測対象）
    _race(conn, 903, date(2020, 4, 1), [1, 2], 1)
    _runner(conn, 903, 10, corners=None)

    base = pl.DataFrame({"race_id": [903], "horse_id": [10]})
    out = compute_f103(conn, base, as_of=date(2025, 1, 1))
    row = out.filter(pl.col("horse_id") == 10)

    # p_i(R1) = 1 - mean(0.5, 1.0) = 0.25 / p_i(R2) = 1 - mean(0.3125) = 0.6875
    # β = 96/35 (手計算で検証済み)
    expected_beta = 96 / 35
    expected_f103 = shrink(2, expected_beta, k=DEFAULT_K, mu_global=FALLBACK_MU_GLOBAL)
    assert abs(row["f103"].to_list()[0] - expected_f103) < 1e-9
    assert row["f103_unavailable"].to_list()[0] == 0


def test_insufficient_valid_pairs_is_structural(conn):
    """有効な (p_i, d_i) の組が1つ以下 → NaN, unavailable=1。"""
    _race(conn, 800, date(2019, 1, 1), [1], 1)
    _runner(conn, 800, 801, corners=[1])  # X10: pos=1.0

    _race(conn, 801, date(2020, 2, 1), [1], 2)
    _runner(conn, 801, 20, corners=[1], margin="クビ")  # H2の唯一の過去走
    _runner(conn, 801, 801, corners=[1])

    _race(conn, 802, date(2020, 3, 1), [1], 1)
    _runner(conn, 802, 20, corners=None)

    base = pl.DataFrame({"race_id": [802], "horse_id": [20]})
    out = compute_f103(conn, base, as_of=date(2025, 1, 1))
    row = out.filter(pl.col("horse_id") == 20)
    assert row["f103"].to_list()[0] is None
    assert row["f103_unavailable"].to_list()[0] == 1


def test_no_history_is_not_structural(conn):
    """過去走が無い → NaN, unavailable=0。"""
    _race(conn, 700, date(2020, 1, 1), [1], 1)
    _runner(conn, 700, 30, corners=None)

    base = pl.DataFrame({"race_id": [700], "horse_id": [30]})
    out = compute_f103(conn, base, as_of=date(2025, 1, 1))
    row = out.filter(pl.col("horse_id") == 30)
    assert row["f103"].to_list()[0] is None
    assert row["f103_unavailable"].to_list()[0] == 0


def test_zero_variance_p_is_structural(conn):
    """`p_i` に分散が無い（全過去走で同値）と回帰不能 → NaN, unavailable=1。"""
    _race(conn, 600, date(2019, 1, 1), [1], 1)
    _runner(conn, 600, 601, corners=[1])  # Z: pos=1.0
    _race(conn, 601, date(2019, 1, 5), [1], 1)
    _runner(conn, 601, 602, corners=[1])  # W: pos=1.0

    _race(conn, 610, date(2020, 2, 1), [1], 2)
    _runner(conn, 610, 40, corners=[1], margin="1")
    _runner(conn, 610, 601, corners=[1])  # 他馬はZのみ → p_i=0.0

    _race(conn, 611, date(2020, 3, 1), [1], 2)
    _runner(conn, 611, 40, corners=[1], margin="2")
    _runner(conn, 611, 602, corners=[1])  # 他馬はWのみ → p_i=0.0（Zと同値）

    _race(conn, 612, date(2020, 4, 1), [1], 1)
    _runner(conn, 612, 40, corners=None)

    base = pl.DataFrame({"race_id": [612], "horse_id": [40]})
    out = compute_f103(conn, base, as_of=date(2025, 1, 1))
    row = out.filter(pl.col("horse_id") == 40)
    assert row["f103"].to_list()[0] is None
    assert row["f103_unavailable"].to_list()[0] == 1


def test_integration_with_build_features(conn):
    """`FeatureFn` として `build_features` に接続できる。"""
    _race(conn, 500, date(2020, 1, 1), [1], 1)
    _runner(conn, 500, 51, corners=None)

    from umagic.features.build import build_features

    df = build_features(conn, as_of=date(2025, 1, 1), race_ids=[500], feature_fns=[compute_f103])
    assert "f103" in df.columns and "f103_unavailable" in df.columns
