"""`F-101` 逃げ意欲スコア（`003-features.md` テスト観点1〜4）。"""

from __future__ import annotations

from datetime import date, datetime, timezone

import polars as pl

from umagic.features.build import build_features
from umagic.features.f101 import compute_f101

NOW = datetime(2026, 8, 18, tzinfo=timezone.utc)


def _race(conn, race_id, race_date, corner_nos, race_number=None, n_starters=4):
    conn.execute(
        "INSERT INTO races (race_id, date, course, race_number, distance, surface, "
        "n_entries, n_starters, corner_nos, source, fetched_at) "
        "VALUES (?, ?, '東京', ?, 2000, '芝', ?, ?, ?, 'netkeiba_jra', ?)",
        [race_id, race_date, race_number or race_id, n_starters, n_starters, corner_nos, NOW],
    )


def _runner(conn, race_id, horse_id, number, corners, status="出走", finish_pos=1):
    if not conn.execute("SELECT 1 FROM horses WHERE horse_id=?", [horse_id]).fetchone():
        conn.execute("INSERT INTO horses VALUES (?, 'h', NULL, NULL, NULL, NULL, "
                    "'netkeiba_jra', ?)", [horse_id, NOW])
    conn.execute(
        "INSERT INTO runners (race_id, horse_id, number, status, finish_pos, time_sec, "
        "corners, source, fetched_at) VALUES (?, ?, ?, ?, ?, 100.0, ?, 'netkeiba_jra', ?)",
        [race_id, horse_id, number, status, finish_pos, corners, NOW],
    )


# テスト観点1: 1角を含む過去走が1走以上ある
def test_1_computes_from_first_corner_history(conn):
    _race(conn, 1, date(2022, 1, 1), [1, 2, 3, 4])
    _runner(conn, 1, 10, 1, corners=[2, 4, 3, 3])  # 1角=2番手, n_starters=4 → 0.5
    _race(conn, 2, date(2023, 1, 1), [1, 2, 3, 4])
    _runner(conn, 2, 10, 1, corners=[])  # ダミー（対象走自身）

    base = pl.DataFrame({"race_id": [2], "horse_id": [10]})
    out = compute_f101(conn, base, as_of=date(2025, 1, 1))
    assert out["f101"].to_list()[0] == 0.5
    assert out["f101_unavailable"].to_list()[0] == 0


# テスト観点2: 過去走はあるが1角を含むものが無い → NaN, unavailable=1
def test_2_no_first_corner_in_history_is_structural(conn):
    _race(conn, 1, date(2022, 1, 1), [3, 4])  # 1角を含まない
    _runner(conn, 1, 10, 1, corners=[2, 2])
    _race(conn, 2, date(2023, 1, 1), [1, 2, 3, 4])
    _runner(conn, 2, 10, 1, corners=[])

    base = pl.DataFrame({"race_id": [2], "horse_id": [10]})
    out = compute_f101(conn, base, as_of=date(2025, 1, 1))
    assert out["f101"].to_list()[0] is None
    assert out["f101_unavailable"].to_list()[0] == 1


# テスト観点3: 過去走が無い → NaN, unavailable=0
def test_3_no_history_is_not_structural(conn):
    _race(conn, 1, date(2023, 1, 1), [1, 2, 3, 4])
    _runner(conn, 1, 10, 1, corners=[])

    base = pl.DataFrame({"race_id": [1], "horse_id": [10]})
    out = compute_f101(conn, base, as_of=date(2025, 1, 1))
    assert out["f101"].to_list()[0] is None
    assert out["f101_unavailable"].to_list()[0] == 0


# テスト観点4: 3角・4角で代用しない（corner_nos=[3,4] の過去走を無視する）
def test_4_no_substitution_with_3rd_4th_corner(conn):
    _race(conn, 1, date(2022, 1, 1), [3, 4])  # 1角無し
    _runner(conn, 1, 10, 1, corners=[1, 1])   # 仮に「1角」として読むと0.25相当になる罠
    _race(conn, 2, date(2023, 1, 1), [1, 2, 3, 4])
    _runner(conn, 2, 10, 1, corners=[])

    base = pl.DataFrame({"race_id": [2], "horse_id": [10]})
    out = compute_f101(conn, base, as_of=date(2025, 1, 1))
    # corners[0]=1 を1角として誤読していれば値が出てしまう。正しくは無視されNaN
    assert out["f101"].to_list()[0] is None
    assert out["f101_unavailable"].to_list()[0] == 1


def test_quantile_uses_lower_tail_not_mean(conn):
    """凡走（大きい pos_ratio）に引きずられない（下位分位点）ことを確認する。"""
    _race(conn, 1, date(2022, 1, 1), [1, 2, 3, 4], n_starters=10)
    _runner(conn, 1, 10, 1, corners=[1, 1, 1, 1])  # 0.1（逃げた）
    _race(conn, 2, date(2022, 2, 1), [1, 2, 3, 4], n_starters=10)
    _runner(conn, 2, 10, 1, corners=[9, 9, 9, 9])  # 0.9（出遅れた凡走）
    _race(conn, 3, date(2023, 1, 1), [1, 2, 3, 4])
    _runner(conn, 3, 10, 1, corners=[])

    base = pl.DataFrame({"race_id": [3], "horse_id": [10]})
    out_low = compute_f101(conn, base, as_of=date(2025, 1, 1), quantile=0.25)
    out_mean = (0.1 + 0.9) / 2
    assert out_low["f101"].to_list()[0] < out_mean  # 下位分位点は平均より低い


def test_dnf_horse_excluded_from_history():
    """D-044: 完走しなかった馬の corners=NULL は履歴として使えない（正常）。"""
    import duckdb
    from umagic.ops_schema import create_ops_schema
    from umagic.schema import create_schema

    c = duckdb.connect()
    create_schema(c)
    create_ops_schema(c)
    _race(c, 1, date(2022, 1, 1), [1, 2, 3, 4])
    _runner(c, 1, 10, 1, corners=None, status="競走中止", finish_pos=None)
    _race(c, 2, date(2023, 1, 1), [1, 2, 3, 4])
    _runner(c, 2, 10, 1, corners=[])

    base = pl.DataFrame({"race_id": [2], "horse_id": [10]})
    out = compute_f101(c, base, as_of=date(2025, 1, 1))
    # 過去走(中止)はあるが corners が NULL のため使えない。1角を含む過去走が
    # 実質0件 → 構造的欠損と同じ扱いになる
    assert out["f101"].to_list()[0] is None
    assert out["f101_unavailable"].to_list()[0] == 1


def test_integration_with_build_features(conn):
    """FeatureFn として build_features に接続できる。"""
    _race(conn, 1, date(2022, 1, 1), [1, 2, 3, 4])
    _runner(conn, 1, 10, 1, corners=[2, 4, 3, 3])
    _race(conn, 2, date(2023, 1, 1), [1, 2, 3, 4])
    _runner(conn, 2, 10, 1, corners=[])

    df = build_features(conn, as_of=date(2025, 1, 1), race_ids=[2],
                        feature_fns=[compute_f101])
    assert "f101" in df.columns and "f101_unavailable" in df.columns
