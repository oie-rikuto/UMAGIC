"""`012-data-quality.md` のテスト観点: fail 系9件の検出、finish_pos_rank の
同着6ケース、warn 系5件、実データ相当の受け入れケース、終了コード。"""

from __future__ import annotations

from datetime import date

import pytest

from tests.conftest import NOW
from umagic.quality import run_quality_checks


def _race(conn, race_id, n_entries, n_starters, corner_nos=None, race_number=None):
    conn.execute(
        "INSERT INTO races (race_id, date, course, race_number, distance, surface, "
        "n_entries, n_starters, corner_nos, source, fetched_at) "
        "VALUES (?, '2023-01-01', '東京', ?, 2000, '芝', ?, ?, ?, 'netkeiba_jra', ?)",
        [race_id, race_number or race_id, n_entries, n_starters, corner_nos, NOW],
    )


def _horse(conn, horse_id):
    if conn.execute("SELECT 1 FROM horses WHERE horse_id=?", [horse_id]).fetchone():
        return
    conn.execute("INSERT INTO horses VALUES (?, 'h', NULL, NULL, NULL, NULL, 'netkeiba_jra', ?)",
                 [horse_id, NOW])


def _runner(conn, race_id, horse_id, number, status="出走", finish_pos=None,
           time_sec=None, corners=None, odds_win=None, popularity=None):
    _horse(conn, horse_id)
    conn.execute(
        "INSERT INTO runners (race_id, horse_id, number, status, finish_pos, time_sec, "
        "corners, odds_win, popularity, source, fetched_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'netkeiba_jra', ?)",
        [race_id, horse_id, number, status, finish_pos, time_sec, corners,
         odds_win, popularity, NOW],
    )


def _clean_race(conn, race_id=1, n=4, corner_nos=(1, 2, 3, 4), race_number=None):
    """全検査を通過する最小構成のレースを1つ作る。"""
    _race(conn, race_id, n, n, list(corner_nos), race_number)
    for i in range(1, n + 1):
        _runner(conn, race_id, i, i, finish_pos=i, time_sec=100.0 + i,
               corners=[1] * len(corner_nos), odds_win=float(i), popularity=i)


def test_clean_race_zero_fail(conn):
    _clean_race(conn)
    report = run_quality_checks(conn)
    assert report.n_fail == 0
    assert report.exit_code == 0


# --- fail 系9件が検出できること --------------------------------------------

def test_headcount_starters_detects_mismatch(conn):
    _race(conn, 1, 4, 4)  # n_starters=4 と宣言
    for i in range(1, 4):  # 実際は3人しか出走している
        _runner(conn, 1, i, i, finish_pos=i, time_sec=100.0)
    report = run_quality_checks(conn)
    assert report.fail_counts["headcount_starters"] == 1


def test_headcount_entries_detects_mismatch(conn):
    _race(conn, 1, 5, 3)
    for i in range(1, 4):
        _runner(conn, 1, i, i, finish_pos=i, time_sec=100.0)
    report = run_quality_checks(conn)
    assert report.fail_counts["headcount_entries"] == 1


def test_orphan_race_detected(conn):
    """runners が0行のレース。INNER JOIN では拾えない（headcount_* も併発）。"""
    _race(conn, 1, 16, 16)
    report = run_quality_checks(conn)
    assert report.fail_counts["orphan_race"] == 1
    assert report.fail_counts["headcount_starters"] == 1
    assert report.fail_counts["headcount_entries"] == 1


@pytest.mark.parametrize("positions,expect_detect", [
    ([1, 2, 3, 4], False),
    ([1, 2, 2, 4], False),
    ([1, 1, 3], False),
    ([1, 2, 2, 3], True),
    ([1, 1, 2], True),
    ([1, 2, 4], True),
])
def test_finish_pos_rank_tie_cases(conn, positions, expect_detect):
    _race(conn, 1, len(positions), len(positions))
    for i, pos in enumerate(positions, start=1):
        _runner(conn, 1, i, i, finish_pos=pos, time_sec=100.0)
    report = run_quality_checks(conn)
    n = report.fail_counts["finish_pos_rank"]
    assert (n > 0) == expect_detect


def test_status_columns_detects_missing_finish_pos(conn):
    _race(conn, 1, 1, 1)
    _runner(conn, 1, 1, 1, status="出走", finish_pos=None, time_sec=None)
    report = run_quality_checks(conn)
    assert report.fail_counts["status_columns"] == 1


def test_status_domain_passes_on_defined_values(conn):
    """定義済みの status では status_domain は0件（正常系の回帰）。"""
    _clean_race(conn, n=3)
    report = run_quality_checks(conn)
    assert report.fail_counts["status_domain"] == 0


def test_status_domain_sql_detects_undefined_value_bypassing_check():
    """`001-schema.md` の CHECK 制約を経由しない経路（他ソース由来の投入等）を
    想定し、`status_domain` の SQL 自体が異常値を検出できることを確認する。
    `runners` テーブルは CHECK 制約があるため直接は再現できず、同じ列構成で
    CHECK を外したテーブルに対して同一の検査SQLを実行する。
    """
    import duckdb

    from umagic.quality import FAIL_CHECKS

    c = duckdb.connect()
    c.execute("CREATE TABLE runners (race_id BIGINT, horse_id BIGINT, status VARCHAR, "
             "finish_pos SMALLINT, time_sec DECIMAL(6,1), popularity SMALLINT, "
             "odds_win DECIMAL(7,1), corners SMALLINT[])")
    c.execute("CREATE TABLE races (race_id BIGINT, corner_nos SMALLINT[])")
    c.execute("INSERT INTO runners VALUES (1, 1, '取消', NULL, NULL, NULL, NULL, NULL)")
    rows = c.execute(FAIL_CHECKS["status_domain"]).fetchall()
    assert len(rows) == 1


def test_payout_horses_detects_nonexistent_number(conn):
    _clean_race(conn, n=4)
    conn.execute(
        "INSERT INTO payouts VALUES (1, '単勝', '9', [9], 100, 1, 'netkeiba_jra', ?)", [NOW],
    )
    report = run_quality_checks(conn)
    assert report.fail_counts["payout_horses"] == 1


def test_payout_horses_ignores_wakuren(conn):
    """枠連は combination が枠番であり馬番と照合しないため検出されない。"""
    _clean_race(conn, n=4)
    conn.execute(
        "INSERT INTO payouts VALUES (1, '枠連', '9-9', [9, 9], 100, 1, 'netkeiba_jra', ?)", [NOW],
    )
    report = run_quality_checks(conn)
    assert report.fail_counts["payout_horses"] == 0


def test_odds_monotonic_detects_reversal(conn):
    _race(conn, 1, 2, 2)
    _runner(conn, 1, 1, 1, finish_pos=1, time_sec=100.0, odds_win=12.3, popularity=1)
    _runner(conn, 1, 2, 2, finish_pos=2, time_sec=101.0, odds_win=4.2, popularity=2)
    report = run_quality_checks(conn)
    assert report.fail_counts["odds_monotonic"] == 1


def test_corners_uniform_detects_all_runners_wrong_length(conn):
    """全馬そろって要素数が corner_nos と違う（旧検査では素通りする典型例）。"""
    _race(conn, 1, 2, 2, corner_nos=[3, 4])
    _runner(conn, 1, 1, 1, finish_pos=1, time_sec=100.0, corners=[1, 2, 3, 4])
    _runner(conn, 1, 2, 2, finish_pos=2, time_sec=101.0, corners=[1, 2, 3, 4])
    report = run_quality_checks(conn)
    assert report.fail_counts["corners_uniform"] == 2


def test_corners_uniform_ignores_dnf_null(conn):
    """D-044: 競走中止馬の corners=NULL は corners_uniform に落ちない。"""
    _race(conn, 1, 2, 2, corner_nos=[1, 2, 3, 4])
    _runner(conn, 1, 1, 1, status="出走", finish_pos=1, time_sec=100.0, corners=[4, 4, 4, 4])
    _runner(conn, 1, 2, 2, status="競走中止", finish_pos=None, time_sec=None, corners=None)
    report = run_quality_checks(conn)
    assert report.fail_counts["corners_uniform"] == 0


# --- 実データ相当の受け入れケース（日本ダービー2023 型） --------------------

def test_acceptance_case_dnf_race_passes_all_checks(conn):
    """202305021211 相当: 18頭・競走中止1頭でも全 fail が0件。"""
    _race(conn, 1, 18, 18, corner_nos=[1, 2, 3, 4])
    for i in range(1, 18):
        _runner(conn, 1, i, i, finish_pos=i, time_sec=140.0 + i,
               corners=[4, 4, 4, 4], odds_win=float(i) + 1, popularity=i)
    _runner(conn, 1, 18, 18, status="競走中止", finish_pos=None, time_sec=None, corners=None)
    report = run_quality_checks(conn)
    assert report.n_fail == 0


def test_acceptance_case_scratch_race_passes(conn):
    """201905030611 相当: 15頭・競走除外2頭でも全 fail が0件。"""
    _race(conn, 1, 15, 13, corner_nos=[3, 4])
    for i in range(1, 14):
        _runner(conn, 1, i, i, finish_pos=i, time_sec=90.0 + i,
               corners=[1, 1], odds_win=float(i) + 1, popularity=i)
    for i in (14, 15):
        _runner(conn, 1, i, i, status="競走除外", finish_pos=None, time_sec=None, corners=None)
    report = run_quality_checks(conn)
    assert report.n_fail == 0


# --- warn 系 -----------------------------------------------------------------

def test_warn_checks_report_rate(conn):
    _clean_race(conn, n=3)
    conn.execute(
        "INSERT INTO rejected_rows VALUES ('netkeiba_jra', '1', '9', 'unknown_finish_marker', '?', ?)",
        [NOW],
    )
    conn.execute(
        "INSERT INTO fetch_log VALUES ('u1', 'netkeiba_jra', 'archive', '1', 200, 'ok', NULL, ?)",
        [NOW],
    )
    conn.execute(
        "INSERT INTO fetch_log VALUES ('u2', 'netkeiba_jra', 'archive', '2', NULL, 'http_error', 'x', ?)",
        [NOW],
    )
    report = run_quality_checks(conn)
    assert report.warn_rates["rejected_rate"] == (1, 4)  # 3 runners + 1 rejected
    assert report.warn_rates["fetch_incomplete"] == (1, 2)
    assert report.warn_rates["laps_coverage"] == (1, 1)  # laps 無し
    assert report.warn_rates["odds_coverage"] == (0, 1)


def test_corners_missing_warn_counts_only_finishers(conn):
    _race(conn, 1, 2, 2, corner_nos=[1, 2, 3, 4])
    _runner(conn, 1, 1, 1, status="出走", finish_pos=1, time_sec=100.0, corners=None)  # 異常
    _runner(conn, 1, 2, 2, status="競走中止", finish_pos=None, time_sec=None, corners=None)  # 正常
    report = run_quality_checks(conn)
    assert report.warn_rates["corners_missing"] == (1, 1)  # 分母は完走馬のみ


# --- レポートと終了コード ----------------------------------------------------

def test_report_exit_code_nonzero_when_fail(conn):
    _race(conn, 1, 4, 4)  # runners を1件も入れない → orphan_race で fail
    report = run_quality_checks(conn)
    assert report.exit_code != 0
    assert "fail" in report.to_markdown()


def test_quality_findings_recorded_without_deleting_source_data(conn):
    """D-041: 不良を見つけても quality_findings に記録するのみで、元データは触らない。"""
    _race(conn, 1, 4, 4)
    n_races_before = conn.execute("SELECT COUNT(*) FROM races").fetchone()[0]
    report = run_quality_checks(conn)
    assert conn.execute("SELECT COUNT(*) FROM races").fetchone()[0] == n_races_before
    findings = conn.execute(
        "SELECT check_id, severity FROM quality_findings WHERE run_id=?", [report.run_id],
    ).fetchall()
    assert ("orphan_race", "fail") in findings
