"""年別内訳とレポート出力（`docs/spec/005-baseline.md` 3節・6節 / `D-018` / `R-027`）。"""

from __future__ import annotations

from datetime import date, datetime, timezone

from umagic.baseline import by_era_breakdown, run_baseline, target_races

NOW = datetime(2026, 8, 19, tzinfo=timezone.utc)
TODAY = date(2026, 8, 19)


def _race(conn, race_id, race_date, grade=None, n_starters=3):
    conn.execute(
        "INSERT INTO races (race_id, date, course, race_number, distance, surface, "
        "grade, n_entries, n_starters, source, fetched_at) "
        "VALUES (?, ?, '東京', ?, 2000, '芝', ?, ?, ?, 'netkeiba_jra', ?)",
        [race_id, race_date, race_id, grade, n_starters, n_starters, NOW],
    )


def _runner(conn, race_id, number, odds_win, finish_pos, popularity):
    horse_id = race_id * 100 + number
    conn.execute(
        "INSERT INTO horses VALUES (?, 'h', NULL, NULL, NULL, NULL, 'netkeiba_jra', ?)",
        [horse_id, NOW],
    )
    conn.execute(
        "INSERT INTO runners (race_id, horse_id, number, status, finish_pos, time_sec, "
        "odds_win, popularity, source, fetched_at) VALUES (?, ?, ?, '出走', ?, 100.0, ?, ?, "
        "'netkeiba_jra', ?)",
        [race_id, horse_id, number, finish_pos, odds_win, popularity, NOW],
    )
    conn.execute(
        "INSERT INTO payouts (race_id, bet_type, comb_key, combination, payout, "
        "source, fetched_at) VALUES (?, '単勝', ?, ?, ?, 'netkeiba_jra', ?)",
        [race_id, str(number), [number], 300 if finish_pos == 1 else 0, NOW],
    )


def _seed(conn):
    _race(conn, 1, date(2020, 1, 1))
    _runner(conn, 1, 1, 2.0, 1, 1)
    _runner(conn, 1, 2, 4.0, 2, 2)

    _race(conn, 2, date(2020, 6, 1))
    _runner(conn, 2, 1, 3.0, 1, 1)
    _runner(conn, 2, 2, 5.0, 2, 2)

    _race(conn, 3, date(2021, 1, 1), grade="G1")  # 封印外（3年より前）
    _runner(conn, 3, 1, 2.5, 1, 1)
    _runner(conn, 3, 2, 6.0, 2, 2)


def test_by_era_n_races_sum_matches_population_total(conn):
    """観点14: `by_era` の各年の `n_races` 合計が母集団全体と一致する。"""
    _seed(conn)
    era = by_era_breakdown(conn, today=TODAY)

    for population in ("all", "g1"):
        total = target_races(conn, population=population, today=TODAY).races.height
        probability_rows = era.filter(
            (era["population"] == population) & (era["metric_kind"] == "probability")
            & (era["metric_name"] == "log_loss")
        )
        assert probability_rows["n_races"].sum() == total


def test_run_baseline_produces_report(conn):
    _seed(conn)
    report = run_baseline(conn, today=TODAY, bootstrap_n=50, seed=1)

    assert report.n_sealed_g1_excluded == 0  # 3レースとも封印期間外
    assert len(report.probability) == 2  # all / g1
    assert len(report.returns) == 2 * 2 * 3  # population × strategy × bet_type

    md = report.to_markdown()
    assert "ベースラインレポート" in md
    assert "確率指標" in md
    assert "回収率" in md
    assert "年別内訳" in md
    assert "g1" in md


def test_run_baseline_with_no_races_does_not_crash(conn):
    report = run_baseline(conn, today=TODAY, bootstrap_n=10, seed=1)
    md = report.to_markdown()
    assert "ベースラインレポート" in md
