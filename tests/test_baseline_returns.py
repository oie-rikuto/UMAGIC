"""ベタ買い戦略の回収率（`docs/spec/005-baseline.md` 4節 / `D-072` `D-073` `D-077`）。"""

from __future__ import annotations

from datetime import date, datetime, timezone

from umagic.baseline import race_ledger

NOW = datetime(2026, 8, 19, tzinfo=timezone.utc)


def _setup_race(conn, race_id=1):
    conn.execute(
        "INSERT INTO races (race_id, date, course, race_number, distance, surface, "
        "n_entries, n_starters, source, fetched_at) "
        "VALUES (?, ?, '東京', 1, 2000, '芝', 4, 4, 'netkeiba_jra', ?)",
        [race_id, date(2020, 1, 1), NOW],
    )
    for number, popularity in [(1, 1), (2, 2), (3, 3), (4, 4)]:
        horse_id = race_id * 100 + number
        conn.execute(
            "INSERT INTO horses VALUES (?, 'h', NULL, NULL, NULL, NULL, 'netkeiba_jra', ?)",
            [horse_id, NOW],
        )
        conn.execute(
            "INSERT INTO runners (race_id, horse_id, number, status, finish_pos, time_sec, "
            "popularity, source, fetched_at) VALUES (?, ?, ?, '出走', ?, 100.0, ?, "
            "'netkeiba_jra', ?)",
            [race_id, horse_id, number, number, popularity, NOW],
        )
    # 複勝は意図的に2行のみ（D-072が動機とした実例: n_entries>=8でも2着払いになるケースを模す）
    payouts = [
        ("単勝", "1", [1], 250),
        ("複勝", "1", [1], 150),
        ("複勝", "2", [2], 120),
        ("ワイド", "1-2", [1, 2], 300),
        ("ワイド", "1-3", [1, 3], 200),
        ("ワイド", "2-3", [2, 3], 180),
    ]
    for bet_type, comb_key, combination, payout in payouts:
        conn.execute(
            "INSERT INTO payouts (race_id, bet_type, comb_key, combination, payout, "
            "source, fetched_at) VALUES (?, ?, ?, ?, ?, 'netkeiba_jra', ?)",
            [race_id, bet_type, comb_key, combination, payout, NOW],
        )


def test_favorite_selects_popularity_one(conn):
    """観点5: `favorite` が `popularity=1` の馬を選ぶ。"""
    _setup_race(conn)
    ledger = race_ledger(conn, [1], strategy="favorite", bet_type="単勝")
    row = ledger.row(0, named=True)
    assert row["n_bets"] == 1
    assert row["n_hits"] == 1
    assert row["payout_yen"] == 250
    assert row["stake_yen"] == 100


def test_fukusho_missing_third_row_does_not_hit(conn):
    """観点6: 複勝payoutが2行のみ → 3着馬(popularity=3)を買っても的中しない。"""
    _setup_race(conn)
    ledger = race_ledger(conn, [1], strategy="uniform", bet_type="複勝")
    row = ledger.row(0, named=True)
    assert row["n_bets"] == 4  # 4頭全頭
    assert row["n_hits"] == 2  # 1番・2番のみ的中（3番・4番はpayout行が無い）
    assert row["payout_yen"] == 150 + 120


def test_missing_payout_combination_yields_zero_no_exception(conn):
    """観点7: `payouts` に無い組み合わせは払戻0円。例外を出さない。"""
    _setup_race(conn)
    ledger = race_ledger(conn, [1], strategy="favorite", bet_type="ワイド")
    row = ledger.row(0, named=True)
    # 1番(favorite)から見た全ペア: 1-2(300), 1-3(200), 1-4(無し→0)
    assert row["n_bets"] == 3
    assert row["n_hits"] == 2
    assert row["payout_yen"] == 300 + 200


def test_uniform_wide_covers_all_pairs(conn):
    _setup_race(conn)
    ledger = race_ledger(conn, [1], strategy="uniform", bet_type="ワイド")
    row = ledger.row(0, named=True)
    assert row["n_bets"] == 6  # C(4,2)
    assert row["n_hits"] == 3
    assert row["payout_yen"] == 300 + 200 + 180
    assert row["stake_yen"] == 600


def test_scratched_horse_excluded_from_uniform_bets(conn):
    _setup_race(conn)
    conn.execute("UPDATE runners SET status='出走取消' WHERE race_id=1 AND number=4")
    ledger = race_ledger(conn, [1], strategy="uniform", bet_type="単勝")
    row = ledger.row(0, named=True)
    assert row["n_bets"] == 3  # 4番は購入対象から除外される


def test_empty_race_ids(conn):
    ledger = race_ledger(conn, [], strategy="favorite", bet_type="単勝")
    assert ledger.is_empty()
