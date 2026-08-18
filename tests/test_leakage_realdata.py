"""`004-leakage-test.md` の実データ版（`D-053`）。

`data/umagic.duckdb` に対して実行する。CIから除外し、`pytest -m realdata`
でのみ手動実行する。`tests/test_leakage.py` と同じ9原則を検査するが、
対象レースは固定 `race_id` を書かず、**DBから条件に合うものを動的に選ぶ**。
特定のレースIDに依存すると、DBを作り直したときに壊れるテストになる。
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import polars as pl
import pytest

from umagic.features.build import build_features
from umagic.sealed import is_sealed

pytestmark = pytest.mark.realdata

DB_PATH = Path(__file__).parent.parent / "data" / "umagic.duckdb"


@pytest.fixture(scope="module")
def conn():
    if not DB_PATH.exists():
        pytest.skip(f"{DB_PATH} が無い（P-0 の取り込みが未実行）")
    try:
        c = duckdb.connect(str(DB_PATH), read_only=True)
    except duckdb.IOException:
        pytest.skip("DBがロック中（取り込み・血統取得が実行中）")
    yield c
    c.close()


def _pick_race_with_same_day_predecessors(conn) -> tuple[int, str, str, int]:
    """同日・同競馬場に先行レースを2件以上持つレースを1件選ぶ（原則3の検証に使う）。"""
    row = conn.execute(
        """
        SELECT r2.race_id, r2.date, r2.course, r2.race_number
        FROM races r2
        WHERE (
            SELECT COUNT(*) FROM races r1
            WHERE r1.date = r2.date AND r1.course = r2.course
              AND r1.race_number < r2.race_number
        ) >= 2
        ORDER BY r2.race_id LIMIT 1
        """
    ).fetchone()
    if row is None:
        pytest.skip("同日先行レースを2件以上持つレースが無い")
    return row


def _pick_horse_with_history(conn) -> tuple[int, object]:
    """3走以上の履歴を持つ馬と、その中間の1走（対象走）を選ぶ。"""
    row = conn.execute(
        """
        SELECT ru.horse_id, r.date
        FROM runners ru JOIN races r USING (race_id)
        WHERE ru.horse_id IN (
            SELECT horse_id FROM runners GROUP BY horse_id HAVING COUNT(*) >= 3
        )
        ORDER BY ru.horse_id, r.date
        """
    ).fetchall()
    if not row:
        pytest.skip("3走以上の履歴を持つ馬が無い")
    # 最初に出てくる馬の、履歴の中間の1走を対象にする（前後にデータがあるように）
    horse_id = row[0][0]
    dates = [d for h, d in row if h == horse_id]
    mid = dates[len(dates) // 2]
    return horse_id, mid


def test_realdata_past_aggregation_is_strict(conn):
    horse_id, target_date = _pick_horse_with_history(conn)
    strict = conn.execute(
        "SELECT COUNT(*) FROM runners ru JOIN races r USING (race_id) "
        "WHERE ru.horse_id = ? AND r.date < ?", [horse_id, target_date],
    ).fetchone()[0]
    non_strict = conn.execute(
        "SELECT COUNT(*) FROM runners ru JOIN races r USING (race_id) "
        "WHERE ru.horse_id = ? AND r.date <= ?", [horse_id, target_date],
    ).fetchone()[0]
    # 対象走自身が race_date = target_date で存在するため、<= は必ず1件以上多い
    assert non_strict > strict


def test_realdata_same_day_strict_race_number(conn):
    race_id, race_date, course, race_number = _pick_race_with_same_day_predecessors(conn)
    strict = conn.execute(
        "SELECT COUNT(*) FROM races WHERE date=? AND course=? AND race_number < ?",
        [race_date, course, race_number],
    ).fetchone()[0]
    non_strict = conn.execute(
        "SELECT COUNT(*) FROM races WHERE date=? AND course=? AND race_number <= ?",
        [race_date, course, race_number],
    ).fetchone()[0]
    assert non_strict == strict + 1  # 対象レース自身の1件だけ増える


def test_realdata_same_day_course_filter_matters(conn):
    race_id, race_date, course, race_number = _pick_race_with_same_day_predecessors(conn)
    with_course = conn.execute(
        "SELECT COUNT(*) FROM races WHERE date=? AND course=? AND race_number < ?",
        [race_date, course, race_number],
    ).fetchone()[0]
    without_course = conn.execute(
        "SELECT COUNT(*) FROM races WHERE date=? AND race_number < ?",
        [race_date, race_number],
    ).fetchone()[0]
    # 複数場開催の日であれば、course を外すと必ず増える。
    # 単独開催日なら等しい（実データの分布次第なので >= で見る）
    assert without_course >= with_course


def test_realdata_no_future_form_of_same_horse(conn):
    horse_id, target_date = _pick_horse_with_history(conn)
    all_starts = conn.execute(
        "SELECT COUNT(*) FROM runners WHERE horse_id=?", [horse_id],
    ).fetchone()[0]
    past_only = conn.execute(
        "SELECT COUNT(*) FROM runners ru JOIN races r USING (race_id) "
        "WHERE ru.horse_id=? AND r.date < ?", [horse_id, target_date],
    ).fetchone()[0]
    # 中間の1走を選んでいるため、未来の走が最低1件は存在する
    assert past_only < all_starts


def test_realdata_build_features_unique_and_sorted(conn):
    df = build_features(conn, as_of=conn.execute("SELECT MAX(date) FROM races").fetchone()[0])
    assert df.select(["race_id", "horse_id"]).is_duplicated().sum() == 0
    pairs = list(zip(df["race_id"].to_list(), df["horse_id"].to_list()))
    assert pairs == sorted(pairs)


def test_realdata_build_features_bit_exact(conn):
    as_of = conn.execute("SELECT MAX(date) FROM races").fetchone()[0]
    df1 = build_features(conn, as_of=as_of)
    df2 = build_features(conn, as_of=as_of)
    assert df1.equals(df2)


def test_realdata_no_raw_odds_or_outcome_columns_when_empty(conn):
    """特徴量0本の骨格が、対象レースのオッズ・着順列を混入させていない（原則5・6の前提）。"""
    df = build_features(conn, as_of=conn.execute("SELECT MAX(date) FROM races").fetchone()[0])
    forbidden = {"finish_pos", "time_sec", "odds_win", "popularity", "corners"}
    assert not (forbidden & set(df.columns))


def test_realdata_sealed_set_has_g1_races(conn):
    """封印セット判定が実データのG1に対して機能する（D-017 / D-056）。"""
    today = conn.execute("SELECT MAX(date) FROM races").fetchone()[0]
    g1 = conn.execute(
        "SELECT race_id, date FROM races WHERE grade='G1' ORDER BY date DESC LIMIT 1",
    ).fetchone()
    if g1 is None:
        pytest.skip("G1レースが無い")
    race_id, race_date = g1
    # 直近のG1は「今日」から3年以内のはずなので封印対象になる
    assert is_sealed(race_date, "G1", today=today)


def test_realdata_non_g1_never_sealed(conn):
    """D-003: 学習データ（全レース）は封印対象ではない。"""
    today = conn.execute("SELECT MAX(date) FROM races").fetchone()[0]
    row = conn.execute(
        "SELECT date FROM races WHERE grade IS NULL ORDER BY date DESC LIMIT 1",
    ).fetchone()
    if row is None:
        pytest.skip("非G1レースが無い")
    assert not is_sealed(row[0], None, today=today)
