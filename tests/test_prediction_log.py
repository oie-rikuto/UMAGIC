"""`D-195` 予想の記録・採点のテスト。

発走前記録の強制（結果確定済みレースの拒否）と、`D-008` の規律
（信頼区間つきの回収率）が守られることを確認する。
"""

import duckdb
import pytest

from umagic.prediction_log import (
    Pick,
    PredictionRecord,
    append_prediction,
    list_predictions,
    now_iso,
    score_predictions,
)


def _record(race_id: int = 202699010101, agent: str = "selector-v1", **kw) -> PredictionRecord:
    return PredictionRecord(
        race_id=race_id, race_date="2026-09-06", logged_at=now_iso(), agent=agent,
        picks=kw.pop("picks", [Pick(bet_type="複勝", horse_numbers=[3])]), **kw,
    )


@pytest.fixture
def settled_conn():
    """1レースだけ結果が確定しているDB。払戻は複勝3番=180円、ワイド3-5番=420円。"""
    conn = duckdb.connect(":memory:")
    conn.execute("CREATE TABLE runners (race_id BIGINT, number SMALLINT, finish_pos SMALLINT)")
    conn.execute("CREATE TABLE payouts (race_id BIGINT, bet_type VARCHAR, "
                 "combination SMALLINT[], payout BIGINT)")
    conn.execute("INSERT INTO runners VALUES (1, 3, 1), (1, 5, 2), (1, 7, 8)")
    conn.execute("INSERT INTO payouts VALUES "
                 "(1, '複勝', [3], 180), (1, 'ワイド', [3, 5], 420), (1, '単勝', [3], 320)")
    return conn


def test_append_and_score_roundtrip(settled_conn, tmp_path):
    log = tmp_path / "log.jsonl"
    append_prediction(_record(race_id=1, picks=[
        Pick(bet_type="複勝", horse_numbers=[3]),
        Pick(bet_type="ワイド", horse_numbers=[3, 5]),
        Pick(bet_type="複勝", horse_numbers=[7]),  # 不的中
    ]), log_path=log)

    out = score_predictions(settled_conn, log_path=log)
    assert out["n_races_logged"] == 1
    assert out["n_races_settled"] == 1
    # 複勝: 1.8 + 0.0 = 1.8 / 2点 = 0.9
    assert out["by_bet_type"]["複勝"]["roi"] == pytest.approx(0.9)
    assert out["by_bet_type"]["複勝"]["hit_rate"] == pytest.approx(0.5)
    assert out["by_bet_type"]["ワイド"]["roi"] == pytest.approx(4.2)
    # 全体: (1.8 + 4.2 + 0.0) / 3 = 2.0
    assert out["overall"]["roi"] == pytest.approx(2.0)
    assert len(out["overall"]["ci95"]) == 2
    assert out["overall"]["ci95"][0] <= out["overall"]["roi"] <= out["overall"]["ci95"][1]


def test_append_rejects_race_with_known_result(settled_conn, tmp_path):
    """発走前記録の強制。結果を見てから書いた予想は検証に使えない。"""
    with pytest.raises(ValueError, match="既に結果が確定"):
        append_prediction(_record(race_id=1), log_path=tmp_path / "log.jsonl",
                          conn=settled_conn)


def test_append_rejects_duplicate_race_and_agent(tmp_path):
    """同じレース・同じエージェントでの後出し差し替えを拒否する。"""
    log = tmp_path / "log.jsonl"
    append_prediction(_record(race_id=1), log_path=log)
    with pytest.raises(ValueError, match="既にあります"):
        append_prediction(_record(race_id=1), log_path=log)
    # 別エージェントなら同じレースに記録できる（構成比較のため）
    append_prediction(_record(race_id=1, agent="analyst-v1"), log_path=log)
    assert len(list_predictions(log_path=log)) == 2


def test_pick_validation_rejects_unsupported_bet_types():
    """`D-008`: 3連単等は検出力が無いため記録対象外。"""
    with pytest.raises(ValueError, match="対象外"):
        Pick(bet_type="三連単", horse_numbers=[1, 2, 3]).validate()
    with pytest.raises(ValueError, match="2頭"):
        Pick(bet_type="ワイド", horse_numbers=[1]).validate()
    with pytest.raises(ValueError, match="1頭"):
        Pick(bet_type="複勝", horse_numbers=[1, 2]).validate()
    with pytest.raises(ValueError, match="stake"):
        Pick(bet_type="複勝", horse_numbers=[1], stake=0).validate()


def test_score_excludes_unsettled_races(settled_conn, tmp_path):
    """結果が未確定のレースは採点から除く（記録には残る）。"""
    log = tmp_path / "log.jsonl"
    append_prediction(_record(race_id=1), log_path=log)
    append_prediction(_record(race_id=999), log_path=log)  # 結果なし
    out = score_predictions(settled_conn, log_path=log)
    assert out["n_races_logged"] == 2
    assert out["n_races_settled"] == 1
    assert out["overall"]["n_bets"] == 1


def test_score_on_empty_log_is_safe(settled_conn, tmp_path):
    out = score_predictions(settled_conn, log_path=tmp_path / "nothing.jsonl")
    assert out["n_races_logged"] == 0
    assert out["overall"] is None
