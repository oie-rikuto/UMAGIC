"""エージェントの予想を発走前に記録し、結果確定後に採点する（`D-195`）。

`docs/mcp-server.md` は「この手順に従った予想が市場を上回るかは未検証。
検証するには予想を一定数溜めて回収率を見る以外に方法が無く、現時点では
その蓄積が無い」と書いていた。**その蓄積を作るための仕組み**である。

**記録は発走前でなければ意味がない。** 結果を見た後に「こう買っていれば」
と書けるなら検証にならない（`D-187` が in-sample 予測で示したのと同じ
罠）。`append_prediction()` は対象レースの結果が本番DBに既に存在する場合
を拒否する（`inference.build_overlay()` の `D-184` ガードと同じ考え方）。

採点は `D-008` の規律に従う——回収率には必ずブートストラップ信頼区間を
添え、少数の結果から結論を出さない。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Literal

import duckdb
import numpy as np
import polars as pl

BetType = Literal["単勝", "複勝", "ワイド"]

# `D-008`: 主評価券種は分散の小さい複勝・ワイド。3連単は控除率27.5%かつ
# 超高分散で検出力が無いため、記録の対象にしない
ALLOWED_BET_TYPES: frozenset[str] = frozenset({"単勝", "複勝", "ワイド"})

DEFAULT_LOG_PATH = Path("data/predictions_log.jsonl")


@dataclass(frozen=True)
class Pick:
    """1点の買い目。`horse_numbers` は馬番（ワイドは2頭）。"""

    bet_type: str
    horse_numbers: list[int]
    stake: float = 1.0  # 単位を固定した相対値。金額ではない
    note: str = ""

    def validate(self) -> None:
        if self.bet_type not in ALLOWED_BET_TYPES:
            raise ValueError(
                f"bet_type={self.bet_type!r} は対象外です。"
                f"{sorted(ALLOWED_BET_TYPES)} のいずれかを使ってください"
                "（3連単等は D-008 により検出力が無く記録しません）"
            )
        n_expected = 2 if self.bet_type == "ワイド" else 1
        if len(self.horse_numbers) != n_expected:
            raise ValueError(
                f"{self.bet_type} の horse_numbers は{n_expected}頭です"
                f"（受け取った値: {self.horse_numbers}）"
            )
        if self.stake <= 0:
            raise ValueError(f"stake は正の数です（受け取った値: {self.stake}）")


@dataclass(frozen=True)
class PredictionRecord:
    """1レースぶんの予想。発走前に確定させる。"""

    race_id: int
    race_date: str
    logged_at: str
    agent: str            # どの構成のエージェントか（比較のため）
    picks: list[Pick]
    confidence: str = ""  # 高 / 中 / 低 など、エージェント自身の申告
    reasoning: str = ""
    model_probs: dict[str, float] = field(default_factory=dict)  # 馬番→UMAGICの勝率

    def to_json(self) -> str:
        return json.dumps({
            "race_id": self.race_id, "race_date": self.race_date,
            "logged_at": self.logged_at, "agent": self.agent,
            "confidence": self.confidence, "reasoning": self.reasoning,
            "model_probs": self.model_probs,
            "picks": [
                {"bet_type": p.bet_type, "horse_numbers": p.horse_numbers,
                 "stake": p.stake, "note": p.note}
                for p in self.picks
            ],
        }, ensure_ascii=False)


def _load_raw(log_path: Path) -> list[dict]:
    if not log_path.exists():
        return []
    out = []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def append_prediction(
    record: PredictionRecord, *, log_path: Path = DEFAULT_LOG_PATH,
    conn: duckdb.DuckDBPyConnection | None = None,
) -> None:
    """予想を1件追記する。

    `conn`（本番DBへの接続）を渡すと、**対象レースの結果が既に存在する
    場合に拒否する**——結果を見てから書いた予想は検証に使えないため。
    同じ `race_id` × `agent` の重複も拒否する（後出しでの差し替え防止）。
    """
    for p in record.picks:
        p.validate()
    if not record.picks:
        raise ValueError("picks が空です。買い目を1点以上指定してください")

    if conn is not None:
        exists = conn.execute(
            "SELECT COUNT(*) FROM runners WHERE race_id = ? AND finish_pos IS NOT NULL",
            [record.race_id],
        ).fetchone()[0]
        if exists:
            raise ValueError(
                f"race_id={record.race_id} は既に結果が確定しています。"
                "発走前に記録した予想でなければ検証に使えません（D-195）。"
            )

    for row in _load_raw(log_path):
        if row["race_id"] == record.race_id and row["agent"] == record.agent:
            raise ValueError(
                f"race_id={record.race_id} / agent={record.agent!r} の予想は既にあります。"
                "同じレースへの上書き・後出しはできません（D-195）。"
            )

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(record.to_json() + "\n")


def _payout_lookup(conn: duckdb.DuckDBPyConnection, race_ids: list[int]) -> dict:
    """`(race_id, bet_type, frozenset(numbers))` → 払戻（100円単位の倍率）。"""
    rows = conn.execute(
        "SELECT race_id, bet_type, combination, payout FROM payouts "
        "WHERE race_id = ANY(?) AND bet_type IN ('単勝','複勝','ワイド')",
        [race_ids],
    ).fetchall()
    out = {}
    for race_id, bet_type, combination, payout in rows:
        key = (race_id, bet_type, frozenset(int(x) for x in combination))
        out[key] = float(payout) / 100.0
    return out


def score_predictions(
    conn: duckdb.DuckDBPyConnection, *, log_path: Path = DEFAULT_LOG_PATH,
    agent: str | None = None, n_boot: int = 2000, seed: int = 42,
) -> dict:
    """記録済みの予想を、結果が確定したものだけ採点する。

    戻り値は券種別・全体の `n`・回収率・95%信頼区間（`D-008` の規律。
    リサンプリング単位は賭け1点）。**結果が未確定のレースは除外する。**
    """
    raw = _load_raw(log_path)
    if agent is not None:
        raw = [r for r in raw if r["agent"] == agent]
    if not raw:
        return {"n_races_logged": 0, "n_races_settled": 0, "bets": [], "by_bet_type": {},
                "overall": None, "note": "記録がありません。"}

    race_ids = sorted({r["race_id"] for r in raw})
    settled = {
        rid for (rid,) in conn.execute(
            "SELECT DISTINCT race_id FROM runners "
            "WHERE race_id = ANY(?) AND finish_pos IS NOT NULL", [race_ids],
        ).fetchall()
    }
    payouts = _payout_lookup(conn, sorted(settled))

    bets = []
    for row in raw:
        if row["race_id"] not in settled:
            continue
        for p in row["picks"]:
            key = (row["race_id"], p["bet_type"], frozenset(p["horse_numbers"]))
            ret = payouts.get(key, 0.0)  # 払戻表に無い＝不的中
            bets.append({
                "race_id": row["race_id"], "agent": row["agent"],
                "bet_type": p["bet_type"], "horse_numbers": p["horse_numbers"],
                "stake": p["stake"], "payout_ratio": ret,
                "profit": ret * p["stake"] - p["stake"],
            })

    def summarize(rows: list[dict]) -> dict | None:
        if not rows:
            return None
        stakes = np.array([b["stake"] for b in rows])
        gross = np.array([b["payout_ratio"] * b["stake"] for b in rows])
        roi = gross.sum() / stakes.sum()
        rng = np.random.default_rng(seed)
        idx = np.arange(len(rows))
        boots = np.empty(n_boot)
        for i in range(n_boot):
            s = rng.choice(idx, len(idx), replace=True)
            boots[i] = gross[s].sum() / stakes[s].sum()
        lo, hi = np.percentile(boots, [2.5, 97.5])
        return {
            "n_bets": len(rows), "roi": round(float(roi), 4),
            "ci95": [round(float(lo), 4), round(float(hi), 4)],
            "hit_rate": round(float((gross > 0).mean()), 4),
        }

    by_type = {}
    for bt in sorted({b["bet_type"] for b in bets}):
        by_type[bt] = summarize([b for b in bets if b["bet_type"] == bt])

    n_settled = len({r["race_id"] for r in raw if r["race_id"] in settled})
    return {
        "n_races_logged": len({r["race_id"] for r in raw}),
        "n_races_settled": n_settled,
        "by_bet_type": by_type,
        "overall": summarize(bets),
        "caveat": (
            f"D-008の規律: 単勝回収率の標準誤差は概算 4/√n。"
            f"現在 n={len(bets)} 点で、この規模では回収率の差を主張できない可能性が高い。"
            "信頼区間を必ず添えて報告すること。"
        ),
    }


def list_predictions(
    *, log_path: Path = DEFAULT_LOG_PATH, agent: str | None = None,
) -> pl.DataFrame:
    """記録済みの予想を一覧で返す（採点はしない）。"""
    raw = _load_raw(log_path)
    if agent is not None:
        raw = [r for r in raw if r["agent"] == agent]
    if not raw:
        return pl.DataFrame(schema={
            "race_id": pl.Int64, "race_date": pl.Utf8, "agent": pl.Utf8,
            "confidence": pl.Utf8, "n_picks": pl.Int64, "logged_at": pl.Utf8,
        })
    return pl.DataFrame([{
        "race_id": r["race_id"], "race_date": r["race_date"], "agent": r["agent"],
        "confidence": r.get("confidence", ""), "n_picks": len(r["picks"]),
        "logged_at": r["logged_at"],
    } for r in raw])


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
