#!/usr/bin/env python3
"""UMAGICをLLMエージェントに接続するMCPサーバー（`Q-048`、着手条件を
ユーザーが明示的に解除。手が空いた時間の実験実装、`D-181`の続き）。

`Q-048`が挙げていた2つの前提——出馬表パーサー・運用推論パス——は
`D-181`で揃った。ここではその上に3つのツールを載せる。

    predict_race      まだ発走していないレースの勝率を予測する
                       （`src/umagic/inference.py` をそのまま使う。
                       学習に本番と同じ設定を使うため、実行に数十分
                       かかる——MCPツールとして遅いことを呼び出し側に
                       明示する）
    lookup_decision    `docs/decisions.md` から指定した `D-xxx` を1件返す
    search_decisions   `docs/decisions.md` をキーワード検索する

**市場に対する優位は未確認のまま**（`D-119`〜`D-180`）。`predict_race`
が返す確率をそのまま賭けの根拠にしないこと——`R-030`が要求する「市場への
上乗せ」は実証されていない。

使い方（stdio transport、Claude Desktop 等から接続する想定）:
    uv run python scripts/mcp_server.py
"""

from __future__ import annotations

import re
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import duckdb
import polars as pl
from mcp.server.mcpserver import MCPServer

from umagic.cache import LocalCacheFetcher
from umagic.inference import build_overlay
from umagic.orchestration import Stage2FoldRunner
from umagic.sources.netkeiba import parse_shutuba
from umagic.training import Fold

ROOT = Path(__file__).parent.parent
DECISIONS_PATH = ROOT / "docs" / "decisions.md"
UA = "UMAGIC-dev/0.1 (personal research; contact: repository owner)"

server = MCPServer(
    name="umagic",
    instructions=(
        "JRA平地G1中心の競馬予測プロジェクト（UMAGIC）へのアクセスを提供する。"
        "predict_raceはまだ発走していないレースの勝率を予測するが、"
        "市場（単勝オッズ）に対する優位は未確認（D-119〜D-180）。"
        "確率をそのまま賭けの根拠にしないこと。"
    ),
)


@server.tool()
def predict_race(race_id: str) -> dict:
    """まだ発走していないJRAレースの各馬の勝率を予測する。

    `race_id` は12桁の netkeiba レースID（例: "202606030811"）。出馬表
    （発走の数日前から公開）を取得・パースし、`data/umagic.duckdb`
    （書き換えない）に対象レースを重ね合わせて、本番と同じ設定
    （`Stage2FoldRunner`、Plackett-Luce top-3、`D-113`の正則化）で
    全履歴を学習してから予測する。

    **実行に数十分かかる**（全履歴の特徴量再計算とLightGBM学習のため）。
    呼び出し側はタイムアウトを長めに取ること。

    戻り値の `win_prob` は市場（単勝オッズ）に対する優位が未実証の値
    （`D-119`〜`D-180`）——賭けの根拠にはしないこと。
    """
    fetcher = LocalCacheFetcher(cache_dir=ROOT / "data" / "cache", user_agent=UA, min_interval=5.0)
    url = f"https://race.netkeiba.com/race/shutuba.html?race_id={race_id}"
    page = fetcher.get(url, source="netkeiba_jra", page_kind="shutuba", source_key=race_id)
    shutuba = parse_shutuba(page)

    if not shutuba.entries:
        return {"error": "出馬表が取得できませんでした（未公開、またはページ構造の想定外）"}

    conn = duckdb.connect(":memory:")
    rid = build_overlay(conn, shutuba)

    target_date = shutuba.race["date"]
    min_date, max_data_date = conn.execute(
        "SELECT MIN(date), MAX(date) FROM races WHERE race_id != ?", [rid],
    ).fetchone()
    train_end = min(target_date - timedelta(days=1), max_data_date)
    gap_days = (target_date - max_data_date).days if max_data_date < target_date - timedelta(days=1) else 0

    fold = Fold(index=0, train_start=min_date, train_end=train_end,
                valid_start=target_date, valid_end=target_date, seed=42)
    runner = Stage2FoldRunner(today=date.today(), sealed_years=0)
    out = runner.predict_fold(conn, fold)

    numbers = conn.execute("SELECT horse_id, number FROM runners WHERE race_id = ?", [rid]).pl()
    entry_names = pl.DataFrame([
        {"number": e["number"], "horse_name": e["horse_name"]} for e in shutuba.entries
    ])
    result = (
        out.join(numbers, on="horse_id", how="inner")
        .join(entry_names, on="number", how="left")
        .select(["number", "horse_name", "y_pred"])
        .rename({"y_pred": "win_prob"})
        .sort("win_prob", descending=True)
    )
    conn.close()

    return {
        "race": {
            "title": shutuba.race["title"], "date": str(target_date),
            "course": shutuba.race["course"], "race_number": shutuba.race["race_number"],
            "grade": shutuba.race["grade"], "race_class": shutuba.race["race_class"],
        },
        "training_data_gap_days": gap_days,
        "predictions": [
            {"number": r["number"], "horse_name": r["horse_name"], "win_prob": round(r["win_prob"], 4)}
            for r in result.iter_rows(named=True)
        ],
        "caveat": "市場（単勝オッズ）に対する優位は未実証（D-119〜D-180）。賭けの根拠にしないこと。",
    }


@server.tool()
def lookup_decision(decision_id: str) -> str:
    """`docs/decisions.md` から指定した決定（例: "D-119"）の全文を1件返す。"""
    text = DECISIONS_PATH.read_text(encoding="utf-8")
    m = re.search(rf"^## {re.escape(decision_id)} .*?(?=^## D-\d+|\Z)", text, re.S | re.M)
    if not m:
        return f"{decision_id} は見つかりませんでした。"
    return m.group(0).strip()


@server.tool()
def search_decisions(query: str, max_results: int = 10) -> list[dict]:
    """`docs/decisions.md` をキーワード検索し、一致した決定の見出しと
    冒頭を返す（本文全体は `lookup_decision` で取る）。"""
    text = DECISIONS_PATH.read_text(encoding="utf-8")
    entries = re.split(r"(?=^## D-\d+)", text, flags=re.M)
    hits = []
    for entry in entries:
        if not entry.startswith("## D-"):
            continue
        if query in entry:
            header = entry.splitlines()[0]
            m = re.search(r"^## (D-\d+) (.*)$", header)
            snippet = entry[:300].replace(header, "").strip()
            hits.append({"id": m.group(1), "title": m.group(2), "snippet": snippet})
            if len(hits) >= max_results:
                break
    return hits


if __name__ == "__main__":
    server.run()
