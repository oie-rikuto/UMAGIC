"""G1/G2/G3開催前トリガーの自動化（試験導入）。

`docs/mcp-server.md`の`predict_race`/`suggest_exotic_bets`と同じロジックを
MCPサーバーを介さず直接呼び出す、Cron起動用の軽量版。決定論的に計算できる
部分（モデル勝率・Harville組み合わせ確率）だけをここで計算する。

**ここでは計算できないもの**: 単勝オッズ。netkeibaは`shutuba`ページはもちろん
モバイル版オッズページ（`race.sp.netkeiba.com/?pid=odds_view`）でも
JavaScriptで後から埋める方式で、静的HTMLの時点では常に`---.-`のプレース
ホルダのまま（本スクリプト実装時に実データで確認済み）。市場との乖離を
使う判断は、Cronで起動されたエージェント自身がWeb検索でオッズを取得して
補う必要がある。

出力: 標準出力にJSON。`weight_confirmed=false`の場合はモデル勝率を計算せず
その旨だけを返す（Cron側で「まだ」と判断してリトライできるように）。
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import duckdb  # noqa: E402
import polars as pl  # noqa: E402

from umagic.cache import LocalCacheFetcher, RobotsDisallowed  # noqa: E402
from umagic.harville import SUPPORTED_BET_TYPES, top_combos  # noqa: E402
from umagic.inference import build_overlay  # noqa: E402
from umagic.production_model import CACHE_META_FILENAME, predict_with_cache  # noqa: E402
from umagic.sources.netkeiba import PostPositionsNotDrawn, parse_shutuba  # noqa: E402

PREDICTION_CACHE_DIR = ROOT / "data" / "prediction_cache"
UA = "umagic-research/1.0 (contact: ricky.big.h@gmail.com)"


def _fetch_shutuba(race_id: str):
    fetcher = LocalCacheFetcher(cache_dir=ROOT / "data" / "cache", user_agent=UA, min_interval=5.0)
    url = f"https://race.netkeiba.com/race/shutuba.html?race_id={race_id}"
    page = fetcher.get(url, source="netkeiba_jra", page_kind="shutuba", source_key=race_id)
    return parse_shutuba(page)


def run(race_id: str) -> dict:
    try:
        shutuba = _fetch_shutuba(race_id)
    except RobotsDisallowed as e:
        return {"race_id": race_id, "error": f"robots.txt拒否: {e}"}
    except PostPositionsNotDrawn as e:
        return {"race_id": race_id, "status": "not_drawn", "detail": str(e)}
    except Exception as e:  # noqa: BLE001
        return {"race_id": race_id, "error": f"{type(e).__name__}: {e}"}

    if not shutuba.entries:
        return {"race_id": race_id, "status": "no_entries"}

    weight_confirmed = all(e.get("horse_weight") is not None for e in shutuba.entries)
    result = {
        "race_id": race_id,
        "race": {k: (str(v) if isinstance(v, date) else v) for k, v in shutuba.race.items()},
        "weight_confirmed": weight_confirmed,
        "n_entries": len(shutuba.entries),
    }
    if not weight_confirmed:
        result["status"] = "weight_not_yet"
        return result

    meta_path = PREDICTION_CACHE_DIR / CACHE_META_FILENAME
    if not meta_path.exists():
        result["error"] = f"推論キャッシュがありません: {meta_path}"
        return result

    conn = duckdb.connect(":memory:")
    try:
        rid = build_overlay(conn, shutuba)
    except ValueError as e:
        conn.close()
        result["error"] = f"重ね合わせ失敗（既に結果確定済みの可能性）: {e}"
        return result

    target_date = shutuba.race["date"]
    cache_meta = json.loads(meta_path.read_text(encoding="utf-8"))
    trained_through = date.fromisoformat(cache_meta["trained_through"])
    gap_days = max(0, (target_date - trained_through).days - 1)

    out = predict_with_cache(conn, rid, target_date, PREDICTION_CACHE_DIR)
    numbers = conn.execute("SELECT horse_id, number FROM runners WHERE race_id = ?", [rid]).pl()
    entry_names = pl.DataFrame([
        {"number": e["number"], "horse_name": e["horse_name"]} for e in shutuba.entries
    ])
    joined = (
        out.join(numbers, on="horse_id", how="inner")
        .join(entry_names, on="number", how="left")
        .select(["number", "horse_name", "win_prob"])
        .sort("win_prob", descending=True)
    )
    conn.close()

    win_prob = {int(r["number"]): float(r["win_prob"]) for r in joined.iter_rows(named=True)}
    horse_names = {int(r["number"]): r["horse_name"] for r in joined.iter_rows(named=True)}

    combos = {}
    for bet_type in sorted(SUPPORTED_BET_TYPES):
        top = top_combos(win_prob, bet_type, top_n=8)
        combos[bet_type] = [{"numbers": c.numbers, "prob": round(c.prob, 5)} for c in top]

    result.update({
        "status": "ready",
        "training_data_gap_days": gap_days,
        "predictions": [
            {"number": n, "horse_name": horse_names[n], "win_prob": round(p, 4)}
            for n, p in sorted(win_prob.items(), key=lambda kv: -kv[1])
        ],
        "harville_combos": combos,
        "caveat": "単勝オッズは静的HTMLに含まれないため未計算。市場との乖離を使う"
                  "判断はこの出力を受け取った側がWeb検索でオッズを補うこと。",
    })
    return result


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("使い方: uv run python scripts/auto_predict.py <race_id>", file=sys.stderr)
        sys.exit(1)
    print(json.dumps(run(sys.argv[1]), ensure_ascii=False, indent=2))
