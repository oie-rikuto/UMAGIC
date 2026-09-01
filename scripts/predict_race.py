#!/usr/bin/env python3
"""まだ発走していないレースの勝率を予測する（`Q-048`、`D-181`〜`D-183`）。

出馬表を取得・パースし、JRA本番DB（`data/umagic.duckdb`）を**書き換えず**
`:memory:` 接続に重ね合わせ（`src/umagic/inference.py`）、事前に作った
推論キャッシュ（`scripts/build_prediction_cache.py`、`D-183`）で予測する。
キャッシュが無い・古い場合は先に作ること。

使い方:
    uv run python scripts/build_prediction_cache.py   # 先に1回（本番DB更新時のみ）
    uv run python scripts/predict_race.py --race-id 202606030811
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import duckdb
import polars as pl

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from umagic.cache import LocalCacheFetcher
from umagic.inference import build_overlay
from umagic.production_model import CACHE_META_FILENAME, predict_with_cache
from umagic.sources.netkeiba import PostPositionsNotDrawn, parse_shutuba

UA = "UMAGIC-dev/0.1 (personal research; contact: repository owner)"
CACHE_DIR = Path("data/prediction_cache")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--race-id", required=True)
    ap.add_argument("--cache-dir", default="data/cache", help="D-014のページキャッシュ（推論キャッシュとは別物）")
    ap.add_argument("--prediction-cache-dir", default=str(CACHE_DIR))
    ap.add_argument("--sleep", type=float, default=5.0, help="D-014 条件2")
    args = ap.parse_args()

    pred_cache_dir = Path(args.prediction_cache_dir)
    meta_path = pred_cache_dir / CACHE_META_FILENAME
    if not meta_path.exists():
        print(f"[エラー] 推論キャッシュがありません: {meta_path}", file=sys.stderr)
        print("先に `uv run python scripts/build_prediction_cache.py` を実行してください", file=sys.stderr)
        return 1

    fetcher = LocalCacheFetcher(cache_dir=Path(args.cache_dir), user_agent=UA,
                               min_interval=args.sleep)
    url = f"https://race.netkeiba.com/race/shutuba.html?race_id={args.race_id}"
    page = fetcher.get(url, source="netkeiba_jra", page_kind="shutuba", source_key=args.race_id)
    try:
        shutuba = parse_shutuba(page)
    except PostPositionsNotDrawn as e:
        print(f"[待機] {e}", file=sys.stderr)
        return 2

    print(f"[出馬表] {shutuba.race['date']} {shutuba.race['course']}"
          f"{shutuba.race['race_number']}R {shutuba.race['title']}"
          f"({shutuba.race['grade'] or shutuba.race['race_class'] or '-'}) "
          f"{len(shutuba.entries)}頭", flush=True)
    if not shutuba.entries:
        print("[エラー] 出馬表が取得できませんでした（未公開、またはページ構造の想定外）", file=sys.stderr)
        return 1

    conn = duckdb.connect(":memory:")
    race_id = build_overlay(conn, shutuba)
    target_date = shutuba.race["date"]

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    trained_through = date.fromisoformat(meta["trained_through"])
    gap_days = (target_date - trained_through).days - 1
    if gap_days > 0:
        print(f"[注意] 推論キャッシュの学習終端（{trained_through}）と対象レース日"
              f"（{target_date}）の間に{gap_days}日の空白があります。本番DBを"
              f"最新化してキャッシュを作り直すと精度が上がります", flush=True)

    print("[予測] キャッシュ済みモデルで予測します（数十秒〜数分）", flush=True)
    out = predict_with_cache(conn, race_id, target_date, pred_cache_dir)

    numbers = conn.execute(
        "SELECT horse_id, number FROM runners WHERE race_id = ?", [race_id],
    ).pl()
    entry_names = pl.DataFrame([
        {"number": e["number"], "horse_name": e["horse_name"]} for e in shutuba.entries
    ])
    result = (
        out.join(numbers, on="horse_id", how="inner")
        .join(entry_names, on="number", how="left")
        .select(["number", "horse_name", "win_prob"])
        .sort("win_prob", descending=True)
    )

    print(f"\n=== {shutuba.race['title']} 予測勝率 ===")
    for row in result.iter_rows(named=True):
        print(f"  {row['number']:2d}番 {row['horse_name']:14s} {row['win_prob']*100:5.1f}%")
    print(f"\n合計: {result['win_prob'].sum():.4f}（1.0に近いはず）")

    out_path = f"data/predict_{args.race_id}.parquet"
    result.write_parquet(out_path)
    print(f"\n[save] {out_path}")
    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
