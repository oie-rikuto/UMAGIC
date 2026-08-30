#!/usr/bin/env python3
"""まだ発走していないレースの勝率を予測する（`Q-048`、`D-181`）。

出馬表を取得・パースし、JRA本番DB（`data/umagic.duckdb`）を**書き換えず**
`:memory:` 接続に重ね合わせ（`src/umagic/inference.py`）、既存の
`Stage2FoldRunner`（本番既定設定のまま）で学習・予測する。

使い方:
    uv run python scripts/predict_race.py --race-id 202606030811
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

import duckdb
import polars as pl

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from umagic.cache import LocalCacheFetcher
from umagic.inference import build_overlay
from umagic.orchestration import Stage2FoldRunner
from umagic.sources.netkeiba import parse_shutuba
from umagic.training import Fold

UA = "UMAGIC-dev/0.1 (personal research; contact: repository owner)"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--race-id", required=True)
    ap.add_argument("--cache-dir", default="data/cache")
    ap.add_argument("--sleep", type=float, default=5.0, help="D-014 条件2")
    args = ap.parse_args()

    fetcher = LocalCacheFetcher(cache_dir=Path(args.cache_dir), user_agent=UA,
                               min_interval=args.sleep)
    url = f"https://race.netkeiba.com/race/shutuba.html?race_id={args.race_id}"
    page = fetcher.get(url, source="netkeiba_jra", page_kind="shutuba", source_key=args.race_id)
    shutuba = parse_shutuba(page)

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
    min_date, max_data_date = conn.execute(
        "SELECT MIN(date), MAX(date) FROM races WHERE race_id != ?", [race_id],
    ).fetchone()
    # `train_end` は実データの終端で止める。本番DB（`data/umagic.duckdb`）の
    # 取り込み終端と対象レース日の間に空白期間があると、素朴に
    # `target_date - 1日` を使うと `fold.inner_valid_start`
    # （`train_end` の1年前）がその空白期間に落ち、inner_validが空になって
    # LightGBMが落ちる（実測、2026-08-30）。取り込み終端が対象レース日に
    # 近いほどこの空白は縮む——本番DBを最新化すればこの分岐は事実上効かなく
    # なる
    train_end = min(target_date - timedelta(days=1), max_data_date)
    if max_data_date < target_date - timedelta(days=1):
        gap_days = (target_date - max_data_date).days
        print(f"[注意] 本番DBの取り込み終端（{max_data_date}）と対象レース日"
              f"（{target_date}）の間に空白期間があります。学習は{max_data_date}"
              f"までのデータで行います（実質{gap_days}日分古い）", flush=True)
    fold = Fold(index=0, train_start=min_date, train_end=train_end,
                valid_start=target_date, valid_end=target_date, seed=42)

    # `today` は「今日の実日付」（封印セットの起点、`D-017`）。対象レース日
    # を渡すと、`sealed_years=0` でも封印範囲が「対象レース当日1日」に
    # 縮退し、予測対象そのものが封印されて除外される（実測、2026-08-30）。
    # 対象レースが未来である以上 `grade='G1'` でも封印には該当しないが、
    # `today` には実日付を使うことでこの縮退を避ける
    runner = Stage2FoldRunner(today=date.today(), sealed_years=0)
    print(f"[学習] {fold.train_start}..{fold.train_end} の全履歴で学習し、"
          f"{target_date} を予測します（数分かかります）", flush=True)
    out = runner.predict_fold(conn, fold)

    entry_names = pl.DataFrame([
        {"horse_id": None, "number": e["number"], "horse_name": e["horse_name"]}
        for e in shutuba.entries
    ]).drop("horse_id")
    # horse_id はoverlay内部の解決結果なので、number経由で出走馬名と結合する
    numbers = conn.execute(
        "SELECT horse_id, number FROM runners WHERE race_id = ?", [race_id],
    ).pl()
    result = (
        out.join(numbers, on="horse_id", how="inner")
        .join(entry_names, on="number", how="left")
        .select(["number", "horse_name", "y_pred"])
        .rename({"y_pred": "win_prob"})
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
