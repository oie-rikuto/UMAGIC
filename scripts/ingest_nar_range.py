#!/usr/bin/env python3
"""地方競馬（NAR）1場分を実際に取り込むスクリプト（`Q-047` 段階②、`D-176`）。

`scripts/ingest_range.py`（JRA本番）をほぼそのまま踏襲するが、次の2点が違う。

- `Source` に `NetkeibaNarSource` を使う（`course_filter` で対象1場に絞る）
- 既定の `--db` は `data/umagic_nar.duckdb`——**JRA本番DB
  （`data/umagic.duckdb`）とは別ファイル**にする。NAR探索は並行探索であり
  （`D-174`）JRAの本番データを一切汚さないことを最優先にした（`D-176`）

`race_class` の CHECK 制約は大井のクラス表記（`A1`〜`C3` と隣接併合）を
含む形に拡張済み（`schema.py`）。大井以外の場は見出し書式が未検証のため、
`--course` に大井以外を渡しても `race_class` は解析されない（`None` の
まま安全側に倒れる。取り込み自体は失敗しない）。

`D-014` 条件2（既定5秒間隔）により、日次インデックス取得＋レースごとの
archive取得で数千レース規模だと数時間単位の実行時間になる（`P-0` と同じ）。
バックグラウンド実行を想定し、既存DBに追記的に動く（中断・再開できる）。

使い方:
    uv run python scripts/ingest_nar_range.py \\
        --from 2022-01-01 --to 2024-12-31
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from datetime import date, datetime
from pathlib import Path

import duckdb

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from umagic.cache import LocalCacheFetcher, RobotsDisallowed
from umagic.loader import completed_race_keys, ingest_range
from umagic.ops_schema import create_ops_schema
from umagic.quality import run_quality_checks
from umagic.schema import create_schema
from umagic.sources.netkeiba import NetkeibaNarSource

UA = "UMAGIC-dev/0.1 (personal research; contact: repository owner)"


def parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", default="data/umagic_nar.duckdb")
    ap.add_argument("--cache-dir", default="data/cache")
    ap.add_argument("--course", default="大井", help="対象1場（Q-047段階②は大井のみ検証済み）")
    ap.add_argument("--from", dest="date_from", required=True, type=parse_date)
    ap.add_argument("--to", dest="date_to", required=True, type=parse_date)
    ap.add_argument("--sleep", type=float, default=5.0,
                    help="D-014 条件2。5.0未満は指定できない")
    ap.add_argument("--no-resume", action="store_true",
                    help="取り込み済みのレースも作り直す。スキーマを変えた後の"
                         "再構築に使う。キャッシュがあればネットワークは使わない")
    args = ap.parse_args()

    db_path = Path(args.db)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not db_path.exists()

    conn = duckdb.connect(str(db_path))
    if is_new:
        create_schema(conn)
        create_ops_schema(conn)
        print(f"[init] 新規DB作成: {db_path}", flush=True)
    else:
        print(f"[resume] 既存DBを使用: {db_path}", flush=True)

    fetcher = LocalCacheFetcher(cache_dir=Path(args.cache_dir), user_agent=UA,
                               min_interval=args.sleep)
    source = NetkeibaNarSource(fetcher, course_filter=args.course)

    n_done = len(completed_race_keys(conn))
    if n_done:
        print(f"[resume] 取り込み済み {n_done} レースをスキップする", flush=True)

    def on_race_error(out):
        print(f"  [{out.outcome}] {out.source_key}: {out.detail}", flush=True)

    def on_day(day, race_keys, outcomes):
        if not race_keys:
            return  # 対象場の開催が無い日
        counts = Counter(o.outcome for o in outcomes)
        print(f"[{day}] {len(race_keys)}R / 累計 "
              f"ok={counts['ok']} empty={counts['empty']} "
              f"error={counts['http_error'] + counts['parse_error']}", flush=True)

    try:
        outcomes = ingest_range(conn, fetcher, source, args.date_from, args.date_to,
                                resume=not args.no_resume,
                                on_day=on_day, on_race_error=on_race_error)
    except RobotsDisallowed as e:
        print(f"[中断] {e}", file=sys.stderr, flush=True)
        return 2

    counts = Counter(o.outcome for o in outcomes)
    print(f"\n取り込み完了: ok={counts['ok']} empty={counts['empty']} "
          f"http_error={counts['http_error']} parse_error={counts['parse_error']}", flush=True)

    report = run_quality_checks(conn, scope_from=args.date_from, scope_to=args.date_to)
    print()
    print(report.to_markdown())
    # `unknown_race_class` はNARでは大井のクラス文字を持たない収得賞金帯
    # レース・重賞（`D-176`）で構造的に一定割合出る。fail系ではなくwarn系
    # なので exit_code には影響しない（このスクリプトでも変えない）
    return report.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
