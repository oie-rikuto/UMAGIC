#!/usr/bin/env python3
"""日付範囲を実際に取り込み、品質検査を実行するスクリプト（`docs/tasks.md` P-0 最終タスク）。

`P-0` の完了条件（2〜3年分を取り込んで `fail` が0件）は、netkeiba への
実アクセスに `D-014` 条件2（既定5秒間隔）がかかるため、数千レース規模だと
数時間単位の実行時間になる。このスクリプトはバックグラウンド実行を想定し、
既存の DuckDB ファイルに対して追記的に動く（何度でも中断・再開できる）。

使い方:
    uv run python scripts/ingest_range.py --db data/umagic.duckdb \\
        --from 2023-01-01 --to 2023-12-31
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
from umagic.sources.netkeiba import NetkeibaJraSource

UA = "UMAGIC-dev/0.1 (personal research; contact: repository owner)"


def parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", default="data/umagic.duckdb")
    ap.add_argument("--cache-dir", default="data/cache")
    ap.add_argument("--from", dest="date_from", required=True, type=parse_date)
    ap.add_argument("--to", dest="date_to", required=True, type=parse_date)
    ap.add_argument("--sleep", type=float, default=5.0,
                    help="D-014 条件2。5.0未満は指定できない")
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
    source = NetkeibaJraSource(fetcher)

    n_done = len(completed_race_keys(conn))
    if n_done:
        print(f"[resume] 取り込み済み {n_done} レースをスキップする", flush=True)

    def on_race_error(out):
        print(f"  [{out.outcome}] {out.source_key}: {out.detail}", flush=True)

    def on_day(day, race_keys, outcomes):
        if not race_keys:
            return  # JRA中央の開催が無い日。平日は大半がこれ
        counts = Counter(o.outcome for o in outcomes)
        print(f"[{day}] {len(race_keys)}R / 累計 "
              f"ok={counts['ok']} empty={counts['empty']} "
              f"error={counts['http_error'] + counts['parse_error']}", flush=True)

    try:
        outcomes = ingest_range(conn, fetcher, source, args.date_from, args.date_to,
                                resume=True, on_day=on_day, on_race_error=on_race_error)
    except RobotsDisallowed as e:
        print(f"[中断] {e}", file=sys.stderr, flush=True)
        return 2

    counts = Counter(o.outcome for o in outcomes)
    print(f"\n取り込み完了: ok={counts['ok']} empty={counts['empty']} "
          f"http_error={counts['http_error']} parse_error={counts['parse_error']}", flush=True)

    report = run_quality_checks(conn, scope_from=args.date_from, scope_to=args.date_to)
    print()
    print(report.to_markdown())
    return report.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
