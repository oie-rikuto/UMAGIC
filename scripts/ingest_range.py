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
from datetime import date, datetime, timezone
from pathlib import Path

import duckdb

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from umagic.cache import LocalCacheFetcher, RobotsDisallowed
from umagic.loader import ingest_race
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
        print(f"[init] 新規DB作成: {db_path}")
    else:
        print(f"[resume] 既存DBを使用: {db_path}")

    fetcher = LocalCacheFetcher(cache_dir=Path(args.cache_dir), user_agent=UA,
                               min_interval=args.sleep)
    source = NetkeibaJraSource(fetcher)

    day = args.date_from
    n_ok = n_empty = n_error = 0
    already_done = {r[0] for r in conn.execute(
        "SELECT source_key FROM fetch_log WHERE page_kind='archive' AND outcome='ok'"
    ).fetchall()}

    try:
        while day <= args.date_to:
            try:
                keys = source.list_race_keys(day)
            except RobotsDisallowed as e:
                print(f"[中断] {e}", file=sys.stderr)
                return 2
            for key in keys:
                if key in already_done:
                    continue
                out = ingest_race(conn, fetcher, source, key)
                if out.outcome == "ok":
                    n_ok += 1
                elif out.outcome == "empty":
                    n_empty += 1
                else:
                    n_error += 1
                    print(f"  [{out.outcome}] {key}: {out.detail}")
            if keys:
                print(f"[{day}] {len(keys)} races -> ok={n_ok} empty={n_empty} error={n_error}")
            day += __import__("datetime").timedelta(days=1)
    except RobotsDisallowed as e:
        print(f"[中断] {e}", file=sys.stderr)
        return 2

    print(f"\n取り込み完了: ok={n_ok} empty={n_empty} error={n_error}")

    report = run_quality_checks(conn, scope_from=args.date_from, scope_to=args.date_to)
    print()
    print(report.to_markdown())
    return report.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
