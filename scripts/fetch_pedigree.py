#!/usr/bin/env python3
"""血統の取得（`D-050`）。

`horses` に行がある馬について `horse_ped` ページを引き、
`sire_id` / `dam_id` / `damsire_id` を埋める。

`D-014` 条件2 により1リクエスト5秒。2〜3年分の約21,000頭で約29時間かかる。
中断しても `fetch_log` から再開できる（`outcome='ok'` の馬を引き直さない）。

使い方:
    uv run python scripts/fetch_pedigree.py --db data/umagic.duckdb
"""

from __future__ import annotations

import argparse
import sys
import urllib.error
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import duckdb

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from umagic.cache import LocalCacheFetcher, RobotsDisallowed
from umagic.ids import resolve
from umagic.ops_schema import record_fetch
from umagic.sources.netkeiba import SOURCE, parse_pedigree, url_for

UA = "UMAGIC-dev/0.1 (personal research; contact: repository owner)"


def pending_horses(conn: duckdb.DuckDBPyConnection) -> list[tuple[int, str]]:
    """血統が未取得の馬を (horse_id, source_key) で返す。

    `fetch_log` に `page_kind='horse_ped'` かつ `outcome='ok'` で残っている馬は
    再取得しない。血統3列が `NULL` のままでも、ページを引いた結果として
    そうなった馬（`pedigree_unparsed`）を引き直さないため。
    """
    return conn.execute(
        """
        SELECT h.horse_id, si.source_key
        FROM horses h
        JOIN source_ids si
          ON si.entity_type = 'horse' AND si.internal_id = h.horse_id
         AND si.source = h.source
        WHERE NOT EXISTS (
            SELECT 1 FROM fetch_log fl
            WHERE fl.page_kind = 'horse_ped'
              AND fl.source_key = si.source_key
              AND fl.outcome = 'ok'
        )
        ORDER BY h.horse_id
        """
    ).fetchall()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", default="data/umagic.duckdb")
    ap.add_argument("--cache-dir", default="data/cache")
    ap.add_argument("--sleep", type=float, default=5.0,
                    help="D-014 条件2。5.0未満は指定できない")
    ap.add_argument("--limit", type=int, default=None, help="動作確認用の件数上限")
    args = ap.parse_args()

    conn = duckdb.connect(args.db)
    targets = pending_horses(conn)
    if args.limit:
        targets = targets[:args.limit]
    print(f"[開始] 未取得 {len(targets)} 頭", flush=True)

    fetcher = LocalCacheFetcher(cache_dir=Path(args.cache_dir), user_agent=UA,
                               min_interval=args.sleep)
    counts: Counter[str] = Counter()

    for i, (horse_id, key) in enumerate(targets, start=1):
        url = url_for(key, "horse_ped")
        try:
            page = fetcher.get(url, source=SOURCE, page_kind="horse_ped", source_key=key)
        except RobotsDisallowed as e:
            print(f"[中断] {e}", file=sys.stderr, flush=True)
            return 2
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as e:
            record_fetch(conn, url=url, source=SOURCE, page_kind="horse_ped",
                         source_key=key, http_status=None, outcome="http_error",
                         detail=str(e), fetched_at=datetime.now(timezone.utc))
            counts["http_error"] += 1
            print(f"  [http_error] {key}: {e}", flush=True)
            continue

        try:
            ped = parse_pedigree(page)
        except Exception as e:  # noqa: BLE001
            record_fetch(conn, url=url, source=SOURCE, page_kind="horse_ped",
                         source_key=key, http_status=200, outcome="parse_error",
                         detail=str(e), fetched_at=page.fetched_at)
            counts["parse_error"] += 1
            continue

        ids = {}
        for role, k in (("sire", ped["sire_key"]), ("dam", ped["dam_key"]),
                        ("damsire", ped["damsire_key"])):
            ids[role] = resolve(conn, "horse", SOURCE, k, page.fetched_at) if k else None

        conn.execute(
            "UPDATE horses SET sire_id = ?, dam_id = ?, damsire_id = ? WHERE horse_id = ?",
            [ids["sire"], ids["dam"], ids["damsire"], horse_id],
        )

        if ped["sire_key"] is None:
            # ページは引けたが血統表を解釈できなかった。D-050 により推測で補わない
            conn.execute(
                "INSERT INTO rejected_rows VALUES (?, ?, NULL, 'pedigree_unparsed', NULL, ?)",
                [SOURCE, key, page.fetched_at],
            )
            counts["unparsed"] += 1
        else:
            counts["ok"] += 1

        record_fetch(conn, url=url, source=SOURCE, page_kind="horse_ped",
                     source_key=key, http_status=200, outcome="ok",
                     detail=None, fetched_at=page.fetched_at)

        if i % 200 == 0:
            print(f"[{i}/{len(targets)}] ok={counts['ok']} unparsed={counts['unparsed']} "
                  f"error={counts['http_error'] + counts['parse_error']}", flush=True)

    print(f"\n完了: ok={counts['ok']} unparsed={counts['unparsed']} "
          f"http_error={counts['http_error']} parse_error={counts['parse_error']}", flush=True)

    filled = conn.execute("SELECT COUNT(sire_id), COUNT(*) FROM horses").fetchone()
    print(f"血統が埋まった馬: {filled[0]} / {filled[1]}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
