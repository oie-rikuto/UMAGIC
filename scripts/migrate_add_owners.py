"""`D-165`: 既存DBに馬主（`owners`）を追加するマイグレーション。

DuckDB は `ALTER TABLE ... DROP CONSTRAINT` を実装していない（1.5.5で確認済み）。
`source_ids.entity_type` の `CHECK` に `'owner'` を加えるには、テーブルを
作り直す必要がある。

手順:
  1. `source_ids` を CHECK 拡張版で作り直す（データを一時テーブル経由で退避）
  2. `owners` テーブルを新設する（`jockeys`/`trainers` と同形）
  3. `runners` に `owner_id BIGINT` 列を追加する
  4. `races` を再取り込みし、キャッシュ済みHTML（`data/cache/`）から
     馬主を再パースして書き込む——ネットワークアクセスは発生しない
     （`LocalCacheFetcher` はキャッシュヒット時にHTTPを発行しない、`D-014`）

**冪等**: `_write_race` は既存行を削除してから挿入するため、同じ `race_id`
を再実行しても安全（`R-021`）。`horses` の行は既存があれば上書きしないため、
`fetch_pedigree.py` が埋めた `sire_id`/`dam_id`/`damsire_id` は失われない。

使い方:
    uv run python scripts/migrate_add_owners.py <db_path> --cache-dir data/cache
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import duckdb

from umagic.cache import LocalCacheFetcher
from umagic.loader import _write_race
from umagic.sources.netkeiba import NetkeibaJraSource

UA = "umagic-research/0.1 (personal research; contact: see docs/decisions.md D-014)"


def migrate_schema(conn: duckdb.DuckDBPyConnection) -> None:
    """`source_ids` の CHECK 拡張、`owners` 新設、`runners.owner_id` 追加。"""
    has_owner_id = any(
        r[0] == "owner_id"
        for r in conn.execute("DESCRIBE runners").fetchall()
    )
    if has_owner_id:
        print("[migrate_schema] 既に適用済み。スキップする", flush=True)
        return

    print("[migrate_schema] source_ids を CHECK 拡張版に作り直す", flush=True)
    conn.execute("""
        CREATE TABLE source_ids_new (
            entity_type  VARCHAR   NOT NULL,
            internal_id  BIGINT    NOT NULL,
            source       VARCHAR   NOT NULL,
            source_key   VARCHAR   NOT NULL,
            fetched_at   TIMESTAMP NOT NULL,
            PRIMARY KEY (entity_type, source, source_key),
            CHECK (entity_type IN ('race', 'horse', 'jockey', 'trainer', 'owner'))
        )
    """)
    conn.execute("INSERT INTO source_ids_new SELECT * FROM source_ids")
    n_old = conn.execute("SELECT COUNT(*) FROM source_ids").fetchone()[0]
    n_new = conn.execute("SELECT COUNT(*) FROM source_ids_new").fetchone()[0]
    assert n_old == n_new, f"source_ids 退避で行数が変わった: {n_old} != {n_new}"
    conn.execute("DROP TABLE source_ids")
    conn.execute("ALTER TABLE source_ids_new RENAME TO source_ids")
    conn.execute(
        "CREATE UNIQUE INDEX ux_source_ids_internal ON source_ids "
        "(entity_type, internal_id, source)"
    )

    print("[migrate_schema] owners テーブルを新設する", flush=True)
    conn.execute("""
        CREATE TABLE owners (
            owner_id    BIGINT    PRIMARY KEY,
            name        VARCHAR   NOT NULL,
            source      VARCHAR   NOT NULL,
            fetched_at  TIMESTAMP NOT NULL
        )
    """)

    print("[migrate_schema] runners.owner_id を追加する", flush=True)
    conn.execute("ALTER TABLE runners ADD COLUMN owner_id BIGINT")
    print("[migrate_schema] 完了", flush=True)


def backfill_owners(conn: duckdb.DuckDBPyConnection, cache_dir: Path) -> None:
    """全レースをキャッシュから再取り込みし、馬主を書き込む（ネットワーク無し）。"""
    race_ids = [r[0] for r in conn.execute(
        "SELECT race_id FROM races ORDER BY race_id"
    ).fetchall()]
    print(f"[backfill_owners] 対象 {len(race_ids)} レース", flush=True)

    fetcher = LocalCacheFetcher(cache_dir=cache_dir, user_agent=UA)
    source = NetkeibaJraSource(fetcher)

    n_ok = n_err = n_owner_missing = 0
    t0 = time.monotonic()
    for i, rid in enumerate(race_ids, start=1):
        url = source.url_for(str(rid), "archive")
        try:
            page = fetcher.get(url, source="netkeiba_jra", page_kind="archive",
                               source_key=str(rid))
        except Exception as e:  # noqa: BLE001 — 取得失敗はログして続行
            print(f"  [WARN] race_id={rid} fetch失敗: {e}", flush=True)
            n_err += 1
            continue
        if not page.from_cache:
            print(f"  [WARN] race_id={rid} はキャッシュに無く、ネットワーク取得した", flush=True)

        parsed = source.parse(page)
        if not parsed.runners:
            n_err += 1
            continue
        _write_race(conn, "netkeiba_jra", parsed)
        n_ok += 1
        if all(r.get("owner_source_key") is None for r in parsed.runners):
            n_owner_missing += 1

        if i % 2000 == 0:
            el = time.monotonic() - t0
            print(f"  [{i}/{len(race_ids)}] {el:.1f}s 経過", flush=True)

    el = time.monotonic() - t0
    print(f"[backfill_owners] 完了 ok={n_ok} err={n_err} "
          f"馬主0件のレース={n_owner_missing} ({el:.1f}s)", flush=True)


def verify(conn: duckdb.DuckDBPyConnection) -> None:
    n_runners = conn.execute("SELECT COUNT(*) FROM runners").fetchone()[0]
    n_with_owner = conn.execute(
        "SELECT COUNT(*) FROM runners WHERE owner_id IS NOT NULL"
    ).fetchone()[0]
    n_owners = conn.execute("SELECT COUNT(*) FROM owners").fetchone()[0]
    n_races = conn.execute("SELECT COUNT(*) FROM races").fetchone()[0]
    print(f"[verify] runners={n_runners} owner_id充足={n_with_owner} "
          f"({n_with_owner / n_runners:.4f}) owners={n_owners} races={n_races}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("db_path")
    ap.add_argument("--cache-dir", default="data/cache")
    args = ap.parse_args()

    conn = duckdb.connect(args.db_path)
    try:
        migrate_schema(conn)
        backfill_owners(conn, Path(args.cache_dir))
        verify(conn)
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
