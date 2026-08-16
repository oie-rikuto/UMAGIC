"""エンティティ同定（`docs/spec/002-loader.md` / `D-035` / `D-038`）。

名寄せを行わない。同じ実体が別ソースで別の内部IDを持つ状態を許容する（`Q-024`）。
"""

from __future__ import annotations

from datetime import datetime

import duckdb


def next_internal_id(conn: duckdb.DuckDBPyConnection, entity_type: str) -> int:
    row = conn.execute(
        "SELECT COALESCE(MAX(internal_id), 0) + 1 FROM source_ids WHERE entity_type = ?",
        [entity_type],
    ).fetchone()
    return row[0]


def resolve(
    conn: duckdb.DuckDBPyConnection,
    entity_type: str,
    source: str,
    source_key: str,
    fetched_at: datetime,
) -> int:
    row = conn.execute(
        "SELECT internal_id FROM source_ids "
        "WHERE entity_type = ? AND source = ? AND source_key = ?",
        [entity_type, source, source_key],
    ).fetchone()
    if row is not None:
        return row[0]

    new_id = next_internal_id(conn, entity_type)
    conn.execute(
        "INSERT INTO source_ids VALUES (?, ?, ?, ?, ?)",
        [entity_type, new_id, source, source_key, fetched_at],
    )
    return new_id
