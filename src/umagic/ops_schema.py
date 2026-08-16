"""運用テーブル（`docs/spec/002-loader.md` / `docs/spec/012-data-quality.md`）。

中間スキーマ（`schema.py`）はソース非依存でなければならない（`D-009`）ため、
`url` や `page_kind` を持つテーブルはここに分ける。`source` / `fetched_at` を
持たないのは `R-012` の対象外だから（`D-046`）。
"""

from __future__ import annotations

import duckdb

DDL_STATEMENTS: list[str] = [
    """
    CREATE TABLE fetch_log (
        url          VARCHAR   PRIMARY KEY,
        source       VARCHAR   NOT NULL,
        page_kind    VARCHAR   NOT NULL,
        source_key   VARCHAR   NOT NULL,
        http_status  INTEGER,
        outcome      VARCHAR   NOT NULL,
        detail       VARCHAR,
        fetched_at   TIMESTAMP NOT NULL,
        CHECK (outcome IN ('ok', 'empty', 'http_error', 'parse_error')),
        CHECK (page_kind IN ('day_index', 'archive', 'shutuba'))
    )
    """,
    """
    CREATE TABLE rejected_rows (
        source       VARCHAR   NOT NULL,
        source_key   VARCHAR   NOT NULL,
        row_ref      VARCHAR,
        reason       VARCHAR   NOT NULL,
        raw          VARCHAR,
        fetched_at   TIMESTAMP NOT NULL
    )
    """,
    """
    CREATE SEQUENCE quality_run_id_seq START 1
    """,
    """
    CREATE TABLE quality_runs (
        run_id      BIGINT    PRIMARY KEY,
        started_at  TIMESTAMP NOT NULL,
        scope_from  DATE,
        scope_to    DATE,
        n_races     BIGINT    NOT NULL,
        n_runners   BIGINT    NOT NULL
    )
    """,
    """
    CREATE TABLE quality_findings (
        run_id     BIGINT    NOT NULL,
        check_id   VARCHAR   NOT NULL,
        severity   VARCHAR   NOT NULL,
        race_id    BIGINT,
        horse_id   BIGINT,
        detail     VARCHAR,
        CHECK (severity IN ('fail', 'warn'))
    )
    """,
]


def create_ops_schema(conn: duckdb.DuckDBPyConnection) -> None:
    for stmt in DDL_STATEMENTS:
        conn.execute(stmt)


# --- fetch_log: D-045 の upsert ---

_FETCH_LOG_UPSERT = """
INSERT INTO fetch_log (url, source, page_kind, source_key, http_status, outcome, detail, fetched_at)
VALUES (?, ?, ?, ?, ?, ?, ?, ?)
ON CONFLICT (url) DO UPDATE SET
    http_status = excluded.http_status,
    outcome     = excluded.outcome,
    detail      = excluded.detail,
    fetched_at  = excluded.fetched_at
"""


def record_fetch(
    conn: duckdb.DuckDBPyConnection,
    *,
    url: str,
    source: str,
    page_kind: str,
    source_key: str,
    http_status: int | None,
    outcome: str,
    detail: str | None,
    fetched_at,
) -> None:
    """`fetch_log` に1行を記録する。同じ URL の2回目は上書きする（`D-045`）。"""
    conn.execute(
        _FETCH_LOG_UPSERT,
        [url, source, page_kind, source_key, http_status, outcome, detail, fetched_at],
    )


def replace_rejected_rows(
    conn: duckdb.DuckDBPyConnection,
    *,
    source: str,
    source_key: str,
    rows: list[dict],
) -> None:
    """`source_key` 単位で `rejected_rows` を全削除してから挿入する。

    追記のみにすると再実行のたびに件数が倍になり、`rejected_rate` が
    実態から離れる（`D-045` と同じ理由）。
    """
    conn.execute(
        "DELETE FROM rejected_rows WHERE source = ? AND source_key = ?",
        [source, source_key],
    )
    for row in rows:
        conn.execute(
            "INSERT INTO rejected_rows VALUES (?, ?, ?, ?, ?, ?)",
            [source, source_key, row.get("row_ref"), row["reason"],
             row.get("raw"), row["fetched_at"]],
        )
