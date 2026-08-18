"""ローダーの orchestration（`docs/spec/002-loader.md`）。

`Fetcher` で取得し、`Source.parse` でパースし、中間スキーマ（`schema.py`）と
運用テーブル（`ops_schema.py`）に書き込む。`race_id` は netkeiba のIDを
そのまま使う（`001-schema.md` / `002-loader.md` の受け入れケースが一貫して
netkeiba の12桁IDを `races.race_id` として扱っているため）。horse / jockey /
trainer はソース間で番号体系が変わりうるため `source_ids` 経由で解決する
（`D-035`）。

書き込みは冪等にする（`R-021`）: 同じ `race_id` を2回書き込んでも、
`fetched_at` を除いて結果が一致するよう、既存行を削除してから挿入する。
"""

from __future__ import annotations

import urllib.error
from dataclasses import dataclass
from datetime import date, datetime, timezone

import duckdb

from umagic.cache import RobotsDisallowed
from umagic.ids import resolve
from umagic.ops_schema import record_fetch, replace_rejected_rows
from umagic.sources.base import Fetcher, ParsedRace, Source


@dataclass
class IngestOutcome:
    source_key: str
    outcome: str          # fetch_log.outcome と同じ語彙
    detail: str | None = None
    n_runners: int = 0
    n_rejected: int = 0


def _is_empty(parsed: ParsedRace) -> bool:
    """着順テーブルが無い（空テンプレート）かどうか。"""
    return not parsed.runners and parsed.race.get("course") is None


def _resolve_person(
    conn: duckdb.DuckDBPyConnection,
    entity_type: str,
    table: str,
    id_column: str,
    source: str,
    fetched_at,
    source_key: str | None,
    name: str | None,
) -> int | None:
    """騎手・調教師を同定し、`jockeys` / `trainers` に登録する（`D-057`）。

    名前が取れなかった場合は行を作らない。`name` が `NOT NULL` のため、
    空の行を作ると後から名前で引けない状態が固定される。
    """
    if not source_key:
        return None
    internal_id = resolve(conn, entity_type, source, source_key, fetched_at)
    if name and conn.execute(
        f"SELECT 1 FROM {table} WHERE {id_column} = ?", [internal_id],
    ).fetchone() is None:
        conn.execute(
            f"INSERT INTO {table} VALUES (?, ?, ?, ?)",
            [internal_id, name, source, fetched_at],
        )
    return internal_id


def _write_race(conn: duckdb.DuckDBPyConnection, source: str, parsed: ParsedRace) -> None:
    race = parsed.race
    race_id = race["race_id"]
    fetched_at = race["fetched_at"]

    # source_ids にも登録する（将来ソースが増えたときの解決点。D-035）。
    # races.race_id 自体は netkeiba の生IDをそのまま使う
    resolve(conn, "race", source, str(race_id), fetched_at)

    # 冪等性（R-021）: 既存行を削除してから挿入する
    conn.execute("DELETE FROM laps WHERE race_id = ?", [race_id])
    conn.execute("DELETE FROM payouts WHERE race_id = ?", [race_id])
    conn.execute("DELETE FROM runners WHERE race_id = ?", [race_id])
    conn.execute("DELETE FROM races WHERE race_id = ?", [race_id])

    conn.execute(
        """
        INSERT INTO races (race_id, date, course, race_number, post_time, distance,
            surface, direction, grade, track_condition, weather, weather_forecast,
            n_entries, n_starters, prize, corner_nos,
            race_class, weight_rule, meeting_no, meeting_day, source, fetched_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [race_id, race["date"], race["course"], race["race_number"], race["post_time"],
         race["distance"], race["surface"], race["direction"], race["grade"],
         race["track_condition"], race["weather"], race["weather_forecast"],
         race["n_entries"], race["n_starters"], race["prize"], race["corner_nos"],
         race.get("race_class"), race.get("weight_rule"),
         race.get("meeting_no"), race.get("meeting_day"),
         source, fetched_at],
    )

    for r in parsed.runners:
        horse_id = resolve(conn, "horse", source, r["horse_source_key"], fetched_at)
        if conn.execute("SELECT 1 FROM horses WHERE horse_id = ?", [horse_id]).fetchone() is None:
            conn.execute(
                "INSERT INTO horses VALUES (?, ?, NULL, NULL, NULL, NULL, ?, ?)",
                [horse_id, r["horse_name"], source, fetched_at],
            )
        jockey_id = _resolve_person(
            conn, "jockey", "jockeys", "jockey_id", source, fetched_at,
            r["jockey_source_key"], r.get("jockey_name"))
        trainer_id = _resolve_person(
            conn, "trainer", "trainers", "trainer_id", source, fetched_at,
            r["trainer_source_key"], r.get("trainer_name"))

        conn.execute(
            """
            INSERT INTO runners (race_id, horse_id, number, frame, jockey_id, trainer_id,
                weight_carried, horse_weight, weight_diff, age, sex, odds_win, popularity,
                status, finish_pos, margin, time_sec, last_3f, corners, affiliation,
                source, fetched_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [race_id, horse_id, r["number"], r["frame"], jockey_id, trainer_id,
             r["weight_carried"], r["horse_weight"], r["weight_diff"], r["age"], r["sex"],
             r["odds_win"], r["popularity"], r["status"], r["finish_pos"], r["margin"],
             r["time_sec"], r["last_3f"], r["corners"], r.get("affiliation"),
             source, r["fetched_at"]],
        )

    for p in parsed.payouts:
        conn.execute(
            "INSERT INTO payouts VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [p["race_id"], p["bet_type"], p["comb_key"], p["combination"], p["payout"],
             p["popularity"], source, p["fetched_at"]],
        )

    for lp in parsed.laps:
        conn.execute(
            "INSERT INTO laps VALUES (?, ?, ?, ?, ?)",
            [lp["race_id"], lp["furlong_no"], lp["lap_sec"], source, lp["fetched_at"]],
        )

    replace_rejected_rows(
        conn, source=source, source_key=str(race_id),
        rows=[{"row_ref": rj.row_ref, "reason": rj.reason, "raw": rj.raw,
               "fetched_at": fetched_at} for rj in parsed.rejected],
    )


def ingest_race(
    conn: duckdb.DuckDBPyConnection,
    fetcher: Fetcher,
    source: Source,
    source_key: str,
) -> IngestOutcome:
    """1レース分を取得・パース・書き込みする。`002-loader.md` の失敗時の挙動に従う。"""
    url = source.url_for(source_key, "archive")
    try:
        page = fetcher.get(url, source=source.name, page_kind="archive", source_key=source_key)
    except RobotsDisallowed:
        raise
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as e:
        record_fetch(conn, url=url, source=source.name, page_kind="archive",
                     source_key=source_key, http_status=None, outcome="http_error",
                     detail=str(e), fetched_at=datetime.now(timezone.utc))
        return IngestOutcome(source_key, "http_error", str(e))

    try:
        parsed = source.parse(page)
    except Exception as e:  # noqa: BLE001 — パース例外は http_error と同様に次へ進む
        record_fetch(conn, url=url, source=source.name, page_kind="archive",
                     source_key=source_key, http_status=200, outcome="parse_error",
                     detail=str(e), fetched_at=page.fetched_at)
        return IngestOutcome(source_key, "parse_error", str(e))

    if _is_empty(parsed):
        record_fetch(conn, url=url, source=source.name, page_kind="archive",
                     source_key=source_key, http_status=200, outcome="empty",
                     detail=None, fetched_at=page.fetched_at)
        return IngestOutcome(source_key, "empty")

    # 書き込みの制約違反を捕まえる。捕まえないと、ヘッダを読めないレースが
    # 1件あるだけで（`races.course` などが `NULL` になり `NOT NULL` 違反）
    # 数千レースの取り込みが落ちる。パースが不完全だったことの現れなので
    # `parse_error` に倒し、1レースを飛ばして次へ進む（`002-loader.md`）。
    #
    # **トランザクションで囲まない。** `_write_race` は子テーブルと `races` を
    # DELETE してから INSERT し直すが、DuckDB は同一トランザクション内だと
    # 削除済みの子行をまだ参照中とみなして外部キー違反を出すため、囲むと
    # 再取り込みが必ず失敗する。
    #
    # 結果として、INSERT の途中で落ちたレースは行が欠けた状態で残りうる。
    # ただし `fetch_log` に `parse_error` が入るので `outcome='ok'` を条件と
    # する再開（`completed_race_keys`）の対象外となり、次回の実行で取り直される。
    try:
        _write_race(conn, source.name, parsed)
    except Exception as e:  # noqa: BLE001
        record_fetch(conn, url=url, source=source.name, page_kind="archive",
                     source_key=source_key, http_status=200, outcome="parse_error",
                     detail=f"write failed: {e}", fetched_at=page.fetched_at)
        return IngestOutcome(source_key, "parse_error", f"write failed: {e}")

    record_fetch(conn, url=url, source=source.name, page_kind="archive",
                 source_key=source_key, http_status=200, outcome="ok",
                 detail=None, fetched_at=page.fetched_at)
    return IngestOutcome(source_key, "ok", n_runners=len(parsed.runners),
                         n_rejected=len(parsed.rejected))


def list_day_races(
    conn: duckdb.DuckDBPyConnection,
    source: Source,
    day: date,
) -> tuple[list[str], IngestOutcome]:
    """1日分の `race_id` を列挙し、`day_index` の取得結果を `fetch_log` に記録する。

    `002-loader.md` は `fetch_log.page_kind` に `day_index` を記録することを
    求めている。記録しないと `fetch_incomplete`（`012-data-quality.md`）の
    分母から日次インデックスが丸ごと抜け、**その日のレースを1件も取り込め
    なかったこと自体が検知できない**。

    `archive` と同じく、`robots.txt` 以外の失敗で全体を止めない
    （`002-loader.md` の失敗時の挙動）。日次インデックスの一過性の
    HTTPエラーで数千レースの取り込みが落ちるのを防ぐ。
    """
    key = day.strftime("%Y%m%d")
    url = source.url_for(key, "day_index")
    try:
        race_keys = source.list_race_keys(day)
    except RobotsDisallowed:
        raise
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as e:
        record_fetch(conn, url=url, source=source.name, page_kind="day_index",
                     source_key=key, http_status=None, outcome="http_error",
                     detail=str(e), fetched_at=datetime.now(timezone.utc))
        return [], IngestOutcome(key, "http_error", str(e))
    except Exception as e:  # noqa: BLE001 — パース例外も次の日へ進む
        record_fetch(conn, url=url, source=source.name, page_kind="day_index",
                     source_key=key, http_status=200, outcome="parse_error",
                     detail=str(e), fetched_at=datetime.now(timezone.utc))
        return [], IngestOutcome(key, "parse_error", str(e))

    record_fetch(conn, url=url, source=source.name, page_kind="day_index",
                 source_key=key, http_status=200,
                 outcome="ok" if race_keys else "empty",
                 detail=None, fetched_at=datetime.now(timezone.utc))
    return race_keys, IngestOutcome(key, "ok" if race_keys else "empty")


def completed_race_keys(conn: duckdb.DuckDBPyConnection) -> set[str]:
    """すでに取り込み済み（`outcome='ok'`）のレースキー。再開時のスキップに使う。"""
    return {r[0] for r in conn.execute(
        "SELECT source_key FROM fetch_log WHERE page_kind = 'archive' AND outcome = 'ok'"
    ).fetchall()}


def ingest_range(
    conn: duckdb.DuckDBPyConnection,
    fetcher: Fetcher,
    source: Source,
    start: date,
    end: date,
    *,
    resume: bool = True,
    on_day=None,
    on_race_error=None,
) -> list[IngestOutcome]:
    """`start`〜`end`（`date`、両端含む）の日付範囲を取り込む。

    `robots.txt` が取得を禁止した場合のみ `RobotsDisallowed` がそのまま
    伝播し、全体を中断する（`D-014` 条件3）。それ以外の失敗は、レースも
    日次インデックスも1件飛ばして続ける（`002-loader.md` の失敗時の挙動）。

    `resume=True` のとき、`fetch_log` に `outcome='ok'` で残っている
    レースを再取得しない。長時間の取り込みを中断・再開できるようにする。
    """
    import datetime as _dt

    outcomes: list[IngestOutcome] = []
    done = completed_race_keys(conn) if resume else set()

    day = start
    while day <= end:
        race_keys, day_outcome = list_day_races(conn, source, day)
        # `empty` は「その日にJRA中央開催が無い」であり平日は大半がこれ。
        # 異常ではないので通知しない（`fetch_log` には記録済み）
        if day_outcome.outcome in ("http_error", "parse_error"):
            outcomes.append(day_outcome)
            if on_race_error:
                on_race_error(day_outcome)

        for key in race_keys:
            if key in done:
                continue
            out = ingest_race(conn, fetcher, source, key)
            outcomes.append(out)
            if out.outcome != "ok" and on_race_error:
                on_race_error(out)

        if on_day:
            on_day(day, race_keys, outcomes)
        day += _dt.timedelta(days=1)
    return outcomes
