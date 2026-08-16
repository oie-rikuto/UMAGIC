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
from datetime import datetime, timezone

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
            n_entries, n_starters, prize, corner_nos, source, fetched_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [race_id, race["date"], race["course"], race["race_number"], race["post_time"],
         race["distance"], race["surface"], race["direction"], race["grade"],
         race["track_condition"], race["weather"], race["weather_forecast"],
         race["n_entries"], race["n_starters"], race["prize"], race["corner_nos"],
         source, fetched_at],
    )

    for r in parsed.runners:
        horse_id = resolve(conn, "horse", source, r["horse_source_key"], fetched_at)
        if conn.execute("SELECT 1 FROM horses WHERE horse_id = ?", [horse_id]).fetchone() is None:
            conn.execute(
                "INSERT INTO horses VALUES (?, ?, NULL, NULL, NULL, NULL, ?, ?)",
                [horse_id, r["horse_name"], source, fetched_at],
            )
        jockey_id = (resolve(conn, "jockey", source, r["jockey_source_key"], fetched_at)
                    if r["jockey_source_key"] else None)
        trainer_id = (resolve(conn, "trainer", source, r["trainer_source_key"], fetched_at)
                     if r["trainer_source_key"] else None)

        conn.execute(
            """
            INSERT INTO runners (race_id, horse_id, number, frame, jockey_id, trainer_id,
                weight_carried, horse_weight, weight_diff, age, sex, odds_win, popularity,
                status, finish_pos, margin, time_sec, last_3f, corners, source, fetched_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [race_id, horse_id, r["number"], r["frame"], jockey_id, trainer_id,
             r["weight_carried"], r["horse_weight"], r["weight_diff"], r["age"], r["sex"],
             r["odds_win"], r["popularity"], r["status"], r["finish_pos"], r["margin"],
             r["time_sec"], r["last_3f"], r["corners"], source, r["fetched_at"]],
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

    _write_race(conn, source.name, parsed)
    record_fetch(conn, url=url, source=source.name, page_kind="archive",
                 source_key=source_key, http_status=200, outcome="ok",
                 detail=None, fetched_at=page.fetched_at)
    return IngestOutcome(source_key, "ok", n_runners=len(parsed.runners),
                         n_rejected=len(parsed.rejected))


def ingest_range(
    conn: duckdb.DuckDBPyConnection,
    fetcher: Fetcher,
    source: Source,
    start,
    end,
) -> list[IngestOutcome]:
    """`start`〜`end`（`date`、両端含む）の日付範囲を取り込む。

    `robots.txt` が取得を禁止した場合は `RobotsDisallowed` がそのまま
    伝播し、全体を中断する（`D-014` 条件3）。それ以外の失敗はレースを
    1件飛ばして続ける（`002-loader.md` の失敗時の挙動）。
    """
    import datetime as _dt

    outcomes: list[IngestOutcome] = []
    day = start
    while day <= end:
        race_keys = source.list_race_keys(day)
        for key in race_keys:
            outcomes.append(ingest_race(conn, fetcher, source, key))
        day += _dt.timedelta(days=1)
    return outcomes
