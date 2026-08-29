"""`docs/tasks.md` の冪等性検証（`R-021`）。同じキャッシュから2回取り込んでも
`fetched_at` を除いて完全一致することを確認する。"""

from __future__ import annotations

from datetime import datetime, timezone

from tests.fixtures.build_archive import build_archive_html
from umagic.loader import ingest_race
from umagic.sources.base import PageKind, RawPage
from umagic.sources.netkeiba import NetkeibaJraSource


class _FixedFetcher:
    """毎回同じ body を返す `Fetcher`。`from_cache` は2回目以降 True。"""

    def __init__(self, body: bytes):
        self.body = body
        self.calls = 0

    def get(self, url: str, *, source: str, page_kind: PageKind, source_key: str) -> RawPage:
        self.calls += 1
        return RawPage(source=source, page_kind=page_kind, source_key=source_key,
                       url=url, body=self.body, encoding="unknown",
                       fetched_at=datetime(2026, 8, 16, tzinfo=timezone.utc),
                       from_cache=self.calls > 1)


def _table_snapshot(conn, table: str, exclude_cols: set[str]) -> list[tuple]:
    cols = [r[0] for r in conn.execute(f"SELECT * FROM {table} LIMIT 0").description]
    keep = [c for c in cols if c not in exclude_cols]
    order = ", ".join(keep)
    return conn.execute(f"SELECT {order} FROM {table} ORDER BY {keep[0]}").fetchall()


def test_reingest_same_cache_is_idempotent(conn):
    html = build_archive_html(
        race_id=42, date_y=2023, date_m=5, date_d=28, corner_nos=[1, 2, 3, 4],
        runners=[
            {"finish": "1", "number": 1, "name": "馬A", "passage": "1-1-1-1",
             "odds": "2.1", "popularity": "1", "time": "1:08.0", "horse_key": "1000000001"},
            {"finish": "中", "number": 2, "name": "馬B", "passage": "", "horse_key": "1000000002"},
        ],
        payouts=[{"bet_type": "単勝", "combo": "1", "payout": 210, "popularity": 1}],
        laps=[12.0, 11.5, 12.3, 12.1],
    )
    fetcher = _FixedFetcher(html)
    source = NetkeibaJraSource(fetcher)

    out1 = ingest_race(conn, fetcher, source, "42")
    assert out1.outcome == "ok"
    n_races_after_1 = conn.execute("SELECT COUNT(*) FROM races").fetchone()[0]
    n_runners_after_1 = conn.execute("SELECT COUNT(*) FROM runners").fetchone()[0]

    snap_races_1 = _table_snapshot(conn, "races", {"fetched_at"})
    snap_runners_1 = _table_snapshot(conn, "runners", {"fetched_at"})
    snap_payouts_1 = _table_snapshot(conn, "payouts", {"fetched_at"})
    snap_laps_1 = _table_snapshot(conn, "laps", {"fetched_at"})
    n_fetch_log_1 = conn.execute("SELECT COUNT(*) FROM fetch_log").fetchone()[0]
    n_source_ids_1 = conn.execute("SELECT COUNT(*) FROM source_ids").fetchone()[0]

    out2 = ingest_race(conn, fetcher, source, "42")
    assert out2.outcome == "ok"

    assert conn.execute("SELECT COUNT(*) FROM races").fetchone()[0] == n_races_after_1
    assert conn.execute("SELECT COUNT(*) FROM runners").fetchone()[0] == n_runners_after_1
    assert _table_snapshot(conn, "races", {"fetched_at"}) == snap_races_1
    assert _table_snapshot(conn, "runners", {"fetched_at"}) == snap_runners_1
    assert _table_snapshot(conn, "payouts", {"fetched_at"}) == snap_payouts_1
    assert _table_snapshot(conn, "laps", {"fetched_at"}) == snap_laps_1

    # fetch_log は行を増やさない（D-045 の upsert）
    assert conn.execute("SELECT COUNT(*) FROM fetch_log").fetchone()[0] == n_fetch_log_1 == 1
    # source_ids も再解決のたびに増えない
    assert conn.execute("SELECT COUNT(*) FROM source_ids").fetchone()[0] == n_source_ids_1

    assert fetcher.calls == 2  # 2回とも呼ばれるが、2回目は from_cache=True


def test_reingest_rejected_rows_do_not_double(conn):
    """再取り込みで rejected_rows が積み上がらない（D-045）。"""
    html = build_archive_html(
        race_id=7, date_y=2023, date_m=1, date_d=1, corner_nos=[1, 2, 3, 4],
        runners=[{"finish": "?", "number": 1, "passage": ""}],  # 未知マーカー
    )
    fetcher = _FixedFetcher(html)
    source = NetkeibaJraSource(fetcher)
    ingest_race(conn, fetcher, source, "7")
    ingest_race(conn, fetcher, source, "7")
    n = conn.execute("SELECT COUNT(*) FROM rejected_rows").fetchone()[0]
    assert n == 1


def _owner_runner(**kw):
    base = dict(finish="1", frame=1, number=1, name="馬", passage="1-1-1-1")
    base.update(kw)
    return base


def test_owner_resolved_and_written(conn):
    """馬主（`D-165`）が `owners` に登録され `runners.owner_id` に書かれる。"""
    html = build_archive_html(
        race_id=8, date_y=2023, date_m=1, date_d=1, corner_nos=[1, 2, 3, 4],
        runners=[_owner_runner(owner_key="180800", owner_name="東京ホースレーシング")],
    )
    fetcher = _FixedFetcher(html)
    source = NetkeibaJraSource(fetcher)
    ingest_race(conn, fetcher, source, "8")

    row = conn.execute(
        "SELECT o.owner_id, o.name FROM runners ru JOIN owners o USING (owner_id) "
        "WHERE ru.race_id = 8"
    ).fetchone()
    assert row is not None
    assert row[1] == "東京ホースレーシング"

    # 同じ馬主が別レースに出ても行が増えない（`resolve()` の冪等性）
    html2 = build_archive_html(
        race_id=9, date_y=2023, date_m=1, date_d=2, corner_nos=[1, 2, 3, 4],
        runners=[_owner_runner(owner_key="180800", owner_name="東京ホースレーシング")],
    )
    fetcher2 = _FixedFetcher(html2)
    source2 = NetkeibaJraSource(fetcher2)
    ingest_race(conn, fetcher2, source2, "9")
    n_owners = conn.execute("SELECT COUNT(*) FROM owners").fetchone()[0]
    assert n_owners == 1
