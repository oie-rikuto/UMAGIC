"""`ingest_range` の耐障害性と再開（長時間の取り込みを守る）。

日次インデックスの一過性のHTTPエラーで数千レースの取り込みが落ちないこと、
中断後に再開できること、`day_index` が `fetch_log` に残ること（`002-loader.md`）。
"""

from __future__ import annotations

import urllib.error
from datetime import date, datetime, timezone

import pytest

from tests.fixtures.build_archive import build_archive_html
from umagic.cache import RobotsDisallowed
from umagic.loader import completed_race_keys, ingest_range
from umagic.sources.base import PageKind, RawPage

FETCHED_AT = datetime(2026, 8, 16, tzinfo=timezone.utc)


class _StubFetcher:
    def __init__(self, body: bytes):
        self.body = body
        self.archive_calls = 0

    def get(self, url: str, *, source: str, page_kind: PageKind, source_key: str) -> RawPage:
        if page_kind == "archive":
            self.archive_calls += 1
        return RawPage(source=source, page_kind=page_kind, source_key=source_key,
                       url=url, body=self.body, encoding="unknown",
                       fetched_at=FETCHED_AT, from_cache=False)


class _StubSource:
    """1日あたり指定のレースキーを返す `Source`。日ごとに例外を仕込める。"""

    name = "netkeiba_jra"

    def __init__(self, fetcher, per_day: dict[date, list[str]], raise_on: dict = None):
        self._fetcher = fetcher
        self._per_day = per_day
        self._raise_on = raise_on or {}
        self.list_calls: list[date] = []

    def list_race_keys(self, day: date) -> list[str]:
        self.list_calls.append(day)
        if day in self._raise_on:
            raise self._raise_on[day]
        return self._per_day.get(day, [])

    def url_for(self, source_key: str, page_kind: PageKind) -> str:
        return f"https://db.netkeiba.com/{page_kind}/{source_key}/"

    def parse(self, page: RawPage):
        from umagic.sources.netkeiba import parse_archive
        return parse_archive(page)


def _html(race_id: int, day: int):
    # HTML 内の race_id は source_key と一致させる。_parse_header は
    # `/race/{source_key}/` の active リンクから course を取るため、
    # ずれていると course が NULL になる
    return build_archive_html(
        race_id=race_id, date_y=2023, date_m=5, date_d=day, race_number=race_id % 12 + 1,
        corner_nos=[1, 2, 3, 4],
        runners=[{"finish": "1", "number": 1, "passage": "1-1-1-1", "time": "1:08.0"}],
    )


def test_day_index_http_error_does_not_abort_range(conn):
    """一過性のHTTPエラーで全体が落ちない。翌日以降の取り込みが続く。"""
    d1, d2 = date(2023, 5, 27), date(2023, 5, 28)
    fetcher = _StubFetcher(_html(202305021211, 28))
    source = _StubSource(
        fetcher, per_day={d2: ["202305021211"]},
        raise_on={d1: urllib.error.URLError("temporary failure")},
    )

    outcomes = ingest_range(conn, fetcher, source, d1, d2)

    # 27日は http_error として記録され、28日は通常どおり取り込まれる
    assert any(o.outcome == "http_error" and o.source_key == "20230527" for o in outcomes)
    assert any(o.outcome == "ok" and o.source_key == "202305021211" for o in outcomes)
    assert conn.execute("SELECT COUNT(*) FROM races").fetchone()[0] == 1


def test_day_index_recorded_in_fetch_log(conn):
    """`002-loader.md`: page_kind に day_index を記録する。"""
    d = date(2023, 5, 28)
    fetcher = _StubFetcher(_html(202305021211, 28))
    source = _StubSource(fetcher, per_day={d: ["202305021211"]})

    ingest_range(conn, fetcher, source, d, d)

    kinds = dict(conn.execute(
        "SELECT page_kind, COUNT(*) FROM fetch_log GROUP BY page_kind"
    ).fetchall())
    assert kinds["day_index"] == 1
    assert kinds["archive"] == 1


def test_day_with_no_races_recorded_as_empty_not_error(conn):
    """開催の無い日は empty として記録され、エラー通知はされない。"""
    d = date(2023, 5, 29)  # 月曜、開催なし
    fetcher = _StubFetcher(b"")
    source = _StubSource(fetcher, per_day={})

    errors = []
    outcomes = ingest_range(conn, fetcher, source, d, d, on_race_error=errors.append)

    assert errors == []
    assert outcomes == []
    row = conn.execute(
        "SELECT outcome FROM fetch_log WHERE page_kind='day_index'"
    ).fetchone()
    assert row[0] == "empty"


def test_robots_disallowed_still_aborts(conn):
    """D-014 条件3 だけは全体を中断する。"""
    d = date(2023, 5, 28)
    fetcher = _StubFetcher(_html(202305021211, 28))
    source = _StubSource(fetcher, per_day={}, raise_on={d: RobotsDisallowed("blocked")})

    with pytest.raises(RobotsDisallowed):
        ingest_range(conn, fetcher, source, d, d)


def test_resume_skips_completed_races(conn):
    """中断・再開: 取り込み済みのレースはHTTPを発行し直さない。"""
    d = date(2023, 5, 28)
    fetcher = _StubFetcher(_html(202305021211, 28))
    source = _StubSource(fetcher, per_day={d: ["202305021211"]})

    ingest_range(conn, fetcher, source, d, d)
    assert fetcher.archive_calls == 1
    assert completed_race_keys(conn) == {"202305021211"}

    # 2回目: resume=True なので archive は引き直さない
    ingest_range(conn, fetcher, source, d, d, resume=True)
    assert fetcher.archive_calls == 1

    # resume=False なら引き直す
    ingest_range(conn, fetcher, source, d, d, resume=False)
    assert fetcher.archive_calls == 2


def test_unwritable_race_does_not_abort_range(conn):
    """ヘッダを読めず制約違反になるレースがあっても全体は止まらず、次へ進む。

    `_parse_header` は `/race/{source_key}/` の active リンクから course を取る。
    HTML 内の race_id がずれていると、一次情報源（active リンク）からは course
    が取れない。`Q-047` 段階②（`D-176`）で smalltxt の開催表記からのフォール
    バックを足したため、smalltxt 自体も壊す（`course=""` だと smalltxt の
    「n回m日目」に会場名が挟まらず `_MEETING_RE` も一致しなくなる）ことで
    両方の経路を落とし、course が NULL のまま `NOT NULL` 違反させる。
    実ページでもレイアウト変更で同じことが起きうる。
    """
    d1, d2 = date(2023, 5, 27), date(2023, 5, 28)
    # 1日目は source_key と HTML 内 race_id が食い違い、かつ course が空の壊れたページ
    broken = build_archive_html(
        race_id=999999999999, date_y=2023, date_m=5, date_d=27,
        race_number=999999999999 % 12 + 1, course="",
        corner_nos=[1, 2, 3, 4],
        runners=[{"finish": "1", "number": 1, "passage": "1-1-1-1", "time": "1:08.0"}],
    )
    good = _html(202305021211, 28)

    class _PerKeyFetcher(_StubFetcher):
        def get(self, url, *, source, page_kind, source_key):
            self.body = good if source_key == "202305021211" else broken
            return super().get(url, source=source, page_kind=page_kind, source_key=source_key)

    fetcher = _PerKeyFetcher(broken)
    source = _StubSource(fetcher, per_day={d1: ["202305021201"], d2: ["202305021211"]})

    outcomes = ingest_range(conn, fetcher, source, d1, d2)

    assert any(o.outcome == "parse_error" and o.source_key == "202305021201" for o in outcomes)
    assert any(o.outcome == "ok" and o.source_key == "202305021211" for o in outcomes)
    # 壊れたレースは ok ではないので、再開時に取り直される
    assert completed_race_keys(conn) == {"202305021211"}


def test_resume_retries_failed_races(conn):
    """失敗したレースは再開時に再取得される（ok のみスキップ対象）。"""
    d = date(2023, 5, 28)
    fetcher = _StubFetcher(b"<html><body>empty template</body></html>")
    source = _StubSource(fetcher, per_day={d: ["202305021211"]})

    outcomes = ingest_range(conn, fetcher, source, d, d)
    assert outcomes[0].outcome == "empty"
    assert completed_race_keys(conn) == set()

    fetcher.body = _html(202305021211, 28)
    outcomes = ingest_range(conn, fetcher, source, d, d, resume=True)
    assert outcomes[0].outcome == "ok"
    assert fetcher.archive_calls == 2
