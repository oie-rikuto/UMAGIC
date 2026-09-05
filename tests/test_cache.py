"""`002-loader.md` 単体テスト観点12、および `D-014` のガード。"""

from __future__ import annotations

import gzip
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest

from umagic.cache import MIN_INTERVAL_SEC, LocalCacheFetcher, RobotsDisallowed
from umagic.sources.base import RawPage


def test_min_interval_floor_enforced(tmp_path):
    with pytest.raises(ValueError):
        LocalCacheFetcher(cache_dir=tmp_path, user_agent="test", min_interval=1.0)


def test_12_cache_hit_skips_http(tmp_path, monkeypatch):
    f = LocalCacheFetcher(cache_dir=tmp_path, user_agent="test")

    calls = {"n": 0}

    def fake_fetch(self, url):
        calls["n"] += 1
        return b"<html>ok</html>", "utf-8", 200

    monkeypatch.setattr(LocalCacheFetcher, "_throttled_fetch", fake_fetch)
    monkeypatch.setattr(LocalCacheFetcher, "check_robots", lambda self, url: None)

    page1 = f.get("https://db.netkeiba.com/race/1/", source="netkeiba_jra",
                  page_kind="archive", source_key="1")
    assert page1.from_cache is False
    assert calls["n"] == 1

    page2 = f.get("https://db.netkeiba.com/race/1/", source="netkeiba_jra",
                  page_kind="archive", source_key="1")
    assert page2.from_cache is True
    assert calls["n"] == 1  # HTTPは発行されない
    assert page2.body == page1.body


def test_robots_disallowed_raises(tmp_path, monkeypatch):
    f = LocalCacheFetcher(cache_dir=tmp_path, user_agent="test")

    def deny(self, url):
        raise RobotsDisallowed("blocked")

    monkeypatch.setattr(LocalCacheFetcher, "check_robots", deny)
    with pytest.raises(RobotsDisallowed):
        f.get("https://db.netkeiba.com/race/1/", source="netkeiba_jra",
              page_kind="archive", source_key="1")


def test_shutuba_pages_are_never_cached(tmp_path, monkeypatch):
    """`D-199`: `shutuba`（発走前の出馬表）は無期限キャッシュの対象から
    外す。同じレースの発走までに登録発表→枠順確定→馬体重発表と内容が
    変わり続けるため、`archive`と同じ「一度取得したら不変」という前提が
    成り立たない。

    実際に本番で3日前のキャッシュを読み続け、枠順が確定した後も
    「未確定」と誤判定し続ける事故が起きた。
    """
    f = LocalCacheFetcher(cache_dir=tmp_path, user_agent="test")

    bodies = [b"<html>before draw</html>", b"<html>after draw</html>"]
    calls = {"n": 0}

    def fake_fetch(self, url):
        body = bodies[min(calls["n"], len(bodies) - 1)]
        calls["n"] += 1
        return body, "utf-8", 200

    monkeypatch.setattr(LocalCacheFetcher, "_throttled_fetch", fake_fetch)
    monkeypatch.setattr(LocalCacheFetcher, "check_robots", lambda self, url: None)

    page1 = f.get("https://race.netkeiba.com/race/shutuba.html?race_id=1",
                  source="netkeiba_jra", page_kind="shutuba", source_key="1")
    page2 = f.get("https://race.netkeiba.com/race/shutuba.html?race_id=1",
                  source="netkeiba_jra", page_kind="shutuba", source_key="1")

    assert page1.from_cache is False
    assert page2.from_cache is False  # 毎回ライブ取得——キャッシュヒットしない
    assert calls["n"] == 2  # HTTPが2回とも発行される
    assert page1.body == bodies[0]
    assert page2.body == bodies[1]  # 2回目は本当に新しい内容が返る
    assert list(tmp_path.glob("*.html.gz")) == []  # ディスクにも残さない


def test_bypass_cache_forces_live_fetch_regardless_of_page_kind(tmp_path, monkeypatch):
    """`D-204`: `bypass_cache=True`を指定した呼び出しは、`day_index`など
    本来キャッシュ対象のページ種別でも常にライブ取得し、キャッシュにも残さない。

    実際の事故: 直近日付の`day_index`を早い時点で取得した空の結果が
    無期限キャッシュに固定され、レース終了後に再取得しても同じ空の
    結果が返り続けた（`D-188`、および本件の再発）。
    """
    f = LocalCacheFetcher(cache_dir=tmp_path, user_agent="test")
    bodies = [b"<html>empty day</html>", b"<html>populated day</html>"]
    calls = {"n": 0}

    def fake_fetch(self, url):
        body = bodies[min(calls["n"], len(bodies) - 1)]
        calls["n"] += 1
        return body, "utf-8", 200

    monkeypatch.setattr(LocalCacheFetcher, "_throttled_fetch", fake_fetch)
    monkeypatch.setattr(LocalCacheFetcher, "check_robots", lambda self, url: None)

    url = "https://db.netkeiba.com/race/list/20260905/"
    page1 = f.get(url, source="netkeiba_jra", page_kind="day_index",
                  source_key="20260905", bypass_cache=True)
    page2 = f.get(url, source="netkeiba_jra", page_kind="day_index",
                  source_key="20260905", bypass_cache=True)

    assert page1.from_cache is False
    assert page2.from_cache is False
    assert calls["n"] == 2
    assert page1.body == bodies[0]
    assert page2.body == bodies[1]
    assert list(tmp_path.glob("*.html.gz")) == []


def test_bypass_cache_false_preserves_normal_day_index_caching(tmp_path, monkeypatch):
    """`bypass_cache`を指定しない（既定`False`）場合、`day_index`は従来
    どおりキャッシュされる——過去の確定済み日を毎回再取得しないため。"""
    f = LocalCacheFetcher(cache_dir=tmp_path, user_agent="test")
    calls = {"n": 0}

    def fake_fetch(self, url):
        calls["n"] += 1
        return b"<html>old day</html>", "utf-8", 200

    monkeypatch.setattr(LocalCacheFetcher, "_throttled_fetch", fake_fetch)
    monkeypatch.setattr(LocalCacheFetcher, "check_robots", lambda self, url: None)

    url = "https://db.netkeiba.com/race/list/20150101/"
    f.get(url, source="netkeiba_jra", page_kind="day_index", source_key="20150101")
    page2 = f.get(url, source="netkeiba_jra", page_kind="day_index", source_key="20150101")

    assert page2.from_cache is True
    assert calls["n"] == 1


def test_archive_pages_still_cache_alongside_shutuba(tmp_path, monkeypatch):
    """`D-199`: `shutuba`を無期限キャッシュから外しても、`archive`
    （結果確定済みの過去レース、不変）は従来どおりキャッシュされる。"""
    f = LocalCacheFetcher(cache_dir=tmp_path, user_agent="test")
    calls = {"n": 0}

    def fake_fetch(self, url):
        calls["n"] += 1
        return b"<html>archive</html>", "utf-8", 200

    monkeypatch.setattr(LocalCacheFetcher, "_throttled_fetch", fake_fetch)
    monkeypatch.setattr(LocalCacheFetcher, "check_robots", lambda self, url: None)

    f.get("https://db.netkeiba.com/race/1/", source="netkeiba_jra",
          page_kind="archive", source_key="1")
    page2 = f.get("https://db.netkeiba.com/race/1/", source="netkeiba_jra",
                  page_kind="archive", source_key="1")

    assert page2.from_cache is True
    assert calls["n"] == 1


def test_cache_body_survives_gzip_roundtrip(tmp_path, monkeypatch):
    f = LocalCacheFetcher(cache_dir=tmp_path, user_agent="test")
    body = "日本ダービー".encode("utf-8")
    monkeypatch.setattr(LocalCacheFetcher, "_throttled_fetch", lambda self, url: (body, "utf-8", 200))
    monkeypatch.setattr(LocalCacheFetcher, "check_robots", lambda self, url: None)
    page = f.get("https://db.netkeiba.com/race/1/", source="netkeiba_jra",
                 page_kind="archive", source_key="1")
    assert page.body == body
    # ディスク上は gzip 圧縮されている
    cache_files = list(tmp_path.glob("*.html.gz"))
    assert len(cache_files) == 1
    assert gzip.decompress(cache_files[0].read_bytes()) == body
