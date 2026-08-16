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
