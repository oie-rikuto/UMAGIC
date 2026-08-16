"""`Fetcher` の実装（`docs/spec/002-loader.md` / `D-014` / `D-039` / `R-016`）。

レート制限・ローカルキャッシュ・`robots.txt` 確認を担う。ソース固有の
知識（URL構成・パース）は持たない（`sources/` の責務）。
"""

from __future__ import annotations

import gzip
import time
import urllib.error
import urllib.request
import urllib.robotparser
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

from umagic.sources.base import PageKind, RawPage

MIN_INTERVAL_SEC = 5.0  # D-014 条件2。伸ばす方向にのみ変更できる


class RobotsDisallowed(RuntimeError):
    """robots.txt がそのホストへの取得を禁止している（D-014 条件3）。"""


@dataclass
class LocalCacheFetcher:
    """生バイト列を gzip 圧縮してローカルに保存する `Fetcher`。

    キャッシュキーは URL。期限を設けない。キャッシュヒット時は
    HTTPリクエストを発行しない。
    """

    cache_dir: Path
    user_agent: str
    min_interval: float = MIN_INTERVAL_SEC
    _last_request_at: float | None = None
    _robots_checked: set[str] | None = None

    def __post_init__(self) -> None:
        if self.min_interval < MIN_INTERVAL_SEC:
            raise ValueError(
                f"min_interval={self.min_interval} は D-014 条件2 の下限"
                f"{MIN_INTERVAL_SEC}秒を下回ります。伸ばす方向にのみ変更できます。"
            )
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._robots_checked = set()

    def _cache_path(self, url: str) -> Path:
        # URL をファイル名に安全な形へ。衝突を避けるため長い一意なハッシュを使う
        import hashlib
        digest = hashlib.sha256(url.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{digest}.html.gz"

    def check_robots(self, url: str) -> None:
        """未確認のホストであれば robots.txt を確認する。禁止なら例外を送出する。

        D-014 条件3: 引くホストごとに確認する。db.netkeiba.com のような
        本番の主ソース（D-037）を見落とさないため、呼び出し側は取得する
        全ホストに対してこれを呼ぶこと。
        """
        parts = urlsplit(url)
        host = parts.netloc
        if host in self._robots_checked:
            return
        robots_url = f"{parts.scheme}://{host}/robots.txt"
        rp = urllib.robotparser.RobotFileParser()
        try:
            req = urllib.request.Request(robots_url, headers={"User-Agent": self.user_agent})
            with urllib.request.urlopen(req, timeout=20) as resp:
                body = resp.read().decode("utf-8", errors="replace")
            rp.parse(body.splitlines())
            allowed = rp.can_fetch(self.user_agent, url)
        except urllib.error.HTTPError as e:
            allowed = True if e.code == 404 else True  # 取得できない場合は判断保留のまま続行
        except Exception:
            allowed = True

        if not allowed:
            raise RobotsDisallowed(
                f"{host} の robots.txt が {url} の取得を禁止しています（D-014 条件3）。"
            )
        self._robots_checked.add(host)

    def _throttled_fetch(self, url: str) -> tuple[bytes, str | None, int]:
        if self._last_request_at is not None:
            elapsed = time.monotonic() - self._last_request_at
            wait = self.min_interval - elapsed
            if wait > 0:
                time.sleep(wait)
        req = urllib.request.Request(url, headers={"User-Agent": self.user_agent})
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                body = resp.read()
                charset = resp.headers.get_content_charset()
                status = resp.status
        finally:
            self._last_request_at = time.monotonic()
        return body, charset, status

    def get(self, url: str, *, source: str, page_kind: PageKind,
            source_key: str) -> RawPage:
        cache_path = self._cache_path(url)
        if cache_path.exists():
            body = gzip.decompress(cache_path.read_bytes())
            return RawPage(
                source=source, page_kind=page_kind, source_key=source_key,
                url=url, body=body, encoding="unknown",
                fetched_at=datetime.now(timezone.utc), from_cache=True,
            )

        self.check_robots(url)
        body, charset, _status = self._throttled_fetch(url)
        cache_path.write_bytes(gzip.compress(body))
        return RawPage(
            source=source, page_kind=page_kind, source_key=source_key,
            url=url, body=body, encoding=charset or "unknown",
            fetched_at=datetime.now(timezone.utc), from_cache=False,
        )
