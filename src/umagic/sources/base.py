"""ソース差し替えの境界（`docs/spec/002-loader.md` / `D-009`）。

`Fetcher` はレート制限・キャッシュ・robots.txt 確認を担う（全ソース共通）。
`Source` は URL構成とパースを担う（ソース固有、`D-009` の差し替え点）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Literal, Protocol

PageKind = Literal["day_index", "archive", "shutuba"]
Outcome = Literal["ok", "empty", "http_error", "parse_error"]


@dataclass(frozen=True)
class RawPage:
    source: str          # 'netkeiba_jra' | 'netkeiba_nar' | 'jrdb'
    page_kind: PageKind
    source_key: str      # day_index なら YYYYMMDD、それ以外はレースのキー
    url: str
    body: bytes           # 生バイト列。復号前
    encoding: str
    fetched_at: datetime
    from_cache: bool


@dataclass(frozen=True)
class RejectedRow:
    source_key: str
    row_ref: str | None   # 馬番など。行を特定できない場合は None
    reason: str
    raw: str


@dataclass(frozen=True)
class ParsedRace:
    race: dict                          # races 1行分
    runners: list[dict] = field(default_factory=list)
    payouts: list[dict] = field(default_factory=list)
    odds: list[dict] = field(default_factory=list)
    laps: list[dict] = field(default_factory=list)
    rejected: list[RejectedRow] = field(default_factory=list)   # R-013


@dataclass(frozen=True)
class ParsedShutuba:
    """発走前の出馬表（`shutuba`）のパース結果。`ParsedRace` とは意図的に
    別型にする——結果・払戻・オッズ・ラップを持たない（まだ発走していない
    ため存在しない）。本番の `ingest_race()`（`ParsedRace` 前提）とは
    別経路（推論パイプライン専用）で使う。"""
    race: dict                          # 開催情報（結果に依存する列は持たない）
    entries: list[dict] = field(default_factory=list)   # 出走予定馬（着順・オッズ無し）


class Fetcher(Protocol):
    """取得の責務。レート制限・キャッシュ・robots.txt 確認を担う。"""

    def get(self, url: str, *, source: str, page_kind: PageKind,
            source_key: str) -> RawPage: ...


class Source(Protocol):
    """ソース固有の知識を閉じ込める。`D-009` の差し替え点。"""

    name: str

    def list_race_keys(self, day: date) -> list[str]: ...

    def url_for(self, source_key: str, page_kind: PageKind) -> str: ...

    def parse(self, page: RawPage) -> ParsedRace: ...
