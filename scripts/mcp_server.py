#!/usr/bin/env python3
"""UMAGICをLLMエージェントに接続するMCPサーバー（`Q-048`、着手条件を
ユーザーが明示的に解除。手が空いた時間の実験実装、`D-181`/`D-182`の続き）。

`Q-048`が挙げていた2つの前提——出馬表パーサー・運用推論パス——は
`D-181`で揃った。ここではその上にツールを載せる。

    predict_race      まだ発走していないレースの勝率を予測する
                       （`src/umagic/inference.py` + 推論キャッシュ
                       `D-183`。対象レース1件だけ特徴量計算する高速経路。
                       キャッシュが無い場合は先に
                       `scripts/build_prediction_cache.py` を実行すること）
    lookup_doc        `D-xxx`/`Q-xxx`/`R-xxx`/`F-xxx` の全文を1件返す
                       （それぞれ `decisions.md`/`open-questions.md`/
                       `requirements.md`/`domain-knowledge.md`）
    search_docs       上記4ファイルをキーワード横断検索する
    list_source_files `src/umagic/` または `docs/` 配下のファイル一覧を返す
    read_source       上記配下の `.py`/`.md` ファイルの中身を返す
                       （実装本体・仕様書を直接読ませる。`docs-writing.md`
                       の「なぜ」（decisions.md）と「何を作るか」（spec/）
                       の分離はそのままエージェント側にも渡る）

**市場に対する優位は未確認のまま**（`D-119`〜`D-180`）。`predict_race`
が返す確率をそのまま賭けの根拠にしないこと——`R-030`が要求する「市場への
上乗せ」は実証されていない。

使い方（stdio transport、Claude Desktop 等から接続する想定）:
    uv run python scripts/mcp_server.py
"""

from __future__ import annotations

import json
import re
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import duckdb
import polars as pl
from mcp.server.mcpserver import MCPServer

from umagic.cache import LocalCacheFetcher
from umagic.inference import build_overlay
from umagic.production_model import CACHE_META_FILENAME, predict_with_cache
from umagic.sources.netkeiba import parse_shutuba

ROOT = Path(__file__).parent.parent
UA = "UMAGIC-dev/0.1 (personal research; contact: repository owner)"
PREDICTION_CACHE_DIR = ROOT / "data" / "prediction_cache"

# ID接頭辞 → (ファイル, 見出しレベル)。`docs-writing.md` のID体系どおり
# （decisions.md/open-questions.md は `## D-xxx`、requirements.md は
# `### R-xxx`、domain-knowledge.md は `#### F-xxx`）
_DOC_CONFIG = {
    "D": (ROOT / "docs" / "decisions.md", 2),
    "Q": (ROOT / "docs" / "open-questions.md", 2),
    "R": (ROOT / "docs" / "requirements.md", 3),
    "F": (ROOT / "docs" / "domain-knowledge.md", 4),
}

# `read_source`/`list_source_files` が読める範囲。リポジトリ外への
# パストラバーサルを防ぐため、許可ディレクトリの実体パスに限定する
_READABLE_ROOTS = [(ROOT / "src" / "umagic").resolve(), (ROOT / "docs").resolve()]

server = MCPServer(
    name="umagic",
    instructions=(
        "JRA平地G1中心の競馬予測プロジェクト（UMAGIC）へのアクセスを提供する。"
        "predict_raceはまだ発走していないレースの勝率を予測するが、"
        "市場（単勝オッズ）に対する優位は未確認（D-119〜D-180）。"
        "確率をそのまま賭けの根拠にしないこと。"
    ),
)


@server.tool()
def predict_race(race_id: str) -> dict:
    """まだ発走していないJRAレースの各馬の勝率を予測する。

    `race_id` は12桁の netkeiba レースID（例: "202606030811"）。出馬表
    （発走の数日前から公開）を取得・パースし、`data/umagic.duckdb`
    （書き換えない）に対象レースを重ね合わせて、事前に作った推論キャッシュ
    （`scripts/build_prediction_cache.py`、`D-183`）で予測する——対象
    レース1件だけ特徴量計算する高速経路（本番DB全履歴の再学習はしない）。

    **キャッシュが無い場合はエラーを返す。** 先に
    `uv run python scripts/build_prediction_cache.py` を実行すること
    （本番DBを更新した後も再実行が必要）。

    戻り値の `win_prob` は市場（単勝オッズ）に対する優位が未実証の値
    （`D-119`〜`D-180`）——賭けの根拠にはしないこと。
    """
    meta_path = PREDICTION_CACHE_DIR / CACHE_META_FILENAME
    if not meta_path.exists():
        return {"error": f"推論キャッシュがありません: {meta_path}。"
                          "先に scripts/build_prediction_cache.py を実行してください。"}

    fetcher = LocalCacheFetcher(cache_dir=ROOT / "data" / "cache", user_agent=UA, min_interval=5.0)
    url = f"https://race.netkeiba.com/race/shutuba.html?race_id={race_id}"
    page = fetcher.get(url, source="netkeiba_jra", page_kind="shutuba", source_key=race_id)
    shutuba = parse_shutuba(page)

    if not shutuba.entries:
        return {"error": "出馬表が取得できませんでした（未公開、またはページ構造の想定外）"}

    conn = duckdb.connect(":memory:")
    rid = build_overlay(conn, shutuba)
    target_date = shutuba.race["date"]

    cache_meta = json.loads(meta_path.read_text(encoding="utf-8"))
    trained_through = date.fromisoformat(cache_meta["trained_through"])
    gap_days = max(0, (target_date - trained_through).days - 1)

    out = predict_with_cache(conn, rid, target_date, PREDICTION_CACHE_DIR)

    numbers = conn.execute("SELECT horse_id, number FROM runners WHERE race_id = ?", [rid]).pl()
    entry_names = pl.DataFrame([
        {"number": e["number"], "horse_name": e["horse_name"]} for e in shutuba.entries
    ])
    result = (
        out.join(numbers, on="horse_id", how="inner")
        .join(entry_names, on="number", how="left")
        .select(["number", "horse_name", "win_prob"])
        .sort("win_prob", descending=True)
    )
    conn.close()

    return {
        "race": {
            "title": shutuba.race["title"], "date": str(target_date),
            "course": shutuba.race["course"], "race_number": shutuba.race["race_number"],
            "grade": shutuba.race["grade"], "race_class": shutuba.race["race_class"],
        },
        "training_data_gap_days": gap_days,
        "predictions": [
            {"number": r["number"], "horse_name": r["horse_name"], "win_prob": round(r["win_prob"], 4)}
            for r in result.iter_rows(named=True)
        ],
        "caveat": "市場（単勝オッズ）に対する優位は未実証（D-119〜D-180）。賭けの根拠にしないこと。",
    }


@server.tool()
def lookup_doc(doc_id: str) -> str:
    """`D-xxx`（決定）・`Q-xxx`（未決事項）・`R-xxx`（要件）・`F-xxx`
    （特徴量カタログ）のいずれかを指定して、対応する項目の全文を1件返す。

    `docs-writing.md` の分離方針どおり、`D-xxx` は「なぜ」（判断の根拠）、
    `R-xxx`/`F-xxx` は「何を」（満たすべきこと・特徴量の定義）を持つ。
    実装の詳細（「どう作るか」）は `read_source` で仕様書（`docs/spec/`）
    やソース本体を直接読むこと。
    """
    prefix = doc_id.split("-")[0].upper()
    cfg = _DOC_CONFIG.get(prefix)
    if not cfg:
        return f"未対応のID接頭辞です: {prefix!r}（D/Q/R/F に対応）"
    path, level = cfg
    hashes = "#" * level
    text = path.read_text(encoding="utf-8")
    m = re.search(
        rf"^{hashes} {re.escape(doc_id)}\b.*?(?=^#{{1,{level}}} |\Z)", text, re.S | re.M,
    )
    if not m:
        return f"{doc_id} は {path.name} に見つかりませんでした。"
    return m.group(0).strip()


@server.tool()
def search_docs(query: str, doc: str | None = None, max_results: int = 10) -> list[dict]:
    """`decisions.md`/`open-questions.md`/`requirements.md`/
    `domain-knowledge.md` をキーワード横断検索する。

    `doc` を `"D"`/`"Q"`/`"R"`/`"F"` のいずれかに絞ると、そのファイルだけ
    検索する（省略時は4ファイル全部）。ヒットした項目の見出しと冒頭を
    返す——本文全体は `lookup_doc` で取る。
    """
    targets = [_DOC_CONFIG[doc.upper()]] if doc else list(_DOC_CONFIG.values())
    hits = []
    for path, level in targets:
        hashes = "#" * level
        text = path.read_text(encoding="utf-8")
        entries = re.split(rf"(?=^{hashes} [A-Z]-\d)", text, flags=re.M)
        for entry in entries:
            if not re.match(rf"^{hashes} [A-Z]-\d", entry):
                continue
            if query not in entry:
                continue
            header = entry.splitlines()[0]
            hm = re.match(rf"^{hashes} ([A-Z]-\d+)\s*(.*)$", header)
            snippet = entry[len(header):].strip()[:300]
            hits.append({"id": hm.group(1), "title": hm.group(2), "file": path.name, "snippet": snippet})
            if len(hits) >= max_results:
                return hits
    return hits


def _resolve_readable(path: str) -> Path | None:
    target = (ROOT / path).resolve()
    if not any(target == r or r in target.parents for r in _READABLE_ROOTS):
        return None
    return target


@server.tool()
def list_source_files(subdir: str = "src/umagic") -> list[str]:
    """`src/umagic/` または `docs/` 配下の `.py`/`.md` ファイル一覧を返す
    （`read_source` に渡すパスを探すのに使う）。`subdir` はリポジトリルート
    からの相対パス（例: `"src/umagic/features"`、`"docs/spec"`）。"""
    target = _resolve_readable(subdir)
    if target is None or not target.is_dir():
        return []
    return sorted(
        str(p.relative_to(ROOT)) for p in target.rglob("*")
        if p.is_file() and p.suffix in (".py", ".md")
    )


@server.tool()
def read_source(path: str) -> str:
    """`src/umagic/` または `docs/` 配下の `.py`/`.md` ファイルの中身を
    そのまま返す。`path` はリポジトリルートからの相対パス
    （例: `"src/umagic/features/f304.py"`、`"docs/spec/007-stage2-ranker.md"`）。
    範囲外のパスやこの2拡張子以外は拒否する。"""
    target = _resolve_readable(path)
    if target is None:
        return "許可されていないパスです。src/umagic/ または docs/ 配下のみ読めます。"
    if not target.is_file():
        return f"ファイルが見つかりません: {path}"
    if target.suffix not in (".py", ".md"):
        return "対応していない拡張子です（.py/.md のみ）。"
    return target.read_text(encoding="utf-8")


if __name__ == "__main__":
    server.run()
