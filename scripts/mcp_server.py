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
    explain_race      同じレースの**スコアの内訳**を`F-xxx`単位で返す
                       （`D-192`。SHAP値。モデルが「なぜ」その評価をした
                       かを、人手で計算できない量に分解して見せる）
    query_history     本番DB（38,000レース超）に読み取り専用SQLを投げる
                       （`D-192`。過去の傾向を自由に問い合わせる）
    log_prediction    **発走前に**予想を記録する（`D-195`）。結果確定済み
                       のレースは拒否する
    score_agent       記録済みの予想を採点する（`D-195`。信頼区間つき）
    list_logged_predictions  記録の一覧
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
from umagic.production_model import (
    CACHE_META_FILENAME,
    explain_with_cache,
    predict_with_cache,
)
from umagic.prediction_log import (
    DEFAULT_LOG_PATH,
    Pick,
    PredictionRecord,
    append_prediction,
    list_predictions,
    now_iso,
    score_predictions,
)
from umagic.sources.netkeiba import PostPositionsNotDrawn, parse_shutuba

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
    try:
        shutuba = parse_shutuba(page)
    except PostPositionsNotDrawn as e:
        return {"error": str(e), "retry_later": True}

    if not shutuba.entries:
        return {"error": "出馬表が取得できませんでした（未公開、またはページ構造の想定外）"}

    conn = duckdb.connect(":memory:")
    try:
        rid = build_overlay(conn, shutuba)
    except ValueError as e:  # D-184 の重複ガード。生の例外にせず理由を返す
        conn.close()
        return {"error": str(e),
                "hint": "既に結果が確定しているレースです。過去レースの成績は "
                        "query_history で引けます。"}
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
def explain_race(race_id: str, top_k: int = 6) -> dict:
    """`predict_race` と同じレースについて、**各馬のスコアの内訳**を返す
    （`D-192`）。

    LightGBMのSHAP値を169の特徴量列ごとに求め、`F-xxx`（`domain-knowledge.md`
    の特徴量カタログ）単位に合計して、寄与の大きい順に `top_k` 件返す。
    「モデルがなぜこの馬を高く（低く）評価したか」を、人手では計算できない
    量（`F-304` 速度指数・`F-809` キャリア成績率・`F-701` 人気帯で交絡除去
    した騎手の実力など）に分解して見せるためのツール。

    **`contribution` はスコア（softmax前の生margin）のスケールで、
    レース内の相対比較にのみ意味がある。** 確率そのものへの寄与ではない。

    使い方: `predict_race` で全体像を掴み、市場と乖離した馬について
    `explain_race` で理由を見て、その理由が外部情報（追い切り・馬体重・
    当日の馬場等）と整合するかを検証する——という順で使う。
    """
    meta_path = PREDICTION_CACHE_DIR / CACHE_META_FILENAME
    if not meta_path.exists():
        return {"error": f"推論キャッシュがありません: {meta_path}。"
                          "先に scripts/build_prediction_cache.py を実行してください。"}

    fetcher = LocalCacheFetcher(cache_dir=ROOT / "data" / "cache", user_agent=UA, min_interval=5.0)
    url = f"https://race.netkeiba.com/race/shutuba.html?race_id={race_id}"
    page = fetcher.get(url, source="netkeiba_jra", page_kind="shutuba", source_key=race_id)
    try:
        shutuba = parse_shutuba(page)
    except PostPositionsNotDrawn as e:
        return {"error": str(e), "retry_later": True}
    if not shutuba.entries:
        return {"error": "出馬表が取得できませんでした（未公開、またはページ構造の想定外）"}

    conn = duckdb.connect(":memory:")
    try:
        try:
            rid = build_overlay(conn, shutuba)
        except ValueError as e:  # D-184 の重複ガード。生の例外にせず理由を返す
            return {"error": str(e),
                    "hint": "既に結果が確定しているレースです。過去レースの成績は "
                            "query_history で引けます。"}
        target_date = shutuba.race["date"]
        contrib = explain_with_cache(conn, rid, target_date, PREDICTION_CACHE_DIR, top_k=top_k)
        numbers = conn.execute(
            "SELECT horse_id, number FROM runners WHERE race_id = ?", [rid],
        ).pl()
    finally:
        conn.close()

    names = pl.DataFrame([
        {"number": e["number"], "horse_name": e["horse_name"]} for e in shutuba.entries
    ])
    joined = (contrib.join(numbers, on="horse_id", how="inner")
                     .join(names, on="number", how="left"))

    by_horse: dict[int, dict] = {}
    for r in joined.iter_rows(named=True):
        entry = by_horse.setdefault(
            r["number"], {"number": r["number"], "horse_name": r["horse_name"], "drivers": []},
        )
        entry["drivers"].append({
            "feature": r["family"], "label": r["label"],
            "contribution": round(r["contribution"], 4),
        })

    return {
        "race": {"title": shutuba.race["title"], "date": str(shutuba.race["date"])},
        "horses": [by_horse[k] for k in sorted(by_horse)],
        "note": "contribution はスコア（softmax前）のスケール。レース内の相対比較にのみ意味がある。"
                "特徴量の定義は lookup_doc('F-xxx') で引ける。",
        "caveat": "市場に対する優位は未実証（D-119〜D-190）。内訳が説得的でも賭けの根拠にしないこと。",
    }


@server.tool()
def query_history(sql: str, max_rows: int = 200) -> dict:
    """本番DB（`data/umagic.duckdb`、38,000レース超・54万出走行）に
    **読み取り専用**のSQLを投げる（`D-192`）。

    `SELECT` / `WITH` で始まるクエリのみ受け付ける。接続自体が
    `read_only=True` なので書き込みは物理的にできない（`D-176`/`D-182`
    と同じ設計）。結果は `max_rows` 件で打ち切る。

    主なテーブル:
      races    race_id, date, course, race_number, distance, surface,
               direction, grade, track_condition, weather, n_starters,
               race_class, prize, corner_nos
      runners  race_id, horse_id, number, frame, jockey_id, trainer_id,
               weight_carried, horse_weight, weight_diff, age, sex,
               odds_win, popularity, status, finish_pos, margin,
               time_sec, last_3f, corners, owner_id
      horses / jockeys / trainers / owners   各エンティティのマスタ
      payouts  race_id, bet_type, comb_key, combination, payout, popularity
      odds     race_id, bet_type, comb_key, odds_low, odds_high, as_of
      laps     レースのラップタイム

    **`runners.status` の値は日本語**: `'出走'`（通常。大半がこれ）・
    `'出走取消'`・`'競走除外'`・`'競走中止'`・`'失格'`・`'降着'`。
    成績を集計するときは `status NOT IN ('出走取消','競走除外')` で
    絞るのが既定（`D-073`。この2つは馬券の対象外で確率の正規化にも
    含めない）。**`status='ran'` のような英語の値は存在しない。**

    `races.surface` は `'芝'`/`'ダート'`、`races.grade` は `NULL`
    （無格付。大半がこれ）・`'G1'`/`'G2'`/`'G3'`・`'L'`（リステッド）。
    `races.course` は `'中山'`/`'東京'` 等の日本語の競馬場名。

    **列の値が期待と違うときは、まず `SELECT DISTINCT <列> FROM <表>`
    で実際の値を確認すること**（結果が0件のときの原因の大半がこれ）。

    **このDBは結果が確定した過去レースのみを持つ。** 予測対象の未来の
    レースはここに無い（`predict_race` が出馬表から別途取得する）。
    """
    stripped = sql.lstrip().lstrip("(").lstrip()
    if not re.match(r"(?i)^(select|with)\b", stripped):
        return {"error": "SELECT または WITH で始まるクエリのみ実行できます。"}
    if max_rows < 1 or max_rows > 2000:
        return {"error": "max_rows は 1〜2000 の範囲で指定してください。"}

    conn = duckdb.connect(str(ROOT / "data" / "umagic.duckdb"), read_only=True)
    try:
        df = conn.execute(sql).pl()
    except Exception as e:  # noqa: BLE001 — SQLの誤りをそのままエージェントに返す
        return {"error": f"クエリの実行に失敗しました: {e}"}
    finally:
        conn.close()

    truncated = len(df) > max_rows
    df = df.head(max_rows)
    return {
        "columns": df.columns,
        "rows": [
            {k: (str(v) if not isinstance(v, (int, float, type(None))) else v)
             for k, v in row.items()}
            for row in df.iter_rows(named=True)
        ],
        "n_rows": len(df),
        "truncated": truncated,
    }


@server.tool()
def log_prediction(
    race_id: str, picks: list[dict], agent: str = "selector-v1",
    confidence: str = "", reasoning: str = "", model_probs: dict | None = None,
) -> dict:
    """**発走前に**予想を記録する（`D-195`）。後で採点するために使う。

    `picks` は買い目のリスト。各要素は
    `{"bet_type": "複勝", "horse_numbers": [3], "stake": 1.0, "note": "..."}`。
    `bet_type` は `"単勝"`/`"複勝"`/`"ワイド"` のみ（3連単等は`D-008`により
    検出力が無いため対象外）。`stake` は金額ではなく単位を固定した相対値。

    **結果が既に確定しているレースは拒否する。** 結果を見てから書いた予想は
    検証に使えないため（`D-187` が示した in-sample の罠と同じ）。同じ
    `race_id` × `agent` の重複記録も拒否する（後出しでの差し替え防止）。

    `agent` は構成の名前。異なる方針のエージェントを比較したいときに分ける。
    `confidence` は自己申告の確信度（`"高"`/`"中"`/`"低"` など）。

    **記録しただけでは何も検証されない。** `score_predictions` で採点でき
    るのは結果が確定してからで、`D-008` の規律上、意味のある結論には
    相当数の蓄積が要る。
    """
    try:
        pick_objs = [
            Pick(bet_type=p["bet_type"], horse_numbers=[int(n) for n in p["horse_numbers"]],
                 stake=float(p.get("stake", 1.0)), note=str(p.get("note", "")))
            for p in picks
        ]
    except (KeyError, TypeError, ValueError) as e:
        return {"error": f"picks の形式が不正です: {e}"}

    conn = duckdb.connect(str(ROOT / "data" / "umagic.duckdb"), read_only=True)
    try:
        race_date = conn.execute(
            "SELECT date FROM races WHERE race_id = ?", [int(race_id)],
        ).fetchone()
        record = PredictionRecord(
            race_id=int(race_id),
            race_date=str(race_date[0]) if race_date else "",
            logged_at=now_iso(), agent=agent, picks=pick_objs,
            confidence=confidence, reasoning=reasoning,
            model_probs={str(k): float(v) for k, v in (model_probs or {}).items()},
        )
        append_prediction(record, log_path=ROOT / DEFAULT_LOG_PATH, conn=conn)
    except ValueError as e:
        return {"error": str(e)}
    finally:
        conn.close()

    return {
        "logged": True, "race_id": int(race_id), "agent": agent,
        "n_picks": len(pick_objs),
        "note": "発走前の記録として保存した。結果確定後に score_predictions で採点できる。",
    }


@server.tool()
def score_agent(agent: str | None = None) -> dict:
    """記録済みの予想のうち、**結果が確定したものだけ**を採点する（`D-195`）。

    券種別・全体の賭け数・回収率・95%信頼区間（レース単位ではなく賭け
    1点単位のブートストラップ）を返す。`agent` を指定するとその構成だけを
    採点する（省略時は全件）。

    **`D-008` の規律を必ず守ること**——単勝回収率の標準誤差は概算
    `4/√n` で、n=240程度でも74%〜126%の範囲は有意差を主張できない。
    少数の結果から「当たった」「この方法は有効だ」と結論しないこと。
    戻り値の `caveat` に現在の n に基づく注意を含めている。
    """
    conn = duckdb.connect(str(ROOT / "data" / "umagic.duckdb"), read_only=True)
    try:
        return score_predictions(conn, log_path=ROOT / DEFAULT_LOG_PATH, agent=agent)
    finally:
        conn.close()


@server.tool()
def list_logged_predictions(agent: str | None = None) -> dict:
    """記録済みの予想を一覧で返す（採点はしない、`D-195`）。"""
    df = list_predictions(log_path=ROOT / DEFAULT_LOG_PATH, agent=agent)
    return {"n": len(df), "predictions": df.to_dicts()}


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
