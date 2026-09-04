"""まだ発走していないレースを予測する運用推論パス（`Q-048`、`D-181`）。

`Q-047` 段階②の`data/umagic_nar.duckdb`分離（`D-176`）と同じ理由——
**JRA本番DB（`data/umagic.duckdb`）を一切書き換えない**——で、対象レース
を書き込むのではなく、`:memory:` 接続に本番DBを読み取り専用で
`ATTACH` し、`races`/`runners`/`horses`/`jockeys`/`trainers`/`source_ids`
を「本番データ ∪ 対象レース1件」の VIEW として重ねる。

この重ね合わせにより、`build_features()`・`Stage2FoldRunner` など既存の
学習・推論コードは一切変更せずにそのまま使える——`races`/`runners` を
素朴なテーブル名で参照するSQLは、対象レースが最初から本番DBにあった
かのように振る舞う。
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path

import duckdb
import polars as pl

from umagic.sources.base import ParsedShutuba

# `__file__` から辿った絶対パス（`D-200`）。相対パスのままだと、この
# プロセスの**カレントディレクトリ**次第で解決先が変わる。`uv run` や
# 開発時のシェルはプロジェクトルートが cwd になるため長らく問題が
# 顕在化しなかったが、Claude Desktop が `mcp_server.py` をサブプロセス
# として起動する際は cwd がプロジェクトルートではなく（実測: `/`）、
# `data/umagic.duckdb` が `/data/umagic.duckdb` に解決されて
# `IOException: database does not exist` になっていた——本番当日に
# `predict_race`/`explain_race` が原因不明のまま失敗し続けた真因。
PROD_DB_PATH = str(Path(__file__).resolve().parent.parent.parent / "data" / "umagic.duckdb")

# `races`/`runners` に UNION する側で埋める `source` 値。本番の "netkeiba_jra"
# と区別する——対象レースの行は結果が存在しない未確定データであることを
# 後から見分けられるようにするため
PENDING_SOURCE = "netkeiba_jra_pending"

_ENTITY_TABLES = {"horse": "horses", "jockey": "jockeys", "trainer": "trainers"}


def _resolve_entities(
    conn: duckdb.DuckDBPyConnection, entity_type: str, source_keys: list[str],
) -> tuple[dict[str, int], int]:
    """`prod.source_ids` から既存IDを引く。無ければ `prod` 側の最大値+1から
    連番で割り当てる（このセッション内だけの一時ID。`prod` には書かない）。
    戻り値: (source_key -> internal_id, 次に使える連番)。
    """
    keys = list(dict.fromkeys(source_keys))  # 重複除去、順序維持
    if not keys:
        return {}, 0
    rows = conn.execute(
        "SELECT source_key, internal_id FROM prod.source_ids "
        "WHERE entity_type = ? AND source_key = ANY(?)",
        [entity_type, keys],
    ).fetchall()
    resolved = {k: v for k, v in rows}
    next_id = conn.execute(
        "SELECT COALESCE(MAX(internal_id), 0) + 1 FROM prod.source_ids WHERE entity_type = ?",
        [entity_type],
    ).fetchone()[0]
    for k in keys:
        if k not in resolved:
            resolved[k] = next_id
            next_id += 1
    return resolved, next_id


def build_overlay(conn: duckdb.DuckDBPyConnection, shutuba: ParsedShutuba) -> int:
    """`:memory:` 接続 `conn` に `prod`（本番DB、読み取り専用）を ATTACH し、
    `races`/`runners`/`horses`/`jockeys`/`trainers`/`owners`/`source_ids`/
    `payouts`/`odds`/`laps` を「本番 ∪ 対象レース」の VIEW として作る。

    戻り値: 対象レースの `race_id`（`shutuba.race["race_id"]` と同じ）。
    """
    conn.execute(f"ATTACH '{PROD_DB_PATH}' AS prod (READ_ONLY)")
    r = shutuba.race
    fetched_at = datetime.now(timezone.utc)

    # `D-184`: 対象レースが既に本番DBに実在すると（本番DBが対象レース日を
    # 追い越して最新化された場合に起こりうる）、UNION ALLビューで
    # race_id が重複し、`join_asof` 等の下流結合が異常な実行計画になる
    # （実測: 特徴量計算が数時間かかった。原因はこの重複だった）。
    # 静かに壊れたデータを返すのではなく、ここで明示的に検出して拒否する
    existing = conn.execute(
        "SELECT COUNT(*) FROM prod.races WHERE race_id = ?", [r["race_id"]],
    ).fetchone()[0]
    if existing:
        raise ValueError(
            f"race_id={r['race_id']} は既に本番DB（{PROD_DB_PATH}）に存在します"
            "（対象レース日を本番DBの取り込みが追い越した可能性があります）。"
            "build_overlay はまだ結果の無いレース専用のため、この重ね合わせは行いません。"
        )

    horse_ids, _ = _resolve_entities(conn, "horse", [e["horse_source_key"] for e in shutuba.entries])
    jockey_ids, _ = _resolve_entities(
        conn, "jockey", [e["jockey_source_key"] for e in shutuba.entries if e["jockey_source_key"]],
    )
    trainer_ids, _ = _resolve_entities(
        conn, "trainer", [e["trainer_source_key"] for e in shutuba.entries if e["trainer_source_key"]],
    )

    n_starters = len(shutuba.entries)  # 出馬表に現れる頭数（取消等は既に反映済み、D-181）
    races_df = pl.DataFrame([{
        "race_id": r["race_id"], "date": r["date"], "course": r["course"],
        "race_number": r["race_number"], "post_time": r["post_time"],
        "distance": r["distance"], "surface": r["surface"], "direction": r["direction"],
        "grade": r["grade"], "track_condition": r["track_condition"], "weather": r["weather"],
        "weather_forecast": None, "n_entries": r["n_entries"] or n_starters, "n_starters": n_starters,
        "prize": None, "corner_nos": None, "race_class": r["race_class"],
        "weight_rule": r["weight_rule"], "meeting_no": r["meeting_no"], "meeting_day": r["meeting_day"],
        "source": PENDING_SOURCE, "fetched_at": fetched_at,
    }])

    runner_rows = []
    new_horses, new_jockeys, new_trainers = [], [], []
    known_horses = set(conn.execute("SELECT horse_id FROM prod.horses").fetchall())
    known_jockeys = set(conn.execute("SELECT jockey_id FROM prod.jockeys").fetchall())
    known_trainers = set(conn.execute("SELECT trainer_id FROM prod.trainers").fetchall())
    for e in shutuba.entries:
        hid = horse_ids[e["horse_source_key"]]
        jid = jockey_ids.get(e["jockey_source_key"]) if e["jockey_source_key"] else None
        tid = trainer_ids.get(e["trainer_source_key"]) if e["trainer_source_key"] else None
        if (hid,) not in known_horses:
            new_horses.append({"horse_id": hid, "name": e["horse_name"], "birth": None,
                                "sire_id": None, "dam_id": None, "damsire_id": None,
                                "source": PENDING_SOURCE, "fetched_at": fetched_at})
        if jid is not None and (jid,) not in known_jockeys:
            new_jockeys.append({"jockey_id": jid, "name": e["jockey_name"],
                                 "source": PENDING_SOURCE, "fetched_at": fetched_at})
        if tid is not None and (tid,) not in known_trainers:
            new_trainers.append({"trainer_id": tid, "name": e["trainer_name"],
                                  "source": PENDING_SOURCE, "fetched_at": fetched_at})
        runner_rows.append({
            "race_id": r["race_id"], "horse_id": hid, "number": e["number"], "frame": e["frame"],
            "jockey_id": jid, "trainer_id": tid, "owner_id": None,
            "weight_carried": e["weight_carried"], "horse_weight": e["horse_weight"],
            "weight_diff": e["weight_diff"], "age": e["age"], "sex": e["sex"],
            "odds_win": None, "popularity": None, "status": "出走",
            "finish_pos": None, "margin": None, "time_sec": None, "last_3f": None,
            "corners": None, "affiliation": e["affiliation"],
            "source": PENDING_SOURCE, "fetched_at": fetched_at,
        })
    runners_df = pl.DataFrame(runner_rows)

    source_id_rows = (
        [{"entity_type": "horse", "internal_id": v, "source": "netkeiba_jra", "source_key": k,
          "fetched_at": fetched_at} for k, v in horse_ids.items()]
        + [{"entity_type": "jockey", "internal_id": v, "source": "netkeiba_jra", "source_key": k,
            "fetched_at": fetched_at} for k, v in jockey_ids.items()]
        + [{"entity_type": "trainer", "internal_id": v, "source": "netkeiba_jra", "source_key": k,
            "fetched_at": fetched_at} for k, v in trainer_ids.items()]
    )
    source_ids_df = pl.DataFrame(source_id_rows)

    conn.register("_races_new", races_df)
    conn.register("_runners_new", runners_df)
    conn.register("_source_ids_new", source_ids_df)
    conn.execute("CREATE TABLE pending_races AS SELECT * FROM _races_new")
    conn.execute("CREATE TABLE pending_runners AS SELECT * FROM _runners_new")
    conn.execute("CREATE TABLE pending_source_ids AS SELECT * FROM _source_ids_new")
    conn.unregister("_races_new")
    conn.unregister("_runners_new")
    conn.unregister("_source_ids_new")

    if new_horses:
        conn.register("_horses_new", pl.DataFrame(new_horses))
        conn.execute("CREATE TABLE pending_horses AS SELECT * FROM _horses_new")
        conn.unregister("_horses_new")
    else:
        conn.execute("CREATE TABLE pending_horses AS SELECT * FROM prod.horses WHERE FALSE")
    if new_jockeys:
        conn.register("_jockeys_new", pl.DataFrame(new_jockeys))
        conn.execute("CREATE TABLE pending_jockeys AS SELECT * FROM _jockeys_new")
        conn.unregister("_jockeys_new")
    else:
        conn.execute("CREATE TABLE pending_jockeys AS SELECT * FROM prod.jockeys WHERE FALSE")
    if new_trainers:
        conn.register("_trainers_new", pl.DataFrame(new_trainers))
        conn.execute("CREATE TABLE pending_trainers AS SELECT * FROM _trainers_new")
        conn.unregister("_trainers_new")
    else:
        conn.execute("CREATE TABLE pending_trainers AS SELECT * FROM prod.trainers WHERE FALSE")

    conn.execute("CREATE VIEW races AS SELECT * FROM prod.races UNION ALL BY NAME SELECT * FROM pending_races")
    conn.execute("CREATE VIEW runners AS SELECT * FROM prod.runners UNION ALL BY NAME SELECT * FROM pending_runners")
    conn.execute("CREATE VIEW horses AS SELECT * FROM prod.horses UNION ALL BY NAME SELECT * FROM pending_horses")
    conn.execute("CREATE VIEW jockeys AS SELECT * FROM prod.jockeys UNION ALL BY NAME SELECT * FROM pending_jockeys")
    conn.execute("CREATE VIEW trainers AS SELECT * FROM prod.trainers UNION ALL BY NAME SELECT * FROM pending_trainers")
    conn.execute("CREATE VIEW owners AS SELECT * FROM prod.owners")
    conn.execute(
        "CREATE VIEW source_ids AS SELECT * FROM prod.source_ids "
        "UNION ALL BY NAME SELECT * FROM pending_source_ids"
    )
    conn.execute("CREATE VIEW payouts AS SELECT * FROM prod.payouts")
    conn.execute("CREATE VIEW odds AS SELECT * FROM prod.odds")
    conn.execute("CREATE VIEW laps AS SELECT * FROM prod.laps")

    return r["race_id"]
