"""`004-leakage-test.md` の検査9件。合成データのみを使い、CIで常時実行する（`D-053`）。

各検査は「正しい実装なら通る」テストと「欠陥を仕込むと落ちる」テストの
対で書く。後者が無いと、検査そのものが機能しているかを確認できない
（`004` の「テスト観点」）。

`F-xxx` の実装がまだ無いため、各原則を体現する最小の probe 関数
（`FeatureFn` 互換）を都度定義する。特徴量を実装するたびに、その
`F-xxx` を probe と置き換えて同じ検査に通すことで骨組みを再利用する。

`004` の「テスト観点」10件との対応:

| # | 仕込む欠陥 | 本ファイルの欠陥注入テスト |
|---|---|---|
| 1 | 過去集計を `<=` | `test_fault_past_aggregation_boundary_detected` |
| 2 | 同日判定から `course` を外す | `test_fault_same_day_no_course_detected` |
| 3 | 同日判定を `<=` | `test_fault_same_day_le_detected` |
| 4 | 同日以降の成績混入 | `test_fault_future_form_detected` |
| 5 | 対象レースの `time_sec` | `test_fault_target_race_outcome_detected` |
| 6 | 対象レースの `odds_win` | `test_fault_target_race_odds_detected` |
| 7 | `μ_global` を全期間で推定 | `test_fault_as_of_recomputation_detected` |
| 8 | Stage 1 を全期間で学習 | `test_fault_stage1_full_period_training_detected` |
| 9 | 暫定経路への当日特徴量混入 | `test_fault_deadline_violation_detected` |
| 10 | 封印セットのG1を含める | `test_fault_sealed_set_read_detected` |

**#8 は SQL の probe では代替できない**（モデル学習そのものを模擬する
必要がある）ため、`umagic.stage1` を直接使う。`orchestration.py` の
`stage1_fit_full()` が実際に行っているとおり、学習に渡す `race_ids` を
fold の学習期間（`train_start`〜`train_end`）に絞ることが対策であり、
絞り忘れると「学習期間より後のレース」の情報がモデルのパラメータに
混入する。
"""

from __future__ import annotations

from datetime import date, datetime, timezone

import polars as pl
import pytest

from tests.fixtures.leakage_fixture import (
    AS_OF_LATE,
    AS_OF_MID,
    TARGET_COURSE,
    TARGET_DATE,
    TARGET_RACE_ID,
    TARGET_RACE_NUMBER,
    build_leakage_fixture_conn,
)
from umagic.features.build import build_features
from umagic.features.registry import FeatureRegistry, FeatureSpec
from umagic.sealed import is_sealed

FAR_FUTURE = date(2030, 1, 1)


def _empty(schema: dict) -> pl.DataFrame:
    return pl.DataFrame(schema=schema)


# ---------------------------------------------------------------------------
# 原則1: 発走時点で確定している情報のみ
# ---------------------------------------------------------------------------
# 特徴量列に、生の着順・タイム列と同名の列を使わない（命名レベルの防御）。
# 実装が誤って未加工の列をそのまま結合すると、この検査が拾う。

_FORBIDDEN_RAW_COLUMNS = {
    "finish_pos", "time_sec", "margin", "last_3f", "corners",
    "odds_win", "popularity", "status",
}


def _probe_p1(leaky: bool):
    def fn(conn, base, *, as_of):
        if not leaky:
            return _empty({"race_id": pl.Int64, "horse_id": pl.Int64})
        # 生の着順列をそのまま特徴量列名に使ってしまう典型的な誤り
        return base.with_columns(pl.lit(999.0).alias("time_sec"))
    return fn


def test_no_future_columns():
    conn = build_leakage_fixture_conn()
    df = build_features(conn, as_of=FAR_FUTURE, race_ids=[TARGET_RACE_ID],
                        feature_fns=[_probe_p1(leaky=False)])
    assert not (_FORBIDDEN_RAW_COLUMNS & set(df.columns))


def test_fault_no_future_columns_detected():
    conn = build_leakage_fixture_conn()
    df = build_features(conn, as_of=FAR_FUTURE, race_ids=[TARGET_RACE_ID],
                        feature_fns=[_probe_p1(leaky=True)])
    assert _FORBIDDEN_RAW_COLUMNS & set(df.columns)


# ---------------------------------------------------------------------------
# 原則2: 過去成績の集計は race_date < target_race_date で厳密にフィルタする
# ---------------------------------------------------------------------------
# horse_id=100 は対象レース（2023-01-07）より前に3走を持つ。
# `<=` にすると対象レース自身が1件多く数えられる。

def _probe_p2(leaky: bool):
    def fn(conn, base, *, as_of):
        op = "<=" if leaky else "<"
        rows = conn.execute(
            f"""
            SELECT ? AS race_id, 100 AS horse_id, COUNT(*) AS past_n
            FROM runners ru
            JOIN races r USING (race_id)
            JOIN races target ON target.race_id = ?
            WHERE ru.horse_id = 100
              AND r.date {op} target.date
            """,
            [TARGET_RACE_ID, TARGET_RACE_ID],
        ).fetchall()
        return pl.DataFrame(rows, schema=["race_id", "horse_id", "past_n"], orient="row")
    return fn


def test_past_aggregation_is_strict():
    conn = build_leakage_fixture_conn()
    df = build_features(conn, as_of=FAR_FUTURE, race_ids=[TARGET_RACE_ID],
                        feature_fns=[_probe_p2(leaky=False)])
    assert df.filter(pl.col("horse_id") == 100)["past_n"].to_list() == [3]


def test_fault_past_aggregation_boundary_detected():
    conn = build_leakage_fixture_conn()
    df = build_features(conn, as_of=FAR_FUTURE, race_ids=[TARGET_RACE_ID],
                        feature_fns=[_probe_p2(leaky=True)])
    # 対象レース自身が1件多く数えられ、3 と一致しなくなる
    assert df.filter(pl.col("horse_id") == 100)["past_n"].to_list() != [3]


# ---------------------------------------------------------------------------
# 原則3: 同日開催の別レースは F-501 / F-502 のみ例外。判定は厳密不等号（D-010）
# ---------------------------------------------------------------------------
# 対象: 2023-01-07 東京 R3。先行する同日・同競馬場は R1, R2（計4行）。
# `<=` にすると対象自身の2行が、course を外すと同日・中山 R1 の1行が混入する。

def _probe_p3(leaky: str | None):
    """leaky: None（正しい）/ 'le'（<= にする）/ 'no_course'（course を外す）。"""
    def fn(conn, base, *, as_of):
        op = "<=" if leaky == "le" else "<"
        course_clause = "" if leaky == "no_course" else "AND r.course = target.course"
        rows = conn.execute(
            f"""
            SELECT target.race_id, 100 AS horse_id, COUNT(*) AS n
            FROM races target
            JOIN races r ON r.date = target.date {course_clause}
                          AND r.race_number {op} target.race_number
            JOIN runners ru ON ru.race_id = r.race_id
            WHERE target.race_id = ?
            GROUP BY target.race_id
            """,
            [TARGET_RACE_ID],
        ).fetchall()
        return pl.DataFrame(rows, schema=["race_id", "horse_id", "n"], orient="row")
    return fn


def test_same_day_uses_strict_race_number():
    conn = build_leakage_fixture_conn()
    df = build_features(conn, as_of=FAR_FUTURE, race_ids=[TARGET_RACE_ID],
                        feature_fns=[_probe_p3(leaky=None)])
    assert df.filter(pl.col("horse_id") == 100)["n"].to_list() == [4]  # R1 + R2 の各2頭


def test_fault_same_day_le_detected():
    conn = build_leakage_fixture_conn()
    df = build_features(conn, as_of=FAR_FUTURE, race_ids=[TARGET_RACE_ID],
                        feature_fns=[_probe_p3(leaky="le")])
    assert df.filter(pl.col("horse_id") == 100)["n"].to_list() != [4]  # 対象レース自身の2頭が混入


def test_fault_same_day_no_course_detected():
    conn = build_leakage_fixture_conn()
    df = build_features(conn, as_of=FAR_FUTURE, race_ids=[TARGET_RACE_ID],
                        feature_fns=[_probe_p3(leaky="no_course")])
    assert df.filter(pl.col("horse_id") == 100)["n"].to_list() != [4]  # 中山 R1 の1頭が混入


# ---------------------------------------------------------------------------
# 原則4: 「その馬のその後の成績」に由来する集計特徴量は禁止
# ---------------------------------------------------------------------------
# horse_id=100 は対象レースより後にも3走を持つ（2023-03 / 05, 2024-05）。
# 日付フィルタを丸ごと落とすと、未来の3走が混入する。

def _probe_p4(leaky: bool):
    def fn(conn, base, *, as_of):
        if leaky:
            sql = """
                SELECT ? AS race_id, 100 AS horse_id, AVG(finish_pos) AS avg_pos
                FROM runners WHERE horse_id = 100 AND finish_pos IS NOT NULL
            """
            rows = conn.execute(sql, [TARGET_RACE_ID]).fetchall()
        else:
            sql = """
                SELECT target.race_id, 100 AS horse_id, AVG(ru.finish_pos) AS avg_pos
                FROM races target
                JOIN runners ru ON ru.horse_id = 100
                JOIN races r ON r.race_id = ru.race_id AND r.date < target.date
                WHERE target.race_id = ? AND ru.finish_pos IS NOT NULL
                GROUP BY target.race_id
            """
            rows = conn.execute(sql, [TARGET_RACE_ID]).fetchall()
        return pl.DataFrame(rows, schema=["race_id", "horse_id", "avg_pos"], orient="row")
    return fn


def test_no_future_form_of_same_horse():
    conn = build_leakage_fixture_conn()
    df = build_features(conn, as_of=FAR_FUTURE, race_ids=[TARGET_RACE_ID],
                        feature_fns=[_probe_p4(leaky=False)])
    # 過去3走（着順 2, 1, 3）の平均
    assert df.filter(pl.col("horse_id") == 100)["avg_pos"].to_list() == [(2 + 1 + 3) / 3]


def test_fault_future_form_detected():
    conn = build_leakage_fixture_conn()
    df = build_features(conn, as_of=FAR_FUTURE, race_ids=[TARGET_RACE_ID],
                        feature_fns=[_probe_p4(leaky=True)])
    assert df.filter(pl.col("horse_id") == 100)["avg_pos"].to_list() != [(2 + 1 + 3) / 3]


# ---------------------------------------------------------------------------
# 原則5: 対象レースの実測ラップ・走破タイム・着順は特徴量に含めない
# ---------------------------------------------------------------------------

def _probe_p5(leaky: bool):
    def fn(conn, base, *, as_of):
        col = "ru.time_sec" if leaky else "-1.0"
        rows = conn.execute(
            f"SELECT race_id, horse_id, {col} AS leaked FROM runners ru "
            f"WHERE race_id = ? AND horse_id = 100",
            [TARGET_RACE_ID],
        ).fetchall()
        return pl.DataFrame(rows, schema=["race_id", "horse_id", "leaked"], orient="row")
    return fn


def test_target_race_outcome_excluded():
    conn = build_leakage_fixture_conn()
    target_time = conn.execute(
        "SELECT time_sec FROM runners WHERE race_id=? AND horse_id=100", [TARGET_RACE_ID],
    ).fetchone()[0]
    df = build_features(conn, as_of=FAR_FUTURE, race_ids=[TARGET_RACE_ID],
                        feature_fns=[_probe_p5(leaky=False)])
    assert df.filter(pl.col("horse_id") == 100)["leaked"].to_list() != [target_time]


def test_fault_target_race_outcome_detected():
    conn = build_leakage_fixture_conn()
    target_time = conn.execute(
        "SELECT time_sec FROM runners WHERE race_id=? AND horse_id=100", [TARGET_RACE_ID],
    ).fetchone()[0]
    df = build_features(conn, as_of=FAR_FUTURE, race_ids=[TARGET_RACE_ID],
                        feature_fns=[_probe_p5(leaky=True)])
    assert df.filter(pl.col("horse_id") == 100)["leaked"].to_list() == [target_time]


# ---------------------------------------------------------------------------
# 原則6: オッズは予測に使わない。過去オッズは F-701 の範囲に限定（D-002 / R-018）
# ---------------------------------------------------------------------------
# horse_id=100 の過去オッズ平均 = mean(4.0, 2.0, 8.0) = 4.6667（対象は3.5、含めない）

def _probe_p6(leaky: bool):
    def fn(conn, base, *, as_of):
        if leaky:
            sql = """
                SELECT ? AS race_id, 100 AS horse_id, AVG(ru.odds_win) AS avg_odds
                FROM runners ru JOIN races r USING (race_id)
                WHERE ru.horse_id = 100 AND r.date <= ? AND ru.odds_win IS NOT NULL
            """
            rows = conn.execute(sql, [TARGET_RACE_ID, TARGET_DATE]).fetchall()
        else:
            sql = """
                SELECT ? AS race_id, 100 AS horse_id, AVG(ru.odds_win) AS avg_odds
                FROM runners ru JOIN races r USING (race_id)
                WHERE ru.horse_id = 100 AND r.date < ? AND ru.odds_win IS NOT NULL
            """
            rows = conn.execute(sql, [TARGET_RACE_ID, TARGET_DATE]).fetchall()
        return pl.DataFrame(rows, schema=["race_id", "horse_id", "avg_odds"], orient="row")
    return fn


def test_target_race_odds_excluded():
    conn = build_leakage_fixture_conn()
    df = build_features(conn, as_of=FAR_FUTURE, race_ids=[TARGET_RACE_ID],
                        feature_fns=[_probe_p6(leaky=False)])
    expected = (4.0 + 2.0 + 8.0) / 3
    assert abs(df.filter(pl.col("horse_id") == 100)["avg_odds"].to_list()[0] - expected) < 1e-9


def test_fault_target_race_odds_detected():
    conn = build_leakage_fixture_conn()
    df = build_features(conn, as_of=FAR_FUTURE, race_ids=[TARGET_RACE_ID],
                        feature_fns=[_probe_p6(leaky=True)])
    expected = (4.0 + 2.0 + 8.0) / 3  # 対象自身の 3.5 が混ざるとこれと一致しなくなる
    assert abs(df.filter(pl.col("horse_id") == 100)["avg_odds"].to_list()[0] - expected) > 1e-9


# ---------------------------------------------------------------------------
# 原則7: 集計統計量の推定期間も as-of で切る（中核・D-054）
# ---------------------------------------------------------------------------
# 正しい実装は「行ごとの対象レース日付」で母集団を切る（結果として call の
# as_of には依存しない）。誤った実装は「call の as_of」だけで切った単一の
# 定数を全行に適用し、call をまたぐと値が変わってしまう。

def _probe_p7(leaky: bool):
    def fn(conn, base, *, as_of):
        if leaky:
            mu = conn.execute(
                "SELECT AVG(ru.time_sec) FROM runners ru JOIN races r USING (race_id) "
                "WHERE r.date < ?", [as_of],
            ).fetchone()[0]
            return base.with_columns(pl.lit(mu).alias("mu_global"))

        race_dates = dict(conn.execute("SELECT race_id, date FROM races").fetchall())
        rows = []
        for race_id, horse_id in base.select(["race_id", "horse_id"]).iter_rows():
            d = race_dates[race_id]
            mu = conn.execute(
                "SELECT AVG(ru.time_sec) FROM runners ru JOIN races r USING (race_id) "
                "WHERE r.date < ?", [d],
            ).fetchone()[0]
            rows.append((race_id, horse_id, mu))
        return pl.DataFrame(rows, schema=["race_id", "horse_id", "mu_global"], orient="row")
    return fn


def test_as_of_recomputation_invariance():
    conn = build_leakage_fixture_conn()
    key = ["race_id", "horse_id"]

    x1 = build_features(conn, as_of=AS_OF_MID, feature_fns=[_probe_p7(leaky=False)])
    x2 = build_features(conn, as_of=AS_OF_LATE, feature_fns=[_probe_p7(leaky=False)])
    overlap = x2.join(x1.select(key), on=key, how="semi")

    assert x1.sort(key).equals(overlap.sort(key))


def test_fault_as_of_recomputation_detected():
    conn = build_leakage_fixture_conn()
    key = ["race_id", "horse_id"]

    x1 = build_features(conn, as_of=AS_OF_MID, feature_fns=[_probe_p7(leaky=True)])
    x2 = build_features(conn, as_of=AS_OF_LATE, feature_fns=[_probe_p7(leaky=True)])
    overlap = x2.join(x1.select(key), on=key, how="semi")

    assert not x1.sort(key).equals(overlap.sort(key))


# ---------------------------------------------------------------------------
# R-028: 経路の締切より後に確定する特徴量が使われない
# ---------------------------------------------------------------------------

def test_route_respects_deadline():
    reg = FeatureRegistry()
    reg.register(FeatureSpec("F-101", ("f101",), timing="木曜"))
    reg.register(FeatureSpec("F-501", ("f501",), timing="当日", minutes_before_post=30))

    provisional_cols = reg.columns_for("暫定")
    assert "f501" not in provisional_cols
    assert "f101" in provisional_cols


def test_fault_deadline_violation_detected():
    """`F-501` を誤って `木曜` として登録すると、暫定経路に混入する。"""
    reg = FeatureRegistry()
    reg.register(FeatureSpec("F-501", ("f501",), timing="木曜"))  # 誤り。本来は当日
    assert "f501" in reg.columns_for("暫定")  # 誤登録がそのまま混入することを示す


# ---------------------------------------------------------------------------
# D-017 / D-056: 封印セットに触れない
# ---------------------------------------------------------------------------

def _dev_race_ids(conn, today: date, *, respect_seal: bool) -> list[int]:
    rows = conn.execute("SELECT race_id, date, grade FROM races").fetchall()
    return [
        rid for rid, d, grade in rows
        if not (respect_seal and is_sealed(d, grade, today=today))
    ]


def test_sealed_set_not_read():
    conn = build_leakage_fixture_conn()
    ids = _dev_race_ids(conn, today=TARGET_DATE, respect_seal=True)
    assert TARGET_RACE_ID not in ids  # G1・直近3年以内 → 封印対象


def test_fault_sealed_set_read_detected():
    conn = build_leakage_fixture_conn()
    ids = _dev_race_ids(conn, today=TARGET_DATE, respect_seal=False)
    assert TARGET_RACE_ID in ids  # 封印チェックを忘れると読めてしまう


def test_sealed_set_excludes_non_g1():
    """D-003: 学習データ（全レース）は封印対象ではない。G1以外は封印されない。"""
    conn = build_leakage_fixture_conn()
    # 20230101001 は grade=NULL（非G1）
    assert not is_sealed(TARGET_DATE, None, today=TARGET_DATE)


# ---------------------------------------------------------------------------
# 欠陥注入 #8（原則7 / D-054）: Stage 1 を全期間で学習
# ---------------------------------------------------------------------------

_STAGE1_FETCHED_AT = datetime(2026, 8, 22, tzinfo=timezone.utc)


def _insert_stage1_race(
    conn, race_id: int, race_date: date, *, distance: int, n_starters: int, laps: list[float],
) -> None:
    conn.execute(
        "INSERT INTO races (race_id, date, course, race_number, distance, surface, "
        "n_entries, n_starters, source, fetched_at) VALUES "
        "(?, ?, '東京', 1, ?, '芝', ?, ?, 'netkeiba_jra', ?)",
        [race_id, race_date, distance, n_starters, n_starters, _STAGE1_FETCHED_AT],
    )
    for i, lap in enumerate(laps, start=1):
        conn.execute(
            "INSERT INTO laps VALUES (?, ?, ?, 'netkeiba_jra', ?)",
            [race_id, i, lap, _STAGE1_FETCHED_AT],
        )
    for slot in range(n_starters):
        horse_id = race_id * 100 + slot
        if not conn.execute("SELECT 1 FROM horses WHERE horse_id=?", [horse_id]).fetchone():
            conn.execute(
                "INSERT INTO horses VALUES (?, ?, NULL, NULL, NULL, NULL, 'netkeiba_jra', ?)",
                [horse_id, f"馬{horse_id}", _STAGE1_FETCHED_AT],
            )
        conn.execute(
            "INSERT INTO runners (race_id, horse_id, number, status, finish_pos, corners, "
            "source, fetched_at) VALUES (?, ?, ?, '出走', ?, [1,2,3,4], 'netkeiba_jra', ?)",
            [race_id, horse_id, slot + 1, slot + 1, _STAGE1_FETCHED_AT],
        )


def test_fault_stage1_full_period_training_detected():
    """`race_ids` を fold の学習期間に絞らずに Stage 1 を学習すると、

    学習期間より後（=未来）のレースの情報がモデルに混入し、それが
    予測に現れることを示す。`distance=3600m`（他に存在しない値）の
    未来レースだけに極端な `f102_actual` を持たせ、同じ `distance` の
    別レース（学習期間内には存在しない、予測専用のクエリ）への予測が
    「未来レースを学習に含めたかどうか」で変わることを確認する。
    """
    from umagic.stage1 import LightGBMStage1Model, build_inputs, build_target

    conn = build_leakage_fixture_conn()

    # 学習期間（fold.train_start〜train_end に相当）: 通常の距離・通常のラップ
    train_ids = []
    for i in range(6):
        rid = 90_000_000 + i
        _insert_stage1_race(
            conn, rid, date(2015, 1, 1 + i), distance=2000, n_starters=2,
            laps=[12.0, 11.8, 12.1, 11.9],
        )
        train_ids.append(rid)

    # 未来レース（fold.valid_start 以降に相当）。他に存在しない distance=3600m と、
    # 極端な f102_actual になるラップ（前半が極端に遅く、上がりが極端に速い）を持たせる
    future_id = 90_000_900
    _insert_stage1_race(
        conn, future_id, date(2024, 1, 1), distance=3600, n_starters=2,
        laps=[30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0,
              30.0, 30.0, 30.0, 3.0, 3.0, 3.0],
    )

    # 予測専用のクエリレース: distance=3600m だが学習期間・未来レースのどちらにも
    # 含まれない、まったく別の race_id（実運用の「まだ結果が無い対象レース」に相当）
    query_id = 90_000_999
    _insert_stage1_race(
        conn, query_id, date(2018, 6, 1), distance=3600, n_starters=2, laps=[],
    )

    as_of = date(2018, 1, 1)

    def _fit_and_predict(race_ids_for_training: list[int]) -> float:
        target = build_target(conn, race_ids_for_training)
        x = build_inputs(conn, target["race_id"].to_list(), as_of=as_of)
        merged = x.join(target.select(["race_id", "f102_actual"]), on="race_id", how="inner")
        model = LightGBMStage1Model()
        model.fit(
            merged.drop(["race_id", "f102_actual"]), merged["f102_actual"],
            sample_weight=None, seed=1,
        )
        query_x = build_inputs(conn, [query_id], as_of=as_of)
        return float(model.predict(query_x.drop("race_id"))[0])

    # 正しい実装: 学習期間のみ（未来レースを含めない）
    correct_pred = _fit_and_predict(train_ids)

    # 欠陥注入: 「fold で絞る」を忘れ、未来レースまで学習に含める
    leaky_pred = _fit_and_predict(train_ids + [future_id])

    assert correct_pred != pytest.approx(leaky_pred, abs=1e-6)
