"""`P-3` Stage 1 レース質予測（`docs/spec/006-stage1-pace.md`）。

出走馬の構成からレースの想定ペース（`F-102`）を予測する。`D-007` の
2段階アーキテクチャの前段。対象レースの実測ラップは目的変数としてのみ
使い、入力には混入させない（原則5）。
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Protocol

import duckdb
import polars as pl

from umagic.features.f101 import compute_f101
from umagic.training import load_model, save_model

_KEY_COLS = ["race_id"]

# D-090: 対象レースの weather/track_condition は本命（当日・実測）の値を使う。
# 暫定（木曜）ルート用の入力差し替えは 016-inference.md（未作成）の責務。
_INPUTS_SQL = """
SELECT
    b.race_id,
    r.n_starters,
    r.distance,
    r.surface,
    r.direction,
    r.course,
    r.race_class,
    r.weather,
    CASE r.track_condition
        WHEN '良' THEN 0
        WHEN '稍重' THEN 1
        WHEN '重' THEN 2
        WHEN '不良' THEN 3
        ELSE NULL
    END AS track_condition
FROM base b
JOIN races r ON r.race_id = b.race_id
"""

# D-094 と同じ母集団（出走取消・競走除外は除く）。F-101 の集約対象。
_RUNNERS_SQL = """
SELECT race_id, horse_id FROM runners
WHERE race_id = ANY(?) AND status IN ('出走', '降着', '競走中止', '失格')
"""

_LAPS_SQL = """
SELECT l.race_id, l.furlong_no, l.lap_sec, r.distance
FROM laps l JOIN races r USING (race_id)
WHERE l.race_id = ANY(?)
"""

_INPUTS_SCHEMA = {
    "race_id": pl.Int64, "f101_min": pl.Float64, "f101_mean": pl.Float64,
    "f101_q25": pl.Float64, "f101_n_missing": pl.Int64, "n_starters": pl.Int64,
    "distance": pl.Int64, "surface": pl.Utf8, "direction": pl.Utf8, "course": pl.Utf8,
    "race_class": pl.Utf8, "weather": pl.Utf8, "track_condition": pl.Int64,
}

_TARGET_SCHEMA = {"race_id": pl.Int64, "f102_actual": pl.Float64, "n_laps": pl.Int64}


class Stage1Model(Protocol):
    """Stage 1 のモデル実装（`D-088`）。複数の実装を比較できる。"""

    def fit(self, x: pl.DataFrame, y: pl.Series, *, sample_weight: pl.Series | None,
            seed: int) -> None: ...
    def predict(self, x: pl.DataFrame) -> pl.Series: ...
    def save(self, out_dir: Path, meta: dict) -> None: ...
    @classmethod
    def load(cls, model_dir: Path) -> tuple["Stage1Model", dict]: ...


def build_target(conn: duckdb.DuckDBPyConnection, race_ids: list[int]) -> pl.DataFrame:
    """目的変数 `f102_actual` を作る（`D-087`）。

    `laps` が1行も無いレース、`N < 4`（前半が作れない）のレースは
    行に含めない（`D-091`）。
    """
    if not race_ids:
        return pl.DataFrame(schema=_TARGET_SCHEMA)

    df = conn.execute(_LAPS_SQL, [race_ids]).pl()
    if df.is_empty():
        return pl.DataFrame(schema=_TARGET_SCHEMA)

    rows: list[tuple] = []
    for key, group in df.sort(["race_id", "furlong_no"]).group_by("race_id", maintain_order=True):
        race_id = key[0] if isinstance(key, tuple) else key
        n = group.height
        if n < 4:
            continue
        laps = [float(v) for v in group["lap_sec"].to_list()]
        distance = float(group["distance"][0])
        agari_3f = laps[-3] + laps[-2] + laps[-1]
        zenhan_sum = sum(laps[:-3])
        zenhan_distance = distance - 600.0
        f102_actual = agari_3f / 3.0 - zenhan_sum / (zenhan_distance / 200.0)
        rows.append((race_id, f102_actual, n))

    if not rows:
        return pl.DataFrame(schema=_TARGET_SCHEMA)
    return pl.DataFrame(rows, schema=_TARGET_SCHEMA, orient="row")


def build_inputs(
    conn: duckdb.DuckDBPyConnection, race_ids: list[int], *, as_of: date,
) -> pl.DataFrame:
    """Stage 1 の入力を作る（2節）。1レース1行。"""
    if not race_ids:
        return pl.DataFrame(schema=_INPUTS_SCHEMA)

    base = pl.DataFrame({"race_id": race_ids})
    conn.register("base", base)
    try:
        race_level = conn.execute(_INPUTS_SQL).pl()
        runners = conn.execute(_RUNNERS_SQL, [race_ids]).pl()
    finally:
        conn.unregister("base")

    if runners.is_empty():
        f101_agg = pl.DataFrame(schema={
            "race_id": pl.Int64, "f101_min": pl.Float64, "f101_mean": pl.Float64,
            "f101_q25": pl.Float64, "f101_n_missing": pl.Int64,
        })
    else:
        f101_df = compute_f101(conn, runners, as_of=as_of)
        f101_agg = f101_df.group_by("race_id").agg(
            pl.col("f101").min().alias("f101_min"),
            pl.col("f101").mean().alias("f101_mean"),
            pl.col("f101").quantile(0.25, interpolation="linear").alias("f101_q25"),
            pl.col("f101").is_null().sum().cast(pl.Int64).alias("f101_n_missing"),
        )

    out = race_level.join(f101_agg, on="race_id", how="left")
    return out.select(list(_INPUTS_SCHEMA.keys()))


def predict_f102(
    model: Stage1Model, conn: duckdb.DuckDBPyConnection, race_ids: list[int], *, as_of: date,
) -> pl.DataFrame:
    """`F-104` に渡す形（`race_id, f102`）で予測する。`laps` の有無に依存しない（`D-091`）。"""
    x = build_inputs(conn, race_ids, as_of=as_of)
    if x.is_empty():
        return pl.DataFrame(schema={"race_id": pl.Int64, "f102": pl.Float64})
    preds = model.predict(x.drop("race_id"))
    return x.select("race_id").with_columns(pl.Series("f102", preds.to_list()))


# ---------------------------------------------------------------------------
# 既定のモデル実装（D-088: LightGBM 回帰）
# ---------------------------------------------------------------------------

_CATEGORICAL_COLS = ["surface", "direction", "course", "race_class", "weather"]
_NUMERIC_COLS = [
    "f101_min", "f101_mean", "f101_q25", "f101_n_missing",
    "n_starters", "distance", "track_condition",
]
FEATURE_COLS = _NUMERIC_COLS + _CATEGORICAL_COLS


class LightGBMStage1Model:
    """`Stage1Model` の既定実装（`D-088`）。カテゴリ列は自前で整数コード化する

    （`categorical_feature` に渡すため。`polars` → `numpy` の変換は
    `D-042` が許す「LightGBM に渡すときだけ numpy に落とす」に沿う）。
    """

    def __init__(self) -> None:
        self.booster = None
        self._category_maps: dict[str, list[str]] = {}

    def _encode(self, x: pl.DataFrame, *, fit: bool) -> pl.DataFrame:
        out = x.select(FEATURE_COLS)
        for col in _CATEGORICAL_COLS:
            if fit:
                cats = sorted(out[col].drop_nulls().unique().to_list())
                self._category_maps[col] = cats
            mapping = {c: i for i, c in enumerate(self._category_maps.get(col, []))}
            out = out.with_columns(
                pl.col(col)
                .map_elements(lambda v, m=mapping: m.get(v, -1), return_dtype=pl.Int32)
                .alias(col)
            )
        return out.select(FEATURE_COLS)

    def fit(
        self, x: pl.DataFrame, y: pl.Series, *, sample_weight: pl.Series | None, seed: int,
    ) -> None:
        import lightgbm as lgb

        enc = self._encode(x, fit=True)
        cat_idx = [enc.columns.index(c) for c in _CATEGORICAL_COLS]
        ds = lgb.Dataset(
            enc.to_numpy(), label=y.to_numpy(),
            weight=sample_weight.to_numpy() if sample_weight is not None else None,
            categorical_feature=cat_idx, free_raw_data=False,
        )
        params = {
            "objective": "regression", "verbose": -1,
            "seed": seed, "bagging_seed": seed, "feature_fraction_seed": seed,
            "deterministic": True, "force_row_wise": True, "num_threads": 1,
            "min_data_in_leaf": 1,
        }
        self.booster = lgb.train(params, ds, num_boost_round=50)

    def predict(self, x: pl.DataFrame) -> pl.Series:
        enc = self._encode(x, fit=False)
        return pl.Series(self.booster.predict(enc.to_numpy()))

    def save(self, out_dir: Path, meta: dict) -> None:
        full_meta = dict(meta)
        full_meta.setdefault("stage", 1)
        full_meta.setdefault("model_kind", "lightgbm_regression")
        full_meta["feature_names"] = FEATURE_COLS
        full_meta["category_maps"] = self._category_maps
        save_model(self.booster, full_meta, out_dir)

    @classmethod
    def load(cls, model_dir: Path) -> tuple["LightGBMStage1Model", dict]:
        booster, meta = load_model(model_dir)
        model = cls()
        model.booster = booster
        model._category_maps = meta.get("category_maps", {})
        return model, meta
