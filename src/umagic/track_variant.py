"""`F-301` 馬場差推定（`docs/spec/013-track-variant.md` / `D-104`〜`D-107`, `D-111`）。

走破タイムから、レース効果（馬場差 + 展開 + `γ`/`δ` で説明されない
メンバーレベル）と馬効果（その馬の真の能力）を交互反復で分離する
（`D-104`）。`statsmodels` の `MixedLM` も `PyMC` も使わない（`D-104` の
根拠を参照）。

目的変数は `(surface, distance)` ごとに標準化する（`D-105`）。距離帯・
クラス・グレード（`G1`/`G2`/`G3`/`L`/無し、`D-111`）の固定効果を残し、
レース単位の共変量を縮約前に吸収する。`race_class` は `G1`〜`G3` を
含む全グレードを `'オープン'` 1水準に丸めるため、`grade` を独立の
固定効果として追加しないとメンバーレベルの実力差がレース効果 `u_j`
に漏れる（`Q-042` の実測で確認）。

`as_of` 再推定の粒度は `training.cross_fit_blocks()` の4ブロックに揃える
（`D-106`）。ここでは1回ぶんの推定（`fit_track_variant`）のみを実装し、
fold あたり5回のオーケストレーションは `orchestration.py` の責務とする
（`stage1_fit_all()` と対称の設計）。

**学習は全レースで行う**（`D-003`）。G1だけを取り出して呼び出さないこと。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import duckdb
import numpy as np
import polars as pl

MIN_SD = 1e-3        # 013-track-variant.md 2節: 条件別標準偏差の下限
MIN_VAR = 1e-8        # 013-track-variant.md 4節: 分散成分の下限（k の発散を防ぐ、D-102 と同種）
DEFAULT_MAX_ITER = 100
DEFAULT_TOL = 1e-6

_DISTANCE_BAND_CASE = """
    CASE
        WHEN r.distance <= 1400 THEN '短距離'
        WHEN r.distance <= 1800 THEN 'マイル'
        WHEN r.distance <= 2200 THEN '中距離'
        ELSE '長距離'
    END
"""

_POPULATION_SQL = f"""
SELECT ru.race_id, ru.horse_id, CAST(ru.time_sec AS DOUBLE) AS time_sec,
       r.surface, r.distance, {_DISTANCE_BAND_CASE} AS distance_band,
       COALESCE(r.race_class, '不明') AS race_class,
       COALESCE(r.grade, '無し') AS grade
FROM runners ru
JOIN races r ON r.race_id = ru.race_id
WHERE ru.status IN ('出走', '降着', '失格')
  AND ru.time_sec IS NOT NULL
  AND ru.race_id = ANY(?)
"""

_EMPTY_EFFECTS_SCHEMA = {"horse_id": pl.Int64, "effect": pl.Float64}
_EMPTY_RACE_EFFECTS_SCHEMA = {"race_id": pl.Int64, "effect": pl.Float64}
_EMPTY_SCALE_SCHEMA = {"surface": pl.Utf8, "distance": pl.Int64, "sd": pl.Float64}


@dataclass(frozen=True)
class VariantFit:
    """1回の推定結果。`as_of` 時点で利用可能なデータのみから推定したもの。"""

    as_of: date
    horse_effects: pl.DataFrame  # horse_id: Int64, effect: Float64
    race_effects: pl.DataFrame   # race_id: Int64, effect: Float64
    n_rows: int
    n_iter: int
    converged: bool
    sigma2_error: float
    sigma2_race: float
    sigma2_horse: float
    scale: pl.DataFrame          # surface, distance, sd
    main_component: frozenset    # race_id の集合。識別可能な範囲


def _empty_fit(as_of: date) -> VariantFit:
    nan = float("nan")
    return VariantFit(
        as_of=as_of,
        horse_effects=pl.DataFrame(schema=_EMPTY_EFFECTS_SCHEMA),
        race_effects=pl.DataFrame(schema=_EMPTY_RACE_EFFECTS_SCHEMA),
        n_rows=0, n_iter=0, converged=False,
        sigma2_error=nan, sigma2_race=nan, sigma2_horse=nan,
        scale=pl.DataFrame(schema=_EMPTY_SCALE_SCHEMA),
        main_component=frozenset(),
    )


def _compute_scale(pop: pl.DataFrame) -> pl.DataFrame:
    """`(surface, distance)` ごとの、レース内偏差の標準偏差を返す（`D-105`）。

    レース内平均を引いた残差の標本標準偏差（`ddof=1`）。標本が1件しか
    無い組み合わせは `sd` が `null` になる。
    """
    race_mean = pop.group_by("race_id").agg(pl.col("time_sec").mean().alias("race_mean"))
    dev = pop.join(race_mean, on="race_id", how="left").with_columns(
        (pl.col("time_sec") - pl.col("race_mean")).alias("dev")
    )
    return (
        dev.group_by(["surface", "distance"])
        .agg(pl.col("dev").std().alias("sd"))
        .sort(["surface", "distance"])
    )


def _design_matrix(pop: pl.DataFrame) -> tuple[np.ndarray, list[str]]:
    """固定効果（`(surface, distance_band)` 交互作用 + `race_class` + `grade`）の設計行列。

    `grade`（`D-111`）: `race_class` は G1〜G3・L・格付なしのオープン特別を
    すべて `'オープン'` 1水準に丸めており、グレード間の実力差を区別しない。
    実データで `(surface, distance_band)` を揃えても `grade` 別のレース効果
    残差が有意かつ2つの独立5年間で再現する順序（`G1` が最も速い方向、
    `G2` が最も遅い方向）を持つことを確認した（`Q-042`）ため、`grade` を
    独立の固定効果因子として追加する。

    各因子の水準は実データに現れたものだけを使い、辞書式で最小の水準を
    基準（ダミー変数を作らない）として落とす。切片列を含む。水準の順序は
    ソート済みで決定的（`R-021`）。
    """
    n = pop.height
    factors = {
        "sd": (pop["surface"] + "_" + pop["distance_band"]).to_list(),
        "rc": pop["race_class"].to_list(),
        "gr": pop["grade"].to_list(),
    }

    cols = ["_intercept"]
    levels: dict[str, list[str]] = {}
    for prefix, values in factors.items():
        lv = sorted(set(values))
        levels[prefix] = lv
        cols += [f"{prefix}:{v}" for v in lv[1:]]  # 先頭水準を基準として落とす

    X = np.zeros((n, len(cols)), dtype=np.float64)
    X[:, 0] = 1.0
    offset = 1
    for prefix, values in factors.items():
        lv = levels[prefix]
        idx = {v: i for i, v in enumerate(lv)}
        for row, v in enumerate(values):
            i = idx[v]
            if i > 0:
                X[row, offset + (i - 1)] = 1.0
        offset += len(lv) - 1
    return X, cols


def _main_component(race_ids: np.ndarray, horse_ids: np.ndarray) -> frozenset:
    """馬とレースの二部グラフの最大連結成分に属する `race_id` を返す（013 5節）。

    辺は「同じ馬が両方のレースに出た」。Union-Find で `race_id` の成分を求める。
    """
    uniq_races = np.unique(race_ids)
    r_index = {int(r): i for i, r in enumerate(uniq_races)}
    parent = list(range(len(uniq_races)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    by_horse: dict[int, list[int]] = {}
    for rid, hid in zip(race_ids.tolist(), horse_ids.tolist()):
        by_horse.setdefault(hid, []).append(r_index[rid])
    for idxs in by_horse.values():
        first = idxs[0]
        for i in idxs[1:]:
            union(first, i)

    comp_size: dict[int, int] = {}
    for i in range(len(uniq_races)):
        comp_size[find(i)] = comp_size.get(find(i), 0) + 1
    if not comp_size:
        return frozenset()
    main_root = max(comp_size, key=comp_size.get)
    return frozenset(int(uniq_races[i]) for i in range(len(uniq_races)) if find(i) == main_root)


def fit_track_variant(
    conn: duckdb.DuckDBPyConnection,
    race_ids: list[int],
    *,
    as_of: date,
    max_iter: int = DEFAULT_MAX_ITER,
    tol: float = DEFAULT_TOL,
) -> VariantFit:
    """`race_ids` の出走行から `F-301` を推定する。

    `race_ids` はすべて `as_of` より前のレースであること（呼び出し側の
    責務。本関数は日付フィルタを行わない、013 5節）。
    """
    if not race_ids:
        return _empty_fit(as_of)

    pop = conn.execute(_POPULATION_SQL, [race_ids]).pl()
    if pop.is_empty():
        return _empty_fit(as_of)

    scale = _compute_scale(pop)
    usable_scale = scale.filter(pl.col("sd").is_not_null() & (pl.col("sd") >= MIN_SD))
    pop = pop.join(
        usable_scale.select(["surface", "distance", pl.col("sd")]),
        on=["surface", "distance"], how="inner",
    )
    if pop.is_empty():
        return VariantFit(
            as_of=as_of, horse_effects=pl.DataFrame(schema=_EMPTY_EFFECTS_SCHEMA),
            race_effects=pl.DataFrame(schema=_EMPTY_RACE_EFFECTS_SCHEMA),
            n_rows=0, n_iter=0, converged=False,
            sigma2_error=float("nan"), sigma2_race=float("nan"), sigma2_horse=float("nan"),
            scale=scale, main_component=frozenset(),
        )

    pop = pop.with_columns((pl.col("time_sec") / pl.col("sd")).alias("y")).sort(
        ["race_id", "horse_id"]
    )

    race_ids_arr = pop["race_id"].to_numpy()
    horse_ids_arr = pop["horse_id"].to_numpy()
    y = pop["y"].to_numpy().astype(np.float64)

    uniq_races = np.unique(race_ids_arr)
    uniq_horses = np.unique(horse_ids_arr)
    n_race, n_horse = len(uniq_races), len(uniq_horses)
    race_code = np.searchsorted(uniq_races, race_ids_arr)
    horse_code = np.searchsorted(uniq_horses, horse_ids_arr)
    n_race_count = np.bincount(race_code, minlength=n_race).astype(np.float64)
    n_horse_count = np.bincount(horse_code, minlength=n_horse).astype(np.float64)

    X, _ = _design_matrix(pop)
    pinv_X = np.linalg.pinv(X)

    theta = pinv_X @ y
    u = np.zeros(n_race, dtype=np.float64)
    v = np.zeros(n_horse, dtype=np.float64)
    k_race = 1.0
    k_horse = 1.0

    converged = False
    n_iter = 0
    for it in range(1, max_iter + 1):
        n_iter = it
        fitted = X @ theta

        r1 = y - fitted - v[horse_code]
        sum_u = np.bincount(race_code, weights=r1, minlength=n_race)
        new_u = sum_u / (n_race_count + k_race)

        r2 = y - fitted - new_u[race_code]
        sum_v = np.bincount(horse_code, weights=r2, minlength=n_horse)
        new_v = sum_v / (n_horse_count + k_horse)

        # 中心化: 交差ランダム効果は加法定数の分だけ不定なので、平均0に揃える
        # （013 4節）。差分は次の θ 再フィットが自動的に吸収する。
        new_u = new_u - new_u.mean()
        new_v = new_v - new_v.mean()

        theta = pinv_X @ (y - new_u[race_code] - new_v[horse_code])
        fitted = X @ theta
        eps = y - fitted - new_u[race_code] - new_v[horse_code]

        sigma2_error = float(np.var(eps))
        sigma2_race = float(np.var(new_u)) if n_race > 1 else 0.0
        sigma2_horse = float(np.var(new_v)) if n_horse > 1 else 0.0
        k_race = sigma2_error / max(sigma2_race, MIN_VAR)
        k_horse = sigma2_error / max(sigma2_horse, MIN_VAR)

        delta = max(
            float(np.max(np.abs(new_u - u))) if n_race else 0.0,
            float(np.max(np.abs(new_v - v))) if n_horse else 0.0,
        )
        # 収束判定は相対誤差（D-108）。実データでは効果の尺度が
        # (surface, distance) の標準化後でも数十単位に達することがあり
        # （少標本の馬が極端な走破タイムを1回引いた場合など）、絶対
        # tol=1e-6 は非現実的に厳しい。効果の尺度でスケールした相対
        # 誤差にすることで、小規模な合成テスト（尺度~1）では従来どおり
        # 厳密に、実データ（尺度~10〜30）では現実的な反復回数で収束する。
        scale_ref = max(
            1.0,
            float(np.max(np.abs(new_u))) if n_race else 0.0,
            float(np.max(np.abs(new_v))) if n_horse else 0.0,
        )
        u, v = new_u, new_v
        if delta < tol * scale_ref:
            converged = True
            break

    main_component = _main_component(race_ids_arr, horse_ids_arr)
    main_race_idx = {r_i for r_i, rid in enumerate(uniq_races) if int(rid) in main_component}

    race_effects = pl.DataFrame({"race_id": uniq_races.tolist(), "effect": u.tolist()}).filter(
        pl.col("race_id").is_in(list(main_component))
    ).sort("race_id")

    horse_in_main = np.zeros(n_horse, dtype=bool)
    for r_i, h_i in zip(race_code.tolist(), horse_code.tolist()):
        if r_i in main_race_idx:
            horse_in_main[h_i] = True
    keep_horses = uniq_horses[horse_in_main]
    horse_effects = pl.DataFrame({"horse_id": uniq_horses.tolist(), "effect": v.tolist()}).filter(
        pl.col("horse_id").is_in(keep_horses.tolist())
    ).sort("horse_id")

    return VariantFit(
        as_of=as_of,
        horse_effects=horse_effects,
        race_effects=race_effects,
        n_rows=pop.height,
        n_iter=n_iter,
        converged=converged,
        sigma2_error=sigma2_error,
        sigma2_race=sigma2_race,
        sigma2_horse=sigma2_horse,
        scale=scale,
        main_component=main_component,
    )


def horse_effect_series(fits: list[VariantFit]) -> pl.DataFrame:
    """複数の `VariantFit` を `F-302` の入力形式に束ねる（`D-107`）。

    戻り値の列: `horse_id: Int64`, `as_of: Date`, `effect: Float64`。
    `(horse_id, as_of)` で一意。`as_of` の昇順に整列済み（`attach_f302()`
    の `join_asof` の前提）。
    """
    parts = [
        fit.horse_effects.with_columns(pl.lit(fit.as_of).alias("as_of"))
        for fit in fits
        if not fit.horse_effects.is_empty()
    ]
    if not parts:
        return pl.DataFrame(schema={"horse_id": pl.Int64, "as_of": pl.Date, "effect": pl.Float64})
    return pl.concat(parts).select(["horse_id", "as_of", "effect"]).sort(["horse_id", "as_of"])
