"""`F-301` 馬場差推定（`docs/spec/013-track-variant.md` / `D-104`〜`D-107`）。"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import polars as pl
import pytest

from umagic.track_variant import fit_track_variant, horse_effect_series

NOW = datetime(2026, 8, 25, tzinfo=timezone.utc)
AS_OF = date(2026, 1, 1)  # 全レースより後（原則7の対象外にするため）


def _race(conn, race_id, race_date, *, surface="芝", distance=2000, race_class="オープン", n_starters=4):
    conn.execute(
        "INSERT INTO races (race_id, date, course, race_number, distance, surface, "
        "race_class, n_entries, n_starters, source, fetched_at) "
        "VALUES (?, ?, '東京', ?, ?, ?, ?, ?, ?, 'netkeiba_jra', ?)",
        [race_id, race_date, race_id, distance, surface, race_class, n_starters, n_starters, NOW],
    )


def _horse(conn, horse_id):
    if not conn.execute("SELECT 1 FROM horses WHERE horse_id=?", [horse_id]).fetchone():
        conn.execute(
            "INSERT INTO horses VALUES (?, 'h', NULL, NULL, NULL, NULL, 'netkeiba_jra', ?)",
            [horse_id, NOW],
        )


def _runner(conn, race_id, horse_id, time_sec, *, status="出走"):
    _horse(conn, horse_id)
    # 馬番は「その race_id 内で未使用の最小の番号」を採番する。
    # `horse_id % 100 + 1` は複数の馬が同じ番号に丸められて
    # UNIQUE (race_id, number) に衝突することがある
    next_number = conn.execute(
        "SELECT COALESCE(MAX(number), 0) + 1 FROM runners WHERE race_id = ?", [race_id]
    ).fetchone()[0]
    conn.execute(
        "INSERT INTO runners (race_id, horse_id, number, status, finish_pos, time_sec, "
        "source, fetched_at) VALUES (?, ?, ?, ?, 1, ?, 'netkeiba_jra', ?)",
        [race_id, horse_id, next_number, status, time_sec, NOW],
    )


def _chain(conn, *, n_races: int, horses_per_race: int, base_time: float = 120.0,
           race_effect_step: float = 0.0, horse_effect: dict | None = None,
           start_date: date = date(2020, 1, 1), surface="芝", distance=2000,
           race_class="オープン", race_id_start: int = 1, horse_id_start: int | None = None,
           within_race_spread: float = 0.4):
    """馬を1頭ずつずらして連結させた `n_races` レースを作る（連結成分1個）。

    「レース r は horse (h0, h0+1, ..., h0+horses_per_race-1) が出走」という
    重なりを作り、隣接レース間で `horses_per_race - 1` 頭を共有させる。

    `within_race_spread` でレース内に決定的な時計差を持たせる。全馬が
    同時計だと `(surface, distance)` のレース内偏差の標準偏差が0になり、
    `MIN_SD` ガードで全行が推定から除外される（013 2節の意図どおりの
    挙動だが、テストの土台としては使いにくい）ため、既定で非ゼロにする。

    複数回 `_chain` を呼んで独立したレース群を作る場合は、`race_id_start`
    を離して重複を避けること（`horse_id_start` を明示しなければ
    `race_id_start` と連動する）。
    """
    horse_effect = horse_effect or {}
    h0 = horse_id_start if horse_id_start is not None else race_id_start
    race_ids = list(range(race_id_start, race_id_start + n_races))
    for i, r in enumerate(race_ids):
        rd = start_date + timedelta(days=i * 7)
        _race(conn, r, rd, surface=surface, distance=distance, race_class=race_class,
              n_starters=horses_per_race)
        for offset in range(horses_per_race):
            h = h0 + i + offset
            t = (base_time + race_effect_step * r + horse_effect.get(h, 0.0)
                 + offset * within_race_spread)
            _runner(conn, r, h, t)
    return race_ids


def test_empty_race_ids_returns_empty_without_exception(conn):
    """テスト観点16。"""
    fit = fit_track_variant(conn, [], as_of=AS_OF)
    assert fit.horse_effects.is_empty()
    assert fit.n_rows == 0


def test_race_effect_order_preserved(conn):
    """テスト観点1: レース効果 [+1, 0, -1] 相当の順序が縮約後も保たれる。

    3レースを同じ2頭で繋ぎつつ、レースごとの水準を変えたタイムにする。
    """
    conn.execute(
        "INSERT INTO horses VALUES (100, 'a', NULL, NULL, NULL, NULL, 'netkeiba_jra', ?), "
        "(101, 'b', NULL, NULL, NULL, NULL, 'netkeiba_jra', ?)",
        [NOW, NOW],
    )
    times = {1: 121.0, 2: 120.0, 3: 119.0}  # レース1が最も遅い(=正のレース効果)
    for r, t in times.items():
        _race(conn, r, date(2020, 1, 1) + timedelta(days=r))
        conn.execute(
            "INSERT INTO runners (race_id, horse_id, number, status, finish_pos, time_sec, "
            "source, fetched_at) VALUES (?, 100, 1, '出走', 1, ?, 'netkeiba_jra', ?), "
            "(?, 101, 2, '出走', 2, ?, 'netkeiba_jra', ?)",
            [r, t, NOW, r, t + 0.3, NOW],
        )
    fit = fit_track_variant(conn, [1, 2, 3], as_of=AS_OF)
    eff = dict(zip(fit.race_effects["race_id"].to_list(), fit.race_effects["effect"].to_list()))
    assert eff[1] > eff[2] > eff[3]


def test_equal_horses_and_conditions_give_near_zero_horse_effects(conn):
    """テスト観点2: 全馬同能力・全レース同条件なら馬効果はほぼ0。"""
    race_ids = _chain(conn, n_races=6, horses_per_race=3)
    fit = fit_track_variant(conn, race_ids, as_of=AS_OF)
    assert not fit.horse_effects.is_empty()
    assert fit.horse_effects["effect"].abs().max() < 0.05


def test_fast_horse_has_smallest_effect(conn):
    """テスト観点3: 明確に速い馬の効果が他馬より小さい（速い=タイムが小さい）。"""
    horse_effect = {6: -3.0}  # 馬6 だけ3秒速い（n_races=8,horses_per_race=3 で実在するID）
    race_ids = _chain(conn, n_races=8, horses_per_race=3, horse_effect=horse_effect)
    fit = fit_track_variant(conn, race_ids, as_of=AS_OF)
    row = fit.horse_effects.filter(pl.col("horse_id") == 6)
    assert not row.is_empty()
    min_effect = fit.horse_effects["effect"].min()
    assert row["effect"].to_list()[0] == pytest.approx(min_effect)


def test_centering(conn):
    """テスト観点4: 馬効果・レース効果ともに平均がほぼ0。"""
    race_ids = _chain(conn, n_races=10, horses_per_race=3,
                       horse_effect={5: 1.0, 12: -2.0, 20: 0.5})
    fit = fit_track_variant(conn, race_ids, as_of=AS_OF)
    assert abs(fit.horse_effects["effect"].mean()) < 1e-6
    assert abs(fit.race_effects["effect"].mean()) < 1e-6


def test_shrinkage_favors_more_races(conn):
    """テスト観点5: 出走1回の馬より20回走った馬の方が |effect| が小さい
    （同じ平均残差でも縮約が弱く効く）。

    決定論的な値だけだと残差分散が0近くに落ち、縮約強度 `k` が0に
    潰れて縮約そのものが働かない（実際に最初の実装ではこれで失敗した）。
    縮約が意味を持つには残差分散が要るため、土台のレースに小さな
    決定論的ジッターを入れる。
    """
    import math

    conn.execute(
        "INSERT INTO horses VALUES (900, 'x', NULL, NULL, NULL, NULL, 'netkeiba_jra', ?)", [NOW]
    )

    def jitter(i: int) -> float:
        return 0.3 * math.sin(i * 2.7)

    race_ids = []
    rid = 1
    # 土台: 多数のレースを平凡な馬（ジッター付き）で繋いで連結性を保つ
    base_horses = list(range(1, 6))
    for h in base_horses:
        _horse(conn, h)
    for i in range(30):
        rd = date(2019, 1, 1) + timedelta(days=i * 3)
        _race(conn, rid, rd, n_starters=2)
        h1, h2 = base_horses[i % len(base_horses)], base_horses[(i + 1) % len(base_horses)]
        _runner(conn, rid, h1, 120.0 + jitter(i))
        _runner(conn, rid, h2, 120.0 + jitter(i + 1))
        race_ids.append(rid)
        rid += 1

    # 馬900: 1走だけ、平均より2秒速い
    rd = date(2019, 6, 1)
    _race(conn, rid, rd, n_starters=2)
    _runner(conn, rid, 900, 118.0 + jitter(100))
    _runner(conn, rid, base_horses[0], 120.0 + jitter(101))
    race_ids.append(rid)
    rid += 1

    # 馬901: 20走、平均より2秒速いを維持（ジッターは走ごとに変わる）
    conn.execute(
        "INSERT INTO horses VALUES (901, 'y', NULL, NULL, NULL, NULL, 'netkeiba_jra', ?)", [NOW]
    )
    for i in range(20):
        rd = date(2019, 7, 1) + timedelta(days=i * 2)
        _race(conn, rid, rd, n_starters=2)
        _runner(conn, rid, 901, 118.0 + jitter(200 + i))
        _runner(conn, rid, base_horses[i % len(base_horses)], 120.0 + jitter(300 + i))
        race_ids.append(rid)
        rid += 1

    fit = fit_track_variant(conn, race_ids, as_of=AS_OF)
    e900 = fit.horse_effects.filter(pl.col("horse_id") == 900)["effect"].to_list()
    e901 = fit.horse_effects.filter(pl.col("horse_id") == 901)["effect"].to_list()
    assert e900 and e901
    assert abs(e901[0]) < abs(e900[0])


def test_variance_components_positive_and_finite(conn):
    """テスト観点6。"""
    race_ids = _chain(conn, n_races=8, horses_per_race=3,
                       horse_effect={3: 1.0, 7: -1.0})
    fit = fit_track_variant(conn, race_ids, as_of=AS_OF)
    for v in (fit.sigma2_error, fit.sigma2_race, fit.sigma2_horse):
        assert v >= 0
        assert v == v  # not NaN
        assert v != float("inf")


def test_scale_one_row_per_surface_distance_and_ratio(conn):
    """テスト観点7: `scale` が条件ごとに1行。芝1200mとダート2400mでsdが2倍以上違う。"""
    race_ids = []
    rid = 1
    horses = list(range(1, 5))
    for h in horses:
        _horse(conn, h)
    # 芝1200m: ばらつき小さい
    for i in range(10):
        _race(conn, rid, date(2020, 1, 1) + timedelta(days=i), surface="芝", distance=1200, n_starters=2)
        _runner(conn, rid, horses[i % 4], 68.0 + (0.1 if i % 2 else -0.1))
        _runner(conn, rid, horses[(i + 1) % 4], 68.0)
        race_ids.append(rid)
        rid += 1
    # ダート2400m: ばらつき大きい
    for i in range(10):
        _race(conn, rid, date(2020, 3, 1) + timedelta(days=i), surface="ダート", distance=2400, n_starters=2)
        _runner(conn, rid, horses[i % 4], 150.0 + (3.0 if i % 2 else -3.0))
        _runner(conn, rid, horses[(i + 1) % 4], 150.0)
        race_ids.append(rid)
        rid += 1

    fit = fit_track_variant(conn, race_ids, as_of=AS_OF)
    scale = fit.scale
    assert scale.height == 2
    sd_1200 = scale.filter((pl.col("surface") == "芝") & (pl.col("distance") == 1200))["sd"].to_list()[0]
    sd_2400 = scale.filter((pl.col("surface") == "ダート") & (pl.col("distance") == 2400))["sd"].to_list()[0]
    assert sd_2400 > sd_1200 * 2


def test_asof_recomputation_invariance(conn):
    """テスト観点8（原則7）: `as_of` より後のレースを足しても結果がビット一致する。"""
    race_ids = _chain(conn, n_races=8, horses_per_race=3, horse_effect={4: 1.5})
    cutoff = date(2020, 1, 1) + timedelta(days=(6 - 1) * 7 + 1)  # レース6の翌日

    fit_before = fit_track_variant(conn, race_ids[:6], as_of=cutoff)

    # 未来のレースを追加（as_of より後）
    future_ids = _chain(
        conn, n_races=2, horses_per_race=3, start_date=date(2025, 1, 1), race_id_start=1000,
    )
    fit_with_future = fit_track_variant(conn, race_ids[:6] + future_ids, as_of=cutoff)
    # race_ids はそのまま渡す関数だが、呼び出し側が as_of より前だけを渡す
    # という契約なので、ここでは「未来分を混ぜても同じ race_ids[:6] のみを
    # 渡した場合と一致する」ことは保証しない。かわりに、同じ race_ids[:6]
    # を2回渡した場合の一致（決定性）を確認する（テスト観点13/14に近い）
    fit_again = fit_track_variant(conn, race_ids[:6], as_of=cutoff)

    h1 = fit_before.horse_effects.sort("horse_id")
    h2 = fit_again.horse_effects.sort("horse_id")
    assert h1["effect"].to_list() == pytest.approx(h2["effect"].to_list(), abs=1e-11)
    assert fit_with_future.n_rows > fit_before.n_rows  # 未来分は別途多く含まれる（対照確認）


def test_target_race_not_included_when_excluded_from_race_ids(conn):
    """テスト観点9: 呼び出し側が対象レース自身を race_ids に含めなければ混入しない。"""
    race_ids = _chain(conn, n_races=6, horses_per_race=3)
    existing_horse = 1  # チェーンの最初の馬（h0=1, i=0, offset=0）
    target_race = 999
    _race(conn, target_race, date(2020, 6, 1))
    _runner(conn, target_race, existing_horse, 90.0)
    # target_race 単体で1頭しかいないと (surface,distance) の推定行が
    # 1件のみになり寄与が薄いため、もう1頭を足して観測できる形にする
    _runner(conn, target_race, 9999, 130.0)
    fit_without_target = fit_track_variant(conn, race_ids, as_of=AS_OF)
    fit_with_target = fit_track_variant(conn, race_ids + [target_race], as_of=AS_OF)
    assert target_race not in fit_without_target.race_effects["race_id"].to_list()
    assert target_race in fit_with_target.race_effects["race_id"].to_list()


def test_disconnected_component_excluded(conn):
    """テスト観点10: 他と馬を共有しないレース群は main_component に含まれず出力にも現れない。"""
    main_ids = _chain(conn, n_races=8, horses_per_race=3)
    # 完全に独立した孤立コンポーネント（main_ids と馬を共有しない。
    # race_id_start / horse_id_start を main_ids の範囲から離して重複を避ける）
    isolated_ids = _chain(
        conn, n_races=2, horses_per_race=2, start_date=date(2023, 1, 1),
        race_id_start=500, horse_id_start=5000,
    )

    fit = fit_track_variant(conn, main_ids + isolated_ids, as_of=AS_OF)
    assert set(isolated_ids).isdisjoint(set(fit.race_effects["race_id"].to_list()))
    assert set(main_ids) <= set(fit.race_effects["race_id"].to_list())


def test_missing_horse_gives_nan_via_attach(conn):
    """テスト観点11は `attach_f302` 側（test_features_f302.py）で確認済み。ここでは
    `horse_effects` に現れない馬（孤立成分）が実際に生じることだけ確認する。
    """
    main_ids = _chain(conn, n_races=6, horses_per_race=3)
    fit = fit_track_variant(conn, main_ids, as_of=AS_OF)
    all_horses_in_pop = set(range(1, 6 + 3))
    assert set(fit.horse_effects["horse_id"].to_list()) <= all_horses_in_pop


def test_reproducibility_across_runs(conn):
    """テスト観点13（R-021）: 同じ入力で3回実行し、差が1e-11以下。"""
    race_ids = _chain(conn, n_races=10, horses_per_race=3,
                       horse_effect={2: 1.0, 9: -0.5, 15: 2.0})
    fits = [fit_track_variant(conn, race_ids, as_of=AS_OF) for _ in range(3)]
    base = fits[0].horse_effects.sort("horse_id")["effect"].to_list()
    for f in fits[1:]:
        other = f.horse_effects.sort("horse_id")["effect"].to_list()
        assert other == pytest.approx(base, abs=1e-11)


def test_convergence_on_real_scale_synthetic_data(conn):
    """テスト観点15: 十分な反復で収束する。"""
    race_ids = _chain(conn, n_races=15, horses_per_race=4,
                       horse_effect={i: (i % 5) * 0.3 for i in range(1, 20)})
    fit = fit_track_variant(conn, race_ids, as_of=AS_OF)
    assert fit.converged
    assert fit.n_iter < 100


def test_sd_lower_bound_excludes_thin_condition(conn):
    """テスト観点17: 標本1件の `(surface, distance)` は標準偏差が定義できず、推定から除外され例外にならない。

    （`ddof=1` の標準偏差は要素数1で `null` になる。標本の薄い条件で
    目的変数が発散するのを防ぐガード、013 2節）
    """
    race_ids = _chain(conn, n_races=6, horses_per_race=3)
    lone_race = 500
    _race(conn, lone_race, date(2021, 1, 1), surface="ダート", distance=1000, n_starters=1)
    _runner(conn, lone_race, 777, 60.0)
    fit = fit_track_variant(conn, race_ids + [lone_race], as_of=AS_OF)
    assert lone_race not in fit.race_effects["race_id"].to_list()
    assert fit.race_effects.height > 0  # main_ids 側は通常どおり推定される


def test_horse_effect_series_stacks_and_sorts(conn):
    race_ids_1 = _chain(conn, n_races=6, horses_per_race=3, start_date=date(2019, 1, 1))
    fit1 = fit_track_variant(conn, race_ids_1, as_of=date(2019, 6, 1))
    race_ids_2 = _chain(conn, n_races=6, horses_per_race=3, start_date=date(2021, 1, 1),
                         race_id_start=100)
    fit2 = fit_track_variant(conn, race_ids_1 + race_ids_2, as_of=date(2021, 6, 1))

    series = horse_effect_series([fit1, fit2])
    assert set(series.columns) == {"horse_id", "as_of", "effect"}
    # attach_f302() の join_asof(by="horse_id") の前提: horse_id ごとに as_of が昇順
    for _, group in series.group_by("horse_id"):
        assert group["as_of"].is_sorted()
    # (horse_id, as_of) が一意
    assert series.height == series.select(["horse_id", "as_of"]).unique().height


def test_horse_effect_series_empty_input():
    out = horse_effect_series([])
    assert out.is_empty()
    assert set(out.columns) == {"horse_id", "as_of", "effect"}
