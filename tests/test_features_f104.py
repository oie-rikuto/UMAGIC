"""`F-104` 展開交互作用（`docs/spec/003-features.md` テスト観点7 / `D-021`）。"""

from __future__ import annotations

import polars as pl

from umagic.features.f104 import compute_f104


def test_f102_scale_is_not_collapsed():
    """観点7: F-102 が 2.7 と 40.0 の2レース、F-103 は同一 → F-104 が異なる。"""
    df = pl.DataFrame({
        "race_id": [1, 1, 2, 2],
        "n_starters": [2, 2, 2, 2],
        "f102": [2.7, 2.7, 40.0, 40.0],
        "f103": [1.0, 3.0, 1.0, 3.0],  # レースA・Bで同一の生F-103
    })
    out = compute_f104(df)
    race1 = out.filter(pl.col("race_id") == 1).sort("f103")["f104"].to_list()
    race2 = out.filter(pl.col("race_id") == 2).sort("f103")["f104"].to_list()
    assert race1 != race2
    # 同一のz-scoreにF-102の倍率だけがかかる
    assert abs(race2[0] / race1[0] - 40.0 / 2.7) < 1e-9
    assert abs(race2[1] / race1[1] - 40.0 / 2.7) < 1e-9


def test_relativizes_f103_before_product():
    """積を取る前にF-103だけを相対化する（D-021）。符号が z(F-103) と一致する。"""
    df = pl.DataFrame({
        "race_id": [1, 1, 1],
        "n_starters": [3, 3, 3],
        "f102": [5.0, 5.0, 5.0],
        "f103": [1.0, 2.0, 3.0],
    })
    out = compute_f104(df)
    f104 = out["f104"].to_list()
    assert f104[0] < f104[1] < f104[2]  # F-103の順序を保つ
    assert abs(sum(out["f103_z"].to_list())) < 1e-9  # z-score は平均0


def test_null_f102_propagates_to_null_f104():
    """Stage 1 が無い間（P-1〜P-2）は f102 が全欠損 → f104 も欠損のまま。"""
    df = pl.DataFrame({
        "race_id": [1, 1],
        "n_starters": [2, 2],
        "f102": [None, None],
        "f103": [1.0, 2.0],
    })
    out = compute_f104(df)
    assert out["f104"].is_null().all()
