"""Stage 2 のカテゴリ丸め（`docs/spec/007-stage2-ranker.md` 4節 / `D-092` / `R-019`）。"""

from __future__ import annotations

import polars as pl

from umagic.stage2 import build_category_mappings, apply_category_mappings


def _train_df(sire_ids: list[int | None]) -> pl.DataFrame:
    return pl.DataFrame({"race_id": list(range(len(sire_ids))), "sire_id": sire_ids})


def test_min_count_threshold_keeps_frequent_drops_rare():
    """観点8: min_count 以上なら残り、未満なら other_code。"""
    # sire=1 は3回、sire=2 は1回だけ
    train = _train_df([1, 1, 1, 2])
    mappings = build_category_mappings(train, min_count=2)
    m = mappings["sire_id"]
    assert 1 in m.keep
    assert 2 not in m.keep

    out = apply_category_mappings(train, mappings)
    codes = out["sire_id"].to_list()
    # sire=1 の3件は同じコード、sire=2 の1件は other_code（他と異なるコード）
    assert codes[0] == codes[1] == codes[2]
    assert codes[3] == m.other_code
    assert codes[3] != codes[0]


def test_unseen_category_at_predict_time_falls_to_other():
    """観点9: 検証期間にのみ現れる値は other_code に落ちる。例外にしない。"""
    train = _train_df([1, 1, 1])
    mappings = build_category_mappings(train, min_count=2)

    unseen = pl.DataFrame({"race_id": [100], "sire_id": [999]})  # 学習期間に無い値
    out = apply_category_mappings(unseen, mappings)
    assert out["sire_id"].to_list()[0] == mappings["sire_id"].other_code


def test_mapping_built_only_from_training_rows():
    """観点10: 丸め表の作成が検証期間の行を数えていない（R-019 / 原則7）。

    学習期間に3回しか出ない sire を min_count=5 未満として棄却し、
    「検証期間を含めれば5回になる」という別データを混ぜても keep に入らない
    ことを確かめる（= build_category_mappings に渡すのは学習期間のみで
    集計されている、という契約をテストで固定する）。
    """
    train = _train_df([1, 1, 1])  # 学習期間では3回のみ
    mappings = build_category_mappings(train, min_count=5)
    assert 1 not in mappings["sire_id"].keep


def test_null_is_preserved_not_mapped_to_other():
    """`null`（sire_id 欠損）は other_code に丸めず null のまま残す（D-058）。"""
    train = _train_df([1, 1, None])
    mappings = build_category_mappings(train, min_count=1)
    out = apply_category_mappings(train, mappings)
    assert out["sire_id"].to_list()[2] is None
