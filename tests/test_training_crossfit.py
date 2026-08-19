"""クロスフィッティングの分割（`docs/spec/014-training-pipeline.md` 4節 / `D-086`）。"""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from umagic.training import Fold, cross_fit_blocks


def _fold(train_start, train_end):
    return Fold(
        index=0, train_start=train_start, train_end=train_end,
        valid_start=train_end + timedelta(days=1),
        valid_end=train_end + timedelta(days=366), seed=1,
    )


def test_evenly_divisible_blocks_are_contiguous_and_cover_period():
    """観点11: 4ブロックが時系列順に並び、重複せず学習期間を覆う。"""
    fold = _fold(date(2020, 1, 1), date(2020, 1, 1) + timedelta(days=399))  # 400日
    blocks = cross_fit_blocks(fold, n_blocks=4)
    assert len(blocks) == 4
    assert all((e - s).days + 1 == 100 for s, e in blocks)
    # 連続していて重複しない
    for (s0, e0), (s1, e1) in zip(blocks, blocks[1:]):
        assert e0 + timedelta(days=1) == s1
    assert blocks[0][0] == fold.train_start
    assert blocks[-1][1] == fold.train_end


def test_remainder_days_distributed_without_gaps():
    """割り切れない日数でも欠落・重複なく学習期間を覆う。"""
    fold = _fold(date(2020, 1, 1), date(2020, 1, 1) + timedelta(days=400))  # 401日
    blocks = cross_fit_blocks(fold, n_blocks=4)
    total_covered = sum((e - s).days + 1 for s, e in blocks)
    assert total_covered == 401
    assert blocks[0][0] == fold.train_start
    assert blocks[-1][1] == fold.train_end
    for (s0, e0), (s1, e1) in zip(blocks, blocks[1:]):
        assert e0 + timedelta(days=1) == s1


def test_too_few_days_raises():
    fold = _fold(date(2020, 1, 1), date(2020, 1, 2))  # 2日しかない
    with pytest.raises(ValueError):
        cross_fit_blocks(fold, n_blocks=4)
