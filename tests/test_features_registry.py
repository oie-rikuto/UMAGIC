"""`003-features.md` の `FeatureRegistry`（`D-028` / `R-028`）。"""

from __future__ import annotations

import pytest

from umagic.features.registry import FeatureRegistry, FeatureSpec


def test_register_and_all():
    reg = FeatureRegistry()
    reg.register(FeatureSpec("F-101", ("f101",), timing="木曜"))
    reg.register(FeatureSpec("F-501", ("f501",), timing="当日", minutes_before_post=30))
    assert {s.name for s in reg.all()} == {"F-101", "F-501"}


def test_duplicate_registration_rejected():
    reg = FeatureRegistry()
    reg.register(FeatureSpec("F-101", ("f101",), timing="木曜"))
    with pytest.raises(ValueError):
        reg.register(FeatureSpec("F-101", ("f101",), timing="木曜"))


def test_minutes_before_post_only_valid_for_today():
    with pytest.raises(ValueError):
        FeatureSpec("F-101", ("f101",), timing="木曜", minutes_before_post=15)


def test_today_without_minutes_is_allowed():
    """Q-019: 発走何分前かが未確認の当日特徴量が存在する。登録は許容する。"""
    spec = FeatureSpec("F-603", ("f603",), timing="当日", minutes_before_post=None)
    assert spec.minutes_before_post is None


# --- columns_for（D-024 の締切） -------------------------------------------

def _sample_registry() -> FeatureRegistry:
    reg = FeatureRegistry()
    reg.register(FeatureSpec("F-101", ("f101",), timing="木曜"))
    reg.register(FeatureSpec("F-603", ("f603",), timing="当日", minutes_before_post=None))  # Q-019
    reg.register(FeatureSpec("F-501", ("f501",), timing="当日", minutes_before_post=30))
    reg.register(FeatureSpec("F-999", ("f999",), timing="当日", minutes_before_post=10))    # 締切未達
    return reg


def test_provisional_route_excludes_all_today_features():
    """暫定経路は当日情報を一切使わない（D-024）。"""
    cols = _sample_registry().columns_for("暫定")
    assert cols == ["f101"]


def test_main_route_includes_today_features_meeting_deadline():
    cols = _sample_registry().columns_for("本命")
    assert set(cols) == {"f101", "f501"}   # f603(Q-019未確認) と f999(締切未達) は含まない


def test_main_route_excludes_unresolved_deadline_feature():
    """Q-019: 発走何分前か未確認の特徴量は、本命でも締切を満たすと確認できない。"""
    cols = _sample_registry().columns_for("本命")
    assert "f603" not in cols


def test_main_route_excludes_feature_missing_deadline():
    cols = _sample_registry().columns_for("本命")
    assert "f999" not in cols  # T-10 は T-15 の締切に間に合わない


def test_unresolved_deadline_lists_q019_features():
    names = _sample_registry().unresolved_deadline("本命")
    assert names == ["F-603"]
