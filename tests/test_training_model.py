"""モデルの保存・読み込み（`docs/spec/014-training-pipeline.md` 5節 / `D-082`）。"""

from __future__ import annotations

import re

import numpy as np
import pytest

from umagic.training import (
    git_commit,
    load_model,
    save_model,
    uv_lock_sha256,
    verify_feature_order,
)


def _tiny_booster():
    import lightgbm as lgb

    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0.0, 1.0, 2.0, 3.0])
    ds = lgb.Dataset(X, label=y)
    return lgb.train(
        {"objective": "regression", "verbose": -1, "min_data_in_leaf": 1},
        ds, num_boost_round=3,
    )


def test_save_and_load_round_trip_preserves_feature_order(tmp_path):
    """観点12: meta.json の feature_names が順序込みで一致する。"""
    booster = _tiny_booster()
    meta = {
        "feature_names": ["f101", "f101_unavailable", "f303_rank"],
        "class_weights": {"G1": 5.0, "G2": 3.0},
    }
    out_dir = tmp_path / "fold_0"
    save_model(booster, meta, out_dir)

    assert (out_dir / "model.txt").exists()
    assert (out_dir / "meta.json").exists()

    loaded_booster, loaded_meta = load_model(out_dir)
    assert loaded_meta["feature_names"] == meta["feature_names"]
    assert loaded_meta["class_weights"] == meta["class_weights"]

    X = np.array([[1.5]])
    assert loaded_booster.predict(X)[0] == pytest.approx(booster.predict(X)[0])


def test_load_missing_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_model(tmp_path / "does_not_exist")


def test_verify_feature_order_passes_when_matching():
    meta = {"feature_names": ["a", "b", "c"]}
    verify_feature_order(meta, ["a", "b", "c"])  # 例外を出さない


def test_verify_feature_order_raises_on_mismatch(tmp_path):
    """観点13: 保存時と違う列順で使うと ValueError。"""
    booster = _tiny_booster()
    meta = {"feature_names": ["a", "b", "c"]}
    out_dir = tmp_path / "fold_0"
    save_model(booster, meta, out_dir)

    _, loaded_meta = load_model(out_dir)
    with pytest.raises(ValueError):
        verify_feature_order(loaded_meta, ["b", "a", "c"])


def test_git_commit_returns_full_sha_in_this_repo():
    commit = git_commit()
    assert commit is not None
    assert re.fullmatch(r"[0-9a-f]{40}", commit)


def test_uv_lock_sha256_matches_actual_file():
    import hashlib
    from pathlib import Path

    repo_root = Path(__file__).resolve().parent.parent
    expected = hashlib.sha256((repo_root / "uv.lock").read_bytes()).hexdigest()
    assert uv_lock_sha256() == expected


def test_uv_lock_sha256_returns_none_when_missing(tmp_path):
    assert uv_lock_sha256(repo_root=tmp_path) is None
