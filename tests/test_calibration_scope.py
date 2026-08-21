"""確率校正の適用範囲（`docs/spec/015-calibration.md` 3節 / `D-099`）。"""

from __future__ import annotations

import inspect

import umagic.calibration as calibration_module


def test_calibration_module_has_no_population_awareness():
    """観点13: `all`/`g1`（`D-071`）のどちらに適用するかは呼び出し側の責務。

    `Calibrator`/`softmax_by_race`/`fit_calibrator` はいずれも母集団の
    区別（`population` や `grade`）を一切知らない。「`all` 母集団には
    適用しない」（`D-099`）は、`all` 母集団の評価経路で `Calibrator.apply()`
    を**呼ばない**という orchestration 層の判断であり、本モジュール自身は
    それを強制する手段を持たない。ここでは本モジュールが population を
    引き回していないことだけを確認する。
    """
    source = inspect.getsource(calibration_module)
    assert "population" not in source
    assert "'all'" not in source and '"all"' not in source
