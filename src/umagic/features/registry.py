"""特徴量の登録簿（`docs/spec/003-features.md` / `D-028` / `R-028`）。

全 `F-xxx` の確定時刻(`timing`)とレース単位性(`race_level`)を属性として持つ。
どちらも**名指し列挙を避けるための仕組み**である。予測経路が使える列は
`timing` から機械的に決まり、`F-901`（レース内相対化）を適用しない列は
`race_level` から機械的に決まる。特徴量を1つ足すたびに、これらの判定表を
書き換える必要が無い。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Timing = Literal["水曜", "木曜", "当日"]
Route = Literal["暫定", "本命"]

# D-024 の締切。暫定は木曜以前、本命は T-15 以前
_ROUTE_ALLOWED_TIMING: dict[Route, set[Timing]] = {
    "暫定": {"水曜", "木曜"},
    "本命": {"水曜", "木曜", "当日"},
}


@dataclass(frozen=True)
class FeatureSpec:
    name: str                          # 'F-101' など
    columns: tuple[str, ...]           # この特徴量が出す列名（生成前の論理名）
    timing: Timing
    minutes_before_post: int | None = None  # timing='当日' のときのみ。未確認は None（Q-019）
    race_level: bool = False           # True: レース内で全馬共通（F-901 を適用しない。D-021）

    def __post_init__(self) -> None:
        # timing='当日' で minutes_before_post が None なのは許容する
        # （Q-019: 発走何分前かが未確認の当日特徴量が存在する）。
        # 逆方向だけを検査する
        if self.timing != "当日" and self.minutes_before_post is not None:
            raise ValueError(
                f"{self.name}: timing='{self.timing}' なのに minutes_before_post が"
                f"設定されている。当日確定以外では意味を持たない"
            )


class FeatureRegistry:
    """全 `F-xxx` の登録簿。属性を欠いたまま登録できない（`R-028`）。"""

    def __init__(self) -> None:
        self._specs: dict[str, FeatureSpec] = {}

    def register(self, spec: FeatureSpec) -> None:
        if spec.name in self._specs:
            raise ValueError(f"{spec.name} は既に登録されている")
        self._specs[spec.name] = spec

    def all(self) -> list[FeatureSpec]:
        return list(self._specs.values())

    def get(self, name: str) -> FeatureSpec:
        return self._specs[name]

    def columns_for(self, route: Route) -> list[str]:
        """指定した経路（`D-024`）で使ってよい列を、確定時刻から機械的に求める。

        `当日` かつ `minutes_before_post` が締切を満たさない列は除く。
        `minutes_before_post is None`（Q-019 未確認）は「本命でも締切を
        満たすと確認できていない」として除外する。使えるはずのものを
        黙って切り捨てるのではなく、`Q-019` が解決するまでの安全側の
        既定動作として扱う。
        """
        allowed_timing = _ROUTE_ALLOWED_TIMING[route]
        deadline_minutes = 15  # D-024: 本命の締切は発走15分前

        cols: list[str] = []
        for spec in self._specs.values():
            if spec.timing not in allowed_timing:
                continue
            if spec.timing == "当日":
                if route == "暫定":
                    continue  # 暫定は当日情報を一切使わない（D-024）
                if spec.minutes_before_post is None or spec.minutes_before_post < deadline_minutes:
                    continue  # Q-019 未確認、または締切に間に合わない
            cols.extend(spec.columns)
        return cols

    def unresolved_deadline(self, route: Route) -> list[str]:
        """本命経路で締切を満たすか判定できない特徴量名（`Q-019`）。"""
        return [
            spec.name for spec in self._specs.values()
            if spec.timing == "当日" and spec.minutes_before_post is None
        ]
