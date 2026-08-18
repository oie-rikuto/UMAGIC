"""`F-104` 展開交互作用（`docs/spec/003-features.md` / `D-021`）。

`F-102`（Stage 1 の出力、`006-stage1-pace.md`、`P-3`）× `F-103_z`
（`F-103` をレース内相対化した値、`F-901`）。**積を取る前に `F-103`
の側だけを相対化する**（`D-021`）。積を取った後に相対化すると

```
z(F-104) = sign(F-102) · z(F-103)
```

に潰れ、想定ペースの大きさが消える。

`race_level`: `True`

Stage 1 は `P-3` まで存在しないため、`f102` 列が無い（あるいは全欠損の）
入力を渡すと `f104` も欠損のままになる。これは想定内の挙動であり、
フォールバック値で埋めない。
"""

from __future__ import annotations

import polars as pl

from umagic.features.relative import relativize


def compute_f104(
    df: pl.DataFrame,
    *,
    f102_col: str = "f102",
    f103_col: str = "f103",
    race_id_col: str = "race_id",
    n_starters_col: str = "n_starters",
) -> pl.DataFrame:
    """`df` に `f104` 列を追加して返す。

    `df` は `f102_col` `f103_col` `race_id_col` `n_starters_col` を
    あらかじめ持っていること（`build_features` の出力に `F-102` を
    別途 join したもの、を想定）。
    """
    out = relativize(df, f103_col, race_id_col=race_id_col, n_starters_col=n_starters_col)
    out = out.with_columns(
        (pl.col(f102_col) * pl.col(f"{f103_col}_z")).alias("f104")
    )
    return out
