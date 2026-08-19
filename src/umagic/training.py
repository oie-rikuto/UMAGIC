"""`P-3` 学習パイプライン（`docs/spec/014-training-pipeline.md`）。

walk-forward の fold を生成し、`sample_weight` と乱数シードを与え、
学習済みモデルを保存・読み込む。**モデルそのものは定義しない**
（Stage 1 は `umagic.features` を消費する `006-stage1-pace.md`、
Stage 2 は `007-stage2-ranker.md` の責務）。
"""

from __future__ import annotations

import hashlib
import json
import random
import subprocess
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import duckdb
import polars as pl

from umagic.sealed import is_sealed, sealed_range

DEFAULT_SEALED_YEARS = 3
DEFAULT_MIN_TRAIN_YEARS = 3
DEFAULT_SEED = 20260819
DEFAULT_N_BLOCKS = 4

MODEL_FILENAME = "model.txt"
META_FILENAME = "meta.json"


def _minus_years(d: date, n: int) -> date:
    try:
        return d.replace(year=d.year - n)
    except ValueError:
        # 2/29 起点で n 年後がうるう年でない場合（sealed.sealed_range と同じ扱い）
        return d.replace(year=d.year - n, day=28)


@dataclass(frozen=True)
class Fold:
    """walk-forward の1 fold（`D-080`）。"""

    index: int
    train_start: date
    train_end: date  # 検証開始の前日（含む）
    valid_start: date
    valid_end: date  # 含む
    seed: int  # グローバルシードからの派生（D-085）

    @property
    def inner_valid_start(self) -> date:
        """学習期間末尾1年の開始（`D-084`）。"""
        return _minus_years(self.train_end, 1) + timedelta(days=1)


def make_folds(
    conn: duckdb.DuckDBPyConnection,
    *,
    today: date,
    train_years: int | None = None,
    min_train_years: int = DEFAULT_MIN_TRAIN_YEARS,
    sealed_years: int = DEFAULT_SEALED_YEARS,
    seed: int = DEFAULT_SEED,
) -> list[Fold]:
    """walk-forward の fold を生成する（`D-080`）。

    `min_train_years < 3` は `ValueError`（`D-084`。inner学習が1年以下になり
    ネストしたハイパーパラメータ探索が成立しない）。
    """
    if min_train_years < 3:
        raise ValueError(
            f"min_train_years は3以上でなければならない（D-084）: {min_train_years}"
        )

    all_races = conn.execute("SELECT race_id, date, grade FROM races ORDER BY race_id").pl()
    if all_races.is_empty():
        return []

    # D-079: 封印G1は学習からも除外する。対象期間（1.1節）はこの母集団から決める
    sealed_mask = [
        is_sealed(d, g, today=today, n_years=sealed_years)
        for d, g in zip(all_races["date"].to_list(), all_races["grade"].to_list())
    ]
    kept = all_races.filter(~pl.Series(sealed_mask, dtype=pl.Boolean))
    if kept.is_empty():
        return []

    min_date: date = kept["date"].min()
    max_date: date = kept["date"].max()

    first_valid_year = min_date.year + min_train_years
    last_valid_year = max_date.year
    valid_years = list(range(first_valid_year, last_valid_year + 1))

    rng = random.Random(seed)
    folds: list[Fold] = []
    for i, vy in enumerate(valid_years):
        valid_start = date(vy, 1, 1)
        valid_end = date(vy, 12, 31)
        train_end = valid_start - timedelta(days=1)
        train_start = min_date if train_years is None else _minus_years(valid_start, train_years)
        fold_seed = rng.randrange(2**32)
        folds.append(
            Fold(
                index=i, train_start=train_start, train_end=train_end,
                valid_start=valid_start, valid_end=valid_end, seed=fold_seed,
            )
        )
    return folds


def sample_weights(
    conn: duckdb.DuckDBPyConnection,
    race_ids: list[int],
    *,
    class_weights: dict[str | None, float],
) -> pl.DataFrame:
    """レースのクラス（`races.grade`）に応じた `sample_weight` を返す（`D-081`）。

    `class_weights` に無いクラスは `1.0` にフォールバックする。例外にしない。
    """
    if not race_ids:
        return pl.DataFrame(schema={"race_id": pl.Int64, "sample_weight": pl.Float64})

    rows = conn.execute(
        "SELECT race_id, grade FROM races WHERE race_id = ANY(?)", [race_ids]
    ).pl()
    weights = [class_weights.get(g, 1.0) for g in rows["grade"].to_list()]
    return rows.with_columns(pl.Series("sample_weight", weights, dtype=pl.Float64)).select(
        ["race_id", "sample_weight"]
    )


def cross_fit_blocks(
    fold: Fold, *, n_blocks: int = DEFAULT_N_BLOCKS,
) -> list[tuple[date, date]]:
    """Stage 1 のクロスフィッティング用に学習期間を時系列で分割する（`D-086`）。

    `n_blocks` 個のブロックが時系列順に並び、重複せず学習期間を覆う。
    日数をほぼ均等割りし、余りは前方のブロックに1日ずつ足す。
    """
    total_days = (fold.train_end - fold.train_start).days + 1
    if total_days < n_blocks:
        raise ValueError(
            f"学習期間（{total_days}日）が n_blocks（{n_blocks}）に満たない"
        )

    base, remainder = divmod(total_days, n_blocks)
    blocks: list[tuple[date, date]] = []
    cursor = fold.train_start
    for i in range(n_blocks):
        length = base + (1 if i < remainder else 0)
        block_end = cursor + timedelta(days=length - 1)
        blocks.append((cursor, block_end))
        cursor = block_end + timedelta(days=1)
    return blocks


def save_model(booster, meta: dict, out_dir: Path) -> None:
    """`D-082` の形式（`model.txt` ＋ `meta.json`）でモデルを保存する。

    `meta` は呼び出し側が組み立てて渡す（`fold` / `feature_names` など）。
    本関数はそれをそのまま書き出す純粋なシリアライザで、`git_commit()` /
    `uv_lock_sha256()` の呼び出しは行わない。
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(out_dir / MODEL_FILENAME))
    (out_dir / META_FILENAME).write_text(
        json.dumps(meta, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )


def load_model(model_dir: Path) -> tuple[object, dict]:
    """`save_model()` で保存したモデルを読み込む（`D-082`）。"""
    import lightgbm as lgb

    meta = json.loads((model_dir / META_FILENAME).read_text(encoding="utf-8"))
    booster = lgb.Booster(model_file=str(model_dir / MODEL_FILENAME))
    return booster, meta


def verify_feature_order(meta: dict, columns: list[str]) -> None:
    """`meta["feature_names"]` と実際の入力列順を突き合わせる（`D-082`）。

    一致しなければ `ValueError`。`load_model()` 自体は入力を持たないため、
    予測直前に呼び出し側（`006`/`007`）がこれを呼ぶ。
    """
    expected = meta.get("feature_names")
    if expected != columns:
        raise ValueError(
            f"特徴量の列順が保存時と一致しません。保存時={expected!r} 入力={columns!r}"
        )


def git_commit() -> str | None:
    """`git rev-parse HEAD`。取得できなければ `None`（`D-082` の `meta.json` 用）。"""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
            cwd=Path(__file__).resolve().parent, check=True,
        )
        return out.stdout.strip()
    except Exception:
        return None


def uv_lock_sha256(repo_root: Path | None = None) -> str | None:
    """`uv.lock` の SHA-256。見つからなければ `None`（`D-082` の `meta.json` 用）。"""
    root = repo_root or Path(__file__).resolve().parents[2]
    lock_path = root / "uv.lock"
    if not lock_path.exists():
        return None
    return hashlib.sha256(lock_path.read_bytes()).hexdigest()
