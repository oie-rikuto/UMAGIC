#!/usr/bin/env python3
"""運用推論キャッシュを作る（`D-183`）。`predict_race`/MCPサーバーが毎回
全履歴を学習し直さずに済むよう、本番DB全体でStage1・Stage2を学習して
`data/prediction_cache/` に保存する。

**本番DB（`data/umagic.duckdb`）を更新したら再実行すること。**

使い方:
    uv run python scripts/build_prediction_cache.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from umagic.production_model import build_production_cache

DB_PATH = "data/umagic.duckdb"
CACHE_DIR = Path("data/prediction_cache")


def main() -> None:
    t0 = time.monotonic()
    meta = build_production_cache(DB_PATH, CACHE_DIR)
    dt = time.monotonic() - t0
    print(f"[完了] {dt/60:.1f}分")
    for k, v in meta.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
