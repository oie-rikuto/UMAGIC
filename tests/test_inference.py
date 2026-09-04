"""`D-201`: `inference.PROD_DB_PATH` はプロセスのカレントディレクトリに
依存してはいけない。

`data/umagic.duckdb` を要さない（実データ不要）。実データを使った
`build_overlay()` 自体のテストは `tests/test_inference_realdata.py`
（`-m realdata`）にある。
"""
from __future__ import annotations

from pathlib import Path

import pytest


def test_prod_db_path_is_absolute():
    """相対パスのままだと、呼び出し元プロセスのcwd次第で解決先が変わる
    （実際にClaude Desktopがサブプロセスをプロジェクトルート以外のcwdで
    起動し、`/data/umagic.duckdb`に解決されて`predict_race`/`explain_race`
    が本番当日に失敗し続けた、`D-200`/`D-201`）。
    """
    from umagic.inference import PROD_DB_PATH

    assert Path(PROD_DB_PATH).is_absolute()


def test_prod_db_path_unaffected_by_cwd(monkeypatch, tmp_path):
    """cwdをプロジェクトルート以外に変えても解決先が変わらないことを
    直接確認する（`D-201`の再発防止）。

    `PROD_DB_PATH` はモジュールロード時に一度だけ計算されるため、cwdを
    変えた**後で再import**したときも壊れないことを確認する——実際の
    事故は「Claude Desktopがプロジェクトルート以外のcwdでサブプロセスを
    起動し、モジュールロード時点でパスが解決される」という同じ形だった。
    """
    import importlib
    import umagic.inference as inference_module

    repo_root = Path(__file__).resolve().parent.parent
    monkeypatch.chdir(tmp_path)
    try:
        importlib.reload(inference_module)
        assert Path(inference_module.PROD_DB_PATH) == repo_root / "data" / "umagic.duckdb"
    finally:
        importlib.reload(inference_module)  # 他テストに影響しないよう元に戻す


def test_prod_db_path_resolves_to_repo_data_dir():
    """プロジェクトの `data/umagic.duckdb` を指していることを確認する
    （壊れていた形の `/data/umagic.duckdb` ではなく）。"""
    from umagic.inference import PROD_DB_PATH

    p = Path(PROD_DB_PATH)
    assert p.parent.name == "data"
    assert p.name == "umagic.duckdb"
    # リポジトリ直下（このテストファイルの2つ上）と一致する
    repo_root = Path(__file__).resolve().parent.parent
    assert p == repo_root / "data" / "umagic.duckdb"
