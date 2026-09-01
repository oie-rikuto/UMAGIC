# UMAGIC

JRA平地G1に限定した競馬予想モデル。出走全頭の勝率を出力する。

## 現在の状態（2026-09-02）

**`P-0`〜`P-3` は完了。`P-4`（期待値・ケリー）は着手できない状態にある**——モデルは市場（単勝オッズ）に対する上乗せ情報を持たないことが実証されているため（`D-119`）。詳しい経緯は [`docs/README.md`](./docs/README.md) の「現在の状態」に。

これとは別に、`P-4`を経ずに**LLMエージェントをMCPサーバー経由でUMAGICに接続する運用track**が並走している（`D-181`〜`D-198`）。本番DB・推論キャッシュ・予想の記録採点まで実装済み。使い方は [`docs/mcp-server.md`](./docs/mcp-server.md) に。**この2つは独立している**——運用trackの実装が進んだことは、`P-4`が着手できないという結論には影響しない。

| ドキュメント | 内容 |
|---|---|
| [`docs/requirements.md`](./docs/requirements.md) | 満たすべき要件（`R-001`〜`R-030`） |
| [`docs/decisions.md`](./docs/decisions.md) | 確定した設計判断とその根拠（`D-001`〜`D-198`、197件） |
| [`docs/domain-knowledge.md`](./docs/domain-knowledge.md) | 競馬ドメインの知識・特徴量カタログ（`F-xxx`） |
| [`docs/architecture.md`](./docs/architecture.md) | システム構成・スキーマ・評価設計・Phase計画 |
| [`docs/open-questions.md`](./docs/open-questions.md) | 未決事項（`Q-001`〜`Q-048`） |
| [`docs/spec/`](./docs/spec/) | 実装仕様書（`001`〜`007`/`012`〜`015`、いずれも Draft） |
| [`docs/tasks.md`](./docs/tasks.md) | 実作業と完了条件 |
| [`docs/mcp-server.md`](./docs/mcp-server.md) | LLMエージェントへの接続手順・使い方 |

初めて読む場合は [`docs/README.md`](./docs/README.md) から。

## 設計の要点

- **対象**: JRA平地G1（年24競走前後）。ただし学習はG1に絞らずJRA全レースで行う（`D-003`）——G1のみでは20年遡っても約7,700出走行しかなく学習データとして成立しないため
- **目的**: オッズ非依存のファンダメンタルモデルで市場確率を超えることを目標とした。**現データ・現手法では超えられないと実証済み**（`D-119`：市場確率を渡してもなお得られる上乗せは上限の0.1%以下）。当面の目的は将来の新情報投入のための土台整備に置いている（`D-124`）
- **学習データ**: 2015年以降の10年超（2026-08-30時点で38,861レース・546,304出走行）
- **モデル**: Stage 1（レース質予測）→ Stage 2（適性照合、LightGBM + Plackett-Luce）の2段階構成。発走後にしか分からない情報がStage 1をブロックしないよう分離している（`D-007`）
- **運用インターフェース**: MCPサーバー（`predict_race`/`explain_race`/`query_history`等10ツール）でLLMエージェントに接続できる。期待値・ケリー計算を伴う自動的な賭け推奨は実装していない——モデル確率をそのまま期待値計算に使うと実測で全馬ベタ買いより悪化することが確認されている（`D-190`）
- **スタック**: Python 3.12+ / Polars / DuckDB / LightGBM / uv（`D-042`）

## よく使うコマンド

```bash
uv sync                                        # 依存の同期
uv run pytest tests/ -q -m "not realdata"      # CIと同じ範囲。実データ不要（462件）
uv run pytest -m realdata                      # 実データ（data/umagic.duckdb）を要する
uv run python scripts/ingest_range.py --help   # 取り込みと品質検査
uv run python scripts/build_prediction_cache.py     # 推論キャッシュの構築（本番DB更新後に再実行）
uv run python scripts/predict_race.py --race-id <id>  # まだ発走していないレースを予測する
uv run python scripts/mcp_server.py            # MCPサーバーの起動（stdio）
```

## 残っている制約

- `D-014`: netkeibaの取得は個人利用・再配布しない・レート制限（既定5秒間隔）とキャッシュの実装を条件にしている
- `D-017`/`R-026`: 直近3年のG1は開発中は参照しない封印セット。開いたら回数を記録する規律（禁止ではない）
- `Q-018`: 複勝・ワイドの過去発走前オッズは取得経路が見つかっていない
- 新データ源（`JRA-VAN`/`JRDB`/netkeiba有料）は3経路とも見送り済み（`D-151`/`D-160`/`D-170`）。`P-4`を開ける既知の経路は現時点で無い

`data/` は `.gitignore` 対象（`R-017`）。生HTML・キャッシュ・DuckDB・実験用スクリプトはここに置き、コミットしない。
