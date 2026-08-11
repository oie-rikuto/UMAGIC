# UMAGIC

JRA平地G1に限定した競馬予想モデル。

## 現在の状態

**構想フェーズ。実装コードはまだ存在しない。**

設計方針は [`docs/`](./docs/) に整理してある。

| ドキュメント | 内容 |
|---|---|
| [`docs/decisions.md`](./docs/decisions.md) | 確定した設計判断とその根拠（D-001〜D-014） |
| [`docs/domain-knowledge.md`](./docs/domain-knowledge.md) | プロの予想プロセスの形式化・特徴量カタログ |
| [`docs/architecture.md`](./docs/architecture.md) | システム構成・スキーマ・評価設計・Phase計画 |
| [`docs/open-questions.md`](./docs/open-questions.md) | 未決事項（Q-001〜Q-015） |
| [`docs/spec/`](./docs/spec/) | 実装仕様書（今後追加） |

初めて読む場合は [`docs/README.md`](./docs/README.md) から。

## 設計の要点

- **対象**: JRA平地G1（年24競走前後）
- **目的**: オッズ非依存のファンダメンタルモデルを主軸に、市場確率を超えることを目標とする
- **学習範囲**: JRA全レース（約90万行）で学習し、G1に特化させる。G1のみでは約7,700行しかないため
- **モデル**: レース質予測 → 適性照合 の2段階構成。プロの「レースを先に、馬を後に見る」思考を写したもの
- **成果物**: CLI + バックテストレポート

## 次のアクション

`docs/open-questions.md` の **Q-011（必要なデータ項目が実際に取得できるか）** が着手のブロッカー。実データを引いて確認するまで、設計の前提が検証されていない。
