# UMAGIC ドキュメント

JRA平地G1に限定した競馬予想モデルの設計ドキュメント群。

## ドキュメントの役割分担

このプロジェクトでは **「なぜそうするか」と「何を作るか」を意図的に分離** している。
設計の根拠を仕様書に混ぜると、仕様変更のたびに根拠が失われ、
逆に根拠を残したまま仕様を書くと読み手が実装すべきものを見失うため。

| ドキュメント | 役割 | 誰が読むか |
|---|---|---|
| [`requirements.md`](./requirements.md) | 満たすべき要件（`R-xxx`） | 完了判定をする人 |
| [`decisions.md`](./decisions.md) | 確定した設計判断とその根拠（ADRログ） | 設計変更を検討する人 |
| [`domain-knowledge.md`](./domain-knowledge.md) | プロの予想プロセスの形式化・特徴量カタログ | 特徴量を実装する人 |
| [`architecture.md`](./architecture.md) | システム構成・スキーマ・Phase計画・評価設計 | 全体像を掴む人 |
| [`open-questions.md`](./open-questions.md) | 未決事項と、決めるために必要な情報 | 次に判断する人 |
| [`spec/`](./spec/) | **実装仕様書（今後追加）** | 実装する人 |
| [`tasks.md`](./tasks.md) | 実作業と完了条件。**まだ無い** | 手を動かす人 |

`design.md` は作らない（`D-019`）。設計は `architecture.md`（全体）と `spec/`（モジュール単位）が担う。

## ID体系

ドキュメント間の相互参照のため、以下のIDを使う。
仕様書を書く際は、これらのIDを参照して「どの決定・どの特徴量の実装仕様か」を明示する。

| プレフィックス | 対象 | 定義場所 |
|---|---|---|
| `D-xxx` | 決定事項（Decision） | `decisions.md` |
| `F-xxx` | 特徴量（Feature） | `domain-knowledge.md` |
| `Q-xxx` | 未決事項（Open Question） | `open-questions.md` |
| `P-x` | 開発フェーズ（Phase） | `architecture.md` |
| `R-xxx` | 要件（Requirement） | `requirements.md` |

`tasks.md` のタスクにはIDを振らない（`D-019`）。

IDは**一度振ったら再利用しない**。決定が覆った場合は元のIDを `Superseded by D-xxx` として残し、新しいIDを振る。

## 仕様書を書くときの手順

1. `docs/spec/` に `NNN-<名前>.md` を追加する（[テンプレート](./spec/README.md)を使う）
2. 冒頭で関連する `D-xxx` / `F-xxx` を参照する
3. 仕様の過程で新たな設計判断が発生したら、`decisions.md` に `D-xxx` を追加してから仕様書側で参照する
   （仕様書の中に根拠を書き込まない）
4. 未決事項が出たら `open-questions.md` に `Q-xxx` を追加する

## 現在の状態

**構想フェーズ。実装コードはまだ存在しない。**

- 主要な設計方針は `decisions.md` に27件記録済み（うち1件は Superseded）
- 満たすべき要件は `requirements.md` に27件（`R-001`〜`R-027`）
- `Q-011`（必要なデータ項目が実際に取得できるか）は `D-023` で解決済み。**`P-0` の着手ブロッカーは外れている**
- 残る未決は `open-questions.md` を参照。`P-0` に直接効くのは `Q-008`（実装スタック）、`Q-015`（ページ間の食い違い）、`Q-020`（除外率の許容水準）
