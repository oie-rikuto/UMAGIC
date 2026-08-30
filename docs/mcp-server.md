# UMAGIC MCPサーバーの使い方

`Q-048`（`D-181`/`D-182`）で実装した、UMAGICをLLMエージェント（Claude）に
接続するための手順。**着手条件（中央競馬か地方競馬で市場に対して意味の
ある構成ができてから）はユーザーが明示的に一時解除して実装したもの**で、
市場に対する優位という核心問題（`D-119`〜`D-180`）はこのサーバー自体
では解決していないことを踏まえて使うこと。

## サーバーが提供するもの

`scripts/mcp_server.py`（stdio transport）。

| ツール | できること |
|---|---|
| `predict_race(race_id)` | まだ発走していないJRAレースの各馬の勝率を予測する。**実行に数十分かかる**（全履歴を毎回学習し直すため）。戻り値に「市場への優位は未実証」という留保を毎回含む |
| `lookup_doc(doc_id)` | `D-xxx`/`Q-xxx`/`R-xxx`/`F-xxx` のいずれかを指定して全文を1件取得する |
| `search_docs(query, doc=None)` | 上記4体系（`decisions.md`/`open-questions.md`/`requirements.md`/`domain-knowledge.md`）を横断検索する |
| `list_source_files(subdir)` | `src/umagic/` または `docs/` 配下のファイル一覧を返す |
| `read_source(path)` | 実装本体（`.py`）・仕様書（`docs/spec/*.md`）の中身をそのまま返す |

`predict_race` の `race_id` は netkeiba の12桁レースID（例:
`"202606030811"`）。出馬表は発走の数日前（例年水〜木曜）にしか公開され
ない。

## 接続手順

サーバー本体は素の stdio MCP サーバーなので、**Claude Desktop・Claude
Code CLI のどちらからも同じ実行コマンドで接続できる**。`uv`/`PATH` の
違いに左右されないよう、`.venv` の Python を絶対パスで直接呼ぶのが
確実（`uv run` はPATH次第で見つからないことがある。本セッションでも
`uv` コマンド自体はPATHに無く、絶対パスでの呼び出しが必要だった）。

```
python  = /Users/oierikuto/Desktop/UMAGIC/.venv/bin/python
script  = /Users/oierikuto/Desktop/UMAGIC/scripts/mcp_server.py
```

### Claude Code CLI から接続する

```bash
claude mcp add umagic -- /Users/oierikuto/Desktop/UMAGIC/.venv/bin/python \
  /Users/oierikuto/Desktop/UMAGIC/scripts/mcp_server.py
```

登録後は `claude mcp list` で確認できる。**登録した瞬間には稼働中の
セッションのツール一覧には反映されない**（`D-185`で実測確認）——
**新しいセッションを起動して初めて**`umagic`サーバーのツールが使える
ようになる。外すときは `claude mcp remove umagic`。

### Claude Desktop から接続する

設定ファイル（macOS）: `~/Library/Application Support/Claude/claude_desktop_config.json`

```json
{
  "mcpServers": {
    "umagic": {
      "command": "/Users/oierikuto/Desktop/UMAGIC/.venv/bin/python",
      "args": ["/Users/oierikuto/Desktop/UMAGIC/scripts/mcp_server.py"]
    }
  }
}
```

保存後、Claude Desktop を再起動すると認識される。

**Claude Code CLI からの接続は`D-185`で実地検証済み**（`claude mcp add`
→ `✔ Connected`、および`mcp`パッケージのクライアントSDKで
`initialize`→`tools/list`→`tools/call`の実際のプロトコル往復を確認、
`data/mcp_client_test.py`）。**Claude Desktopからの接続は未検証。**
接続できない場合はまず `python scripts/mcp_server.py` を直接実行して
エラーが出ないか確認すること。

## 依存

`mcp` パッケージ（`uv add mcp` で追加済み、`mcp==2.1.1`）。`uv sync` で
他の依存と一緒に入る。

## 注意点

- `predict_race` はネットワークアクセス（出馬表の取得）を伴う。`D-014`
  のレート制限（既定5秒間隔）がそのままかかる
- `data/umagic.duckdb`（JRA本番DB）は読み取り専用で扱い、書き換えない
  （`D-176`/`D-182` と同じ設計）
- 市場（単勝オッズ）に対する優位はまだ実証されていない（`D-119`〜
  `D-180`）。`predict_race` の出力を賭けの根拠にしないこと
