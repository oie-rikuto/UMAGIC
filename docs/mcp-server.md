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

## エージェントでの予想の進め方

`Q-048`が検討時点で出していた結論はこうだった: **モデルは市場に対して上乗せ情報を持たない（`D-163`）ので、`predict_race`の数値をそのままLLMに渡しても市場と同じ結論以上のものは出ない。LLMが価値を出せるとすれば、UMAGICの数値には無い情報（当日の気配・直前のニュース等をWeb検索で拾う）を掛け合わせたときだけ**。この節はその掛け合わせ方を手順として書く。**手順を書くことと、その手順が市場を上回ることを検証済みなのは別**——後者は`D-119`〜`D-186`の時点で未達のままである。

この手順を実行するには、同じセッションに`umagic`サーバーと**Web検索ができる何か**（Claude Codeの組み込みWeb検索、または別のMCPサーバー）の両方が接続されている必要がある。`umagic`サーバー自身は`src/umagic/`と`docs/`配下のローカルファイルしか読めず（`_READABLE_ROOTS`）、外部情報には一切アクセスしない。

### 手順

1. **対象レースの`race_id`を特定する。** `umagic`サーバーにはレース一覧を引く手段が無いため、Web検索で netkeiba のレースページ（`race.netkeiba.com/race/shutuba.html?race_id=...`）を探す。出馬表は発走の数日前（例年水〜木曜）にしか公開されない
2. **`predict_race(race_id)`でベースラインを取る。** 戻り値の`predictions`（全頭の勝率）と、`training_data_gap_days`（キャッシュの学習時点から対象レース日までの日数——空くほど新しい馬体重・戦績が反映されていない）、および毎回含まれる市場優位に関する留保をそのまま保持する。**この留保を省略して伝えないこと**
3. **市場（単勝オッズ）と突き合わせる。** `predict_race`はオッズを特徴量に使わない設計（`D-002`）だが、出力後に比較材料として見るのは制約に触れない——本プロジェクトのバックテスト自体、常にこの比較で評価してきた（`normalize(1/単勝オッズ)`との LogLoss 比較、`R-023`）。乖離が大きい馬（モデルが市場より強気/弱気）ほど、次のステップで理由を探す価値がある
4. **数値の外側にある情報をWeb検索で補う。** これが`Q-048`のいう「UMAGICの数値には無い情報」であり、この手順で唯一UMAGIC単体を超えうる部分。具体的には: 直前の追い切り評価、乗り替わり・斤量変更、馬体重の増減、パドック・返し馬の気配、当日の馬場傾向（含水率・前残り/差しの決まり方）、天候の変化、有力厩舎のコメント。いずれも`predict_race`の学習データには（`training_data_gap_days`ぶん、あるいは構造的に）反映されていない
5. **モデルの既知の限界を`lookup_doc`/`search_docs`で確認する。** 特に`D-119`（市場を上回れないという核心結論）・`D-186`（皐月賞2026での実地検証、単一レースにつき統計的結論は出せない）・`domain-knowledge.md`3節（G1では斤量・前走着順の情報量が一般的な競馬予想と逆転する）。これらを踏まえずにステップ4の定性情報だけで数値を上書きすると、根拠のない自信になる
6. **統合して提示する。** 「UMAGICの勝率 × 市場オッズとの乖離 × ステップ4の定性情報」をセットで示し、**最後に必ず市場優位が未実証であるという留保を明記する。** 賭けの根拠として断定的に提示しないこと（`predict_race`自体の留保と同じ基準）

### 具体的なプロンプト例

```
umagicのpredict_raceで<race_id>を予測して。市場オッズと比較したうえで、
Web検索で当日の馬場傾向・追い切り評価・乗り替わりを調べて、モデルの
数値と食い違う馬があれば理由を探して。最後に、この予想が市場を上回る
という主張はまだ検証されていないことを明記して。
```

### この手順自体の位置づけ

`Q-048`が挙げていた2用途のうち、この手順は(b)「実際の今後のレースを予想させる運用ツール」にあたる。**手順として実行可能になった（`D-181`〜`D-186`）だけであり、この手順を踏んだ予想が市場を上回るかどうかは未検証。** 検証するには、この手順に従った予想を一定数溜めて回収率をブートストラップ信頼区間つきで見る（`D-008`と同じ基準、単勝なら240レース規模の標本が要る）以外に方法が無く、現時点ではその蓄積が無い。

## 注意点

- `predict_race` はネットワークアクセス（出馬表の取得）を伴う。`D-014`
  のレート制限（既定5秒間隔）がそのままかかる
- `data/umagic.duckdb`（JRA本番DB）は読み取り専用で扱い、書き換えない
  （`D-176`/`D-182` と同じ設計）
- 市場（単勝オッズ）に対する優位はまだ実証されていない（`D-119`〜
  `D-180`）。`predict_race` の出力を賭けの根拠にしないこと
