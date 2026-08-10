#!/usr/bin/env python3
"""spec-hearing の実行結果から、機械的に判定できる事実を抽出する。

採点者はこの出力を根拠として使う。目視より速く、イテレーション間で結果がぶれない。

usage: python3 check_run.py <run_outputs_dir>
"""
import json
import re
import sys
from pathlib import Path

# ヒアリング前のリポジトリが持っていたID。これより後の番号が「新設」。
BASE_D = {f"D-{i:03d}" for i in range(1, 10)}
BASE_Q = {f"Q-{i:03d}" for i in range(1, 10)}

# リーク防止・制約が書かれているかを見るための手掛かり
LEAK_HINTS = ["リーク", "leak", "発走時点", "実測ラップ", "目的変数"]
RATIONALE_HINTS = ["根拠", "理由", "なぜ", "検討経緯", "背景"]


def ids_in(text, prefix):
    return set(re.findall(rf"\b{prefix}-\d{{3}}\b", text))


def headings(text, prefix):
    """見出しとして定義されているID（参照ではなく定義）"""
    return re.findall(rf"^##\s+({prefix}-\d{{3}})\b", text, re.M)


def main(root: Path):
    docs = root / "docs"
    spec_dir = docs / "spec"

    spec_files = sorted(p.name for p in spec_dir.glob("*.md") if p.name != "README.md")

    dec = (docs / "decisions.md").read_text(encoding="utf-8") if (docs / "decisions.md").exists() else ""
    oq = (docs / "open-questions.md").read_text(encoding="utf-8") if (docs / "open-questions.md").exists() else ""

    d_defined = headings(dec, "D")
    q_defined = headings(oq, "Q")

    out = {
        "spec_files_created": spec_files,
        "notes_md_exists": (root / "NOTES.md").exists(),
        "decisions": {
            "defined": d_defined,
            "new": sorted(set(d_defined) - BASE_D),
            # 既存IDの使い回しは「同じ見出しが2回定義される」形で現れる
            "duplicated": sorted({i for i in d_defined if d_defined.count(i) > 1}),
        },
        "open_questions": {
            "defined": q_defined,
            "new": sorted(set(q_defined) - BASE_Q),
            "duplicated": sorted({i for i in q_defined if q_defined.count(i) > 1}),
            "q001_marked_resolved": bool(re.search(r"^## Q-001.*\n+\*\*状態\*\*: *Resolved", oq, re.M)),
        },
        "specs": {},
    }

    for name in spec_files:
        text = (spec_dir / name).read_text(encoding="utf-8")
        body = re.split(r"^##\s", text, maxsplit=1)[-1]  # ヘッダ表以降
        out["specs"][name] = {
            "lines": text.count("\n") + 1,
            "refs_D": sorted(ids_in(text, "D")),
            "refs_F": sorted(set(re.findall(r"\bF-\d{3}\b", text))),
            "refs_Q": sorted(ids_in(text, "Q")),
            "has_header_table": all(k in text for k in ("Phase", "関連決定", "状態")),
            "status_draft": bool(re.search(r"状態.*Draft", text)),
            "sections": re.findall(r"^##\s+(.+)$", text, re.M),
            "leak_hint_hits": [h for h in LEAK_HINTS if h in text],
            "rationale_word_hits": [h for h in RATIONALE_HINTS if h in body],
        }

    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
