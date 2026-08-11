#!/usr/bin/env python3
"""Q-011 実現可能性チェック — 使い捨てのスパイク。

設計が前提としているデータ項目が実在するかを、実ページを引いて確認する。
**パーサではない。** ページ構造を一度も見ていない段階で書いているため、
「取れた/取れない」を断定せず、**手掛かりの有無を報告して生HTMLを残す**。
最終判定は保存されたHTMLを人が見て行う。

D-014 の遵守条件を実装で担保する:
  - 実行前に robots.txt を確認し、禁止されていれば中断する（条件3）
  - 既定5秒のスリープを挟む（条件2）
  - 生HTMLを保存し、再取得なしで何度でも見直せるようにする（条件2）

使い方:
    python3 check.py --race-id 201405010303 --race-id 202008030304
    python3 check.py --race-id 201405010303 --sleep 8 --out ./raw

race_id は netkeiba を通常閲覧して拾い、引数で渡すこと。
このスクリプトは race_id を推測しない（未確認の構造を前提にしないため）。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
import urllib.robotparser
from pathlib import Path
from urllib.parse import urlsplit

UA = "UMAGIC-feasibility-check/0.1 (personal research; contact: repository owner)"

# 未検証。記事由来の候補であり、実際に叩いて確かめるのがこのスクリプトの目的。
PAGES = {
    "shutuba": "https://race.netkeiba.com/race/shutuba.html?race_id={rid}",
    "result": "https://race.netkeiba.com/race/result.html?race_id={rid}",
}

# 設計が依存している項目。keywords はどれか1つでも出れば「手掛かりあり」。
TARGETS = {
    "lap": {
        "label": "実測ラップ（前半3F/5F・後半3F）",
        "keywords": ["ラップ", "ペース"],
        "regex": r"\d{2}\.\d\s*-\s*\d{2}\.\d",
        "impact": "D-007 の Stage 1 が成立しない。設計の柱が崩れ、F-102 / F-104 / 006-stage1-pace が連鎖する",
    },
    "corners": {
        "label": "コーナー通過順",
        "keywords": ["コーナー通過順位", "通過"],
        "regex": r"\d+\s*-\s*\d+\s*-\s*\d+",
        "impact": "F-101（逃げ意欲）と F-501（当日内外バイアス）が消える。期待値上位2領域",
    },
    "payouts": {
        "label": "払戻（全券種）",
        "keywords": ["払戻", "単勝", "複勝", "馬連", "ワイド", "馬単", "三連複", "三連単"],
        "regex": None,
        "impact": "D-008 の複勝・ワイド主評価が成立しない。バックテスト全体が止まる",
    },
    "race_number": {
        "label": "レース番号・発走時刻",
        "keywords": ["発走", "R"],
        "regex": r"\d{1,2}:\d{2}",
        "impact": "D-010 の同日前後判定ができず、F-501 / F-502 が使えない",
    },
    "head_count": {
        "label": "頭数（出馬表頭数と実出走頭数）",
        "keywords": ["頭"],
        "regex": r"\d{1,2}\s*頭",
        "impact": "D-012 の母数が定義できず、F-801 / F-901 が曖昧になる",
    },
    "status": {
        "label": "出走状態（取消・除外・中止・失格・降着）",
        "keywords": ["取消", "除外", "中止", "失格", "降着"],
        "regex": None,
        "impact": "D-011 が実装できない。返還処理が壊れ、回収率が歪む",
    },
    "affiliation": {
        "label": "所属（外国馬・地方馬の識別）",
        # 表記は未確認。候補を広めに置き、外れたら保存HTMLを見て直す
        "keywords": ["所属", "外国", "地方", "[地]", "[外]", "美浦", "栗東"],
        "regex": None,
        "impact": "018-cold-start の対象を識別できない",
    },
    "odds": {
        "label": "単勝オッズ・人気",
        "keywords": ["オッズ", "人気"],
        "regex": r"\d+\.\d",
        "impact": "市場確率ベースライン normalize(1/単勝オッズ) が作れず、D-002 の目標が定義できない",
    },
}

ENCODINGS = ["euc-jp", "cp932", "utf-8"]


def decode_best(raw: bytes, header_charset: str | None) -> tuple[str, str]:
    """文字コードを判定して復号する。判定に失敗しうるので使った候補も返す。"""
    candidates = []
    if header_charset:
        candidates.append(header_charset)
    m = re.search(rb'charset=["\']?([\w-]+)', raw[:4096], re.I)
    if m:
        candidates.append(m.group(1).decode("ascii", "ignore"))
    candidates += ENCODINGS

    best, best_enc, best_score = "", "?", -1
    for enc in dict.fromkeys(c.lower() for c in candidates if c):
        try:
            text = raw.decode(enc, errors="strict")
        except (UnicodeDecodeError, LookupError):
            continue
        # ひらがな・カタカナ・漢字の出現数を素朴なスコアにする
        score = len(re.findall(r"[ぁ-んァ-ン一-龥]", text))
        if score > best_score:
            best, best_enc, best_score = text, enc, score
    if best_score < 0:
        best, best_enc = raw.decode("utf-8", errors="replace"), "utf-8(replace)"
    return best, best_enc


def check_robots(url: str) -> tuple[bool, str]:
    """D-014 条件3。禁止されていれば False を返して中断させる。"""
    parts = urlsplit(url)
    robots_url = f"{parts.scheme}://{parts.netloc}/robots.txt"
    rp = urllib.robotparser.RobotFileParser()
    try:
        req = urllib.request.Request(robots_url, headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=20) as resp:
            body = resp.read().decode("utf-8", errors="replace")
        rp.parse(body.splitlines())
        allowed = rp.can_fetch(UA, url)
        return allowed, f"robots.txt あり（{len(body)} bytes） / can_fetch={allowed}"
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return True, "robots.txt なし（404）。D-014 の判断時点と同じ状態"
        return True, f"robots.txt 取得できず（HTTP {e.code}）。判断保留のまま続行"
    except Exception as e:  # noqa: BLE001
        return True, f"robots.txt 取得できず（{type(e).__name__}）。判断保留のまま続行"


def fetch(url: str, timeout: int = 30) -> tuple[bytes, str | None, int]:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read(), resp.headers.get_content_charset(), resp.status


def probe(text: str) -> dict[str, dict]:
    out = {}
    for key, t in TARGETS.items():
        hits = [k for k in t["keywords"] if k and k in text]
        rx = bool(re.search(t["regex"], text)) if t["regex"] else None
        out[key] = {
            "label": t["label"],
            "keyword_hits": hits,
            "regex_hit": rx,
            "found": bool(hits) or bool(rx),
            "impact_if_missing": t["impact"],
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Q-011 実現可能性チェック")
    ap.add_argument("--race-id", action="append", required=True,
                    help="netkeiba を閲覧して拾った race_id。複数指定可")
    ap.add_argument("--out", default="raw", help="生HTMLの保存先（既定: raw）")
    ap.add_argument("--sleep", type=float, default=5.0,
                    help="リクエスト間隔（秒）。D-014 条件2。既定5.0を短くしない")
    args = ap.parse_args()

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    sample = PAGES["shutuba"].format(rid=args.race_id[0])
    allowed, robots_msg = check_robots(sample)
    print(f"[robots] {robots_msg}")
    if not allowed:
        print("\n[中断] robots.txt が取得を禁止しています。"
              "D-014 の条件3に該当するため、決定そのものを見直してください。", file=sys.stderr)
        return 2

    summary: dict[str, dict] = {}
    first = True
    for rid in args.race_id:
        summary[rid] = {}
        for kind, tmpl in PAGES.items():
            url = tmpl.format(rid=rid)
            if not first:
                time.sleep(args.sleep)
            first = False

            print(f"\n=== {rid} / {kind} ===\n{url}")
            try:
                raw, charset, status = fetch(url)
            except Exception as e:  # noqa: BLE001
                print(f"  取得失敗: {type(e).__name__}: {e}")
                summary[rid][kind] = {"error": f"{type(e).__name__}: {e}"}
                continue

            path = outdir / f"{rid}_{kind}.html"
            path.write_bytes(raw)
            text, enc = decode_best(raw, charset)
            print(f"  HTTP {status} / {len(raw)} bytes / encoding={enc} / saved={path}")

            res = probe(text)
            summary[rid][kind] = {"http": status, "encoding": enc,
                                  "bytes": len(raw), "saved": str(path), "targets": res}
            for key, r in res.items():
                mark = "OK " if r["found"] else "NG "
                detail = f"keywords={r['keyword_hits']}" if r["keyword_hits"] else ""
                if r["regex_hit"] is not None:
                    detail += f" regex={r['regex_hit']}"
                print(f"  {mark}{r['label']}  {detail}")

    # 見つからなかった項目の影響をまとめる
    missing: dict[str, str] = {}
    for rid, pages in summary.items():
        for kind, page in pages.items():
            for key, r in (page.get("targets") or {}).items():
                if not r["found"]:
                    missing.setdefault(key, r["impact_if_missing"])
    if missing:
        print("\n=== 全ページで手掛かりが見つからなかった項目 ===")
        for key, impact in missing.items():
            print(f"  - {TARGETS[key]['label']}\n      → {impact}")

    sfile = outdir / "summary.json"
    sfile.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[保存] {sfile}")
    print("\n注意: これは手掛かりの有無であって、パースの成否ではない。"
          "\n      保存されたHTMLを実際に開いて、項目が使える形で存在するかを確認すること。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
