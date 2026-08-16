#!/usr/bin/env python3
"""Q-011 実現可能性チェック — 使い捨てのスパイク。

設計が前提としているデータ項目が実在するかを、実ページを引いて確認する。
**パーサではない。** ページ構造を一度も見ていない段階で書いているため、
「取れた/取れない」を断定せず、**手掛かりの有無を報告して生HTMLを残す**。
最終判定は保存されたHTMLを人が見て行う。

D-014 の遵守条件を実装で担保する:
  - 実行前に robots.txt を**引くホストごとに**確認し、禁止されていれば中断する（条件3）
  - 5秒未満のスリープを受け付けない（条件2）
  - 生HTMLを raw/ にのみ保存し、再取得なしで何度でも見直せるようにする（条件2 / R-017）

使い方:
    python3 check.py --race-id 201405010303 --race-id 202008030304
    python3 check.py --race-id 201405010303 --sleep 8

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
#
# 1回目の実行で判明: race.netkeiba.com は近年のレースにしか実データを返さない。
# 20年前のレース（例: 200605050810）に対しては、着順表・ラップとも空のテンプレート
# （<title>が空、着順テーブルなし）を返す。過去レースの一次アーカイブは
# db.netkeiba.com/race/{rid}/ 側にあり、そちらにはラップタイム・コーナー通過順位・
# 着順表・払戻表が実在することを目視で確認済み（2005年・2006年ジャパンカップ）。
# よって archive を probe 対象に追加する。
PAGES = {
    "shutuba": "https://race.netkeiba.com/race/shutuba.html?race_id={rid}",
    "result": "https://race.netkeiba.com/race/result.html?race_id={rid}",
    "archive": "https://db.netkeiba.com/race/{rid}/",
}

# 設計が依存している項目。
#
# 判定は「keywords を min_hits 個以上含む」かつ「regex があれば一致する」の AND とする。
# OR にすると、どのページにも出る文字（"R" や小数）で found が常に真になり、
# 空テンプレートまで OK と報告する。**偽 OK がこのスクリプトの最悪の失敗モード**であり、
# D-023 はこの出力を根拠に記録されている。取りこぼしは保存HTMLを見れば直せるが、
# 偽 OK は誰も見直さない。
TARGETS = {
    "lap": {
        "label": "実測ラップ（前半3F/5F・後半3F）",
        "keywords": ["ラップ"],
        "min_hits": 1,
        # ラップは4区間以上並ぶ。2区間だけの一致はタイム表記などの巻き込み
        "regex": r"\d{2}\.\d(?:\s*-\s*\d{2}\.\d){3,}",
        "impact": "D-007 の Stage 1 が成立しない。設計の柱が崩れ、F-102 / F-104 / 006-stage1-pace が連鎖する",
    },
    "corners": {
        "label": "コーナー通過順",
        "keywords": ["コーナー通過順位"],
        "min_hits": 1,
        # 行見出しのコーナー番号。D-043 がこの番号に依存する
        "regex": r"[1-4]コーナー",
        "impact": "F-101（逃げ意欲）と F-501（当日内外バイアス）が消える。期待値上位2領域",
    },
    "payouts": {
        "label": "払戻（全券種）",
        "keywords": ["払戻", "単勝", "複勝", "馬連", "ワイド", "馬単", "三連複", "三連単"],
        # 券種名は出馬表のリンクにも出るため、複数そろって初めて払戻表とみなす
        "min_hits": 5,
        "regex": None,
        "impact": "D-008 の複勝・ワイド主評価が成立しない。バックテスト全体が止まる",
    },
    "race_number": {
        "label": "レース番号・発走時刻",
        "keywords": ["発走"],
        "min_hits": 1,
        # 単独の "R" や裸の時刻はナビ・スクリプトにも出る。発走の近傍に限る
        "regex": r"発走[^0-9]{0,12}\d{1,2}:\d{2}",
        "impact": "D-010 の同日前後判定ができず、F-501 / F-502 が使えない",
    },
    "head_count": {
        "label": "頭数（出馬表頭数と実出走頭数）",
        # 保存済み18ページのいずれでも本文に "N頭" は印字されない（<head> の
        # meta description にある "50万頭" を拾っていただけだった）。
        # 頭数は着順表の行数から導く項目であり、テキスト照合では判定できない。
        "derived": True,
        "keywords": [],
        "min_hits": 0,
        "regex": None,
        "impact": "D-012 の母数が定義できず、F-801 / F-901 が曖昧になる",
        "note": "着順表の行数から導く（D-012）。テキスト照合の対象外",
    },
    "status": {
        "label": "出走状態（取消・除外・中止・失格・降着）",
        # 着順欄は "中止" ではなく "中"、"競走除外" ではなく "除" と1文字で書かれる。
        # 単語で探すと db.netkeiba のアーカイブでは常に空振りする。
        # 異常のないレースにはマーカーが1つも出ないため、NG は
        # 「このレースに該当馬がいない」でもありうる。断定に使わないこと。
        "keywords": [],
        "min_hits": 0,
        "regex": r">\s*(?:中|除|取|失)\s*</td>|\d+\s*\(降\)",
        "impact": "D-011 が実装できない。返還処理が壊れ、回収率が歪む",
        "note": "該当馬のいないレースでは NG になる。マーカーの実例は Q-023",
    },
    "affiliation": {
        "label": "所属（外国馬・地方馬の識別）",
        # 表記は未確認。候補を広めに置き、外れたら保存HTMLを見て直す
        "keywords": ["所属", "外国", "地方", "[地]", "[外]", "美浦", "栗東"],
        "min_hits": 1,
        "regex": None,
        "impact": "018-cold-start の対象を識別できない",
    },
    "odds": {
        "label": "単勝オッズ・人気",
        # 裸の小数（\d+\.\d）はインラインJSにも出るため使わない。
        # db.netkeiba は払戻表の "単勝"、race.netkeiba は odds 系のクラス名で出る。
        "keywords": [],
        "min_hits": 0,
        "regex": r"単勝|オッズ|class=\"[^\"]*odds",
        "impact": "市場確率ベースライン normalize(1/単勝オッズ) が作れず、D-002 の目標が定義できない",
    },
}

# 判定対象から外す領域。ナビ・広告・インラインJSに設計項目の文字が紛れ込む
STRIP_RE = re.compile(
    r"<script\b.*?</script>|<style\b.*?</style>|<!--.*?-->|<head\b.*?</head>",
    re.S | re.I,
)

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


def body_text(text: str) -> str:
    """判定対象を本文に絞る。script / style / head / コメントを落とす。"""
    return STRIP_RE.sub(" ", text)


def probe(text: str) -> dict[str, dict]:
    body = body_text(text)
    out = {}
    for key, t in TARGETS.items():
        hits = [k for k in t["keywords"] if k and k in body]
        min_hits = t.get("min_hits", 1)
        kw_ok = len(hits) >= min_hits
        rx = bool(re.search(t["regex"], body)) if t["regex"] else None
        out[key] = {
            "label": t["label"],
            "keyword_hits": hits,
            "min_hits": min_hits,
            "regex_hit": rx,
            # AND 判定。derived はテキスト照合の対象外なので None を返す
            "found": None if t.get("derived") else (kw_ok and (rx is not False)),
            "derived": bool(t.get("derived")),
            "note": t.get("note"),
            "impact_if_missing": t["impact"],
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Q-011 実現可能性チェック")
    ap.add_argument("--race-id", action="append", required=True,
                    help="netkeiba を閲覧して拾った race_id。複数指定可")
    ap.add_argument("--out", default=str(Path(__file__).parent / "raw"),
                    help="生HTMLの保存先。R-017 により raw/ 以外は指定できない")
    ap.add_argument("--sleep", type=float, default=5.0,
                    help="リクエスト間隔（秒）。D-014 条件2。既定5.0を短くしない")
    args = ap.parse_args()

    # D-014 条件2。docstring が「既定5.0を短くしない」と書いているだけでは担保されない
    if args.sleep < 5.0:
        print(f"[中断] --sleep {args.sleep} は D-014 条件2 の下限5.0秒を下回ります。",
              file=sys.stderr)
        return 2

    outdir = Path(args.out)
    # R-017 / D-014 条件1。生HTMLは再配布しない。raw/ 以外はコミットされうる
    if outdir.resolve() != (Path(__file__).parent / "raw").resolve():
        print(f"[中断] --out は {Path(__file__).parent / 'raw'} 以外を指定できません。"
              "\n        生HTMLがコミット対象の場所に落ちるのを防ぐためです（R-017）。",
              file=sys.stderr)
        return 2
    outdir.mkdir(parents=True, exist_ok=True)

    # D-014 条件3。**引くホストすべて**を確認する。
    # db.netkeiba.com は D-037 が本番の主ソースに指定したホストであり、
    # ここを確認しない実装は条件3を満たさない。
    rid0 = args.race_id[0]
    checked_hosts: set[str] = set()
    for kind, tmpl in PAGES.items():
        sample = tmpl.format(rid=rid0)
        host = urlsplit(sample).netloc
        if host in checked_hosts:
            continue
        checked_hosts.add(host)
        allowed, robots_msg = check_robots(sample)
        print(f"[robots] {host}: {robots_msg}")
        if not allowed:
            print(f"\n[中断] {host} の robots.txt が取得を禁止しています。"
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
                mark = "-- " if r["found"] is None else ("OK " if r["found"] else "NG ")
                detail = f"keywords={r['keyword_hits']}" if r["keyword_hits"] else ""
                if r["regex_hit"] is not None:
                    detail += f" regex={r['regex_hit']}"
                if r["note"]:
                    detail += f"  ※{r['note']}"
                print(f"  {mark}{r['label']}  {detail}")

    # どのページにも手掛かりが無かった項目だけを挙げる。
    # ページ単位の欠落の**和**にすると、archive にしかない項目（ラップ・コーナー・払戻）が
    # shutuba で欠けているだけで「取得不可」と報告され、致命的な影響文が並ぶ。
    # 40行目付近に記録したとおり race.netkeiba.com は古いレースで空テンプレートを返すため、
    # これはまさに起きるケースである。正しくは積集合を取る。
    found_anywhere: set[str] = set()
    seen_keys: set[str] = set()
    for pages in summary.values():
        for page in pages.values():
            for key, r in (page.get("targets") or {}).items():
                if r["found"] is None:   # derived はテキスト照合の対象外
                    continue
                seen_keys.add(key)
                if r["found"]:
                    found_anywhere.add(key)
    missing = {k: TARGETS[k]["impact"] for k in seen_keys - found_anywhere}
    if missing:
        print("\n=== どのページでも手掛かりが見つからなかった項目 ===")
        for key, impact in sorted(missing.items()):
            print(f"  - {TARGETS[key]['label']}\n      → {impact}")

    # ページごとの欠落は、どのページを引くべきかの判断材料になるので別に出す
    print("\n=== ページ種別ごとに欠けていた項目（D-037 のページ選択の材料） ===")
    per_kind: dict[str, set[str]] = {}
    for pages in summary.values():
        for kind, page in pages.items():
            for key, r in (page.get("targets") or {}).items():
                if r["found"] is False:
                    per_kind.setdefault(kind, set()).add(key)
    for kind in PAGES:
        gaps = sorted(per_kind.get(kind, set()))
        labels = "、".join(TARGETS[k]["label"] for k in gaps) if gaps else "なし"
        print(f"  {kind}: {labels}")

    sfile = outdir / "summary.json"
    sfile.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[保存] {sfile}")
    print("\n注意: これは手掛かりの有無であって、パースの成否ではない。"
          "\n      保存されたHTMLを実際に開いて、項目が使える形で存在するかを確認すること。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
