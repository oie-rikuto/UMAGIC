"""netkeiba ソース（`docs/spec/002-loader.md`）。

`D-037` のページ選択規則、`archive` ページのパース、`day_index` の
JRAレース列挙を実装する。`shutuba` のパースは `P-0` の完了条件に
含まれない（`docs/tasks.md`）ため未実装。

markup の根拠は `tools/q011_feasibility/raw/` に保存された実ページを
2026-08-16 に目視・regex 照合して確認した（日本ダービー2023 / スプリンターズS2023 /
ユニコーンS2019 / ジャパンカップ2005 / 東京3R2014 / day_index 20230528）。
"""

from __future__ import annotations

import html
import re
from datetime import date, datetime, timezone

from umagic.sources.base import Fetcher, PageKind, ParsedRace, ParsedShutuba, RawPage, RejectedRow
from umagic.sources.encoding import decode_best

SOURCE = "netkeiba_jra"

PAGES: dict[PageKind, str] = {
    "day_index": "https://db.netkeiba.com/race/list/{key}/",
    "archive": "https://db.netkeiba.com/race/{key}/",
    "shutuba": "https://race.netkeiba.com/race/shutuba.html?race_id={key}",
    "horse_ped": "https://db.netkeiba.com/horse/ped/{key}/",
}

# JRA競馬場コード（01〜10）。地方競馬場は対象外（D-025 / D-005）
_JRA_VENUE_CODES = {f"{i:02d}" for i in range(1, 11)}

#: `day_index`のキャッシュを常時バイパスする対象期間（`D-204`）。実測で
#: レース当日の夜（発走から数時間後）でもまだ`db.netkeiba.com`側の
#: 日別一覧に反映されていないことを確認した——3日あれば十分な余裕
_DAY_INDEX_RECENT_DAYS = 3

_STRIP_RE = re.compile(
    r"<script\b.*?</script>|<style\b.*?</style>|<!--.*?-->", re.S | re.I,
)


class PostPositionsNotDrawn(RuntimeError):
    """出馬表は公開されているが、枠順抽選がまだ行われていない（`D-196`）。

    「ページが未公開」とは区別する——待てば解消する一時的な状態であり、
    `F-801`（枠順バイアス）が確定するまで予測しても意味が無い。
    """


def url_for(source_key: str, page_kind: PageKind) -> str:
    """`D-037` のページ選択規則。確定済みレースは `archive`、発走前は `shutuba`。"""
    return PAGES[page_kind].format(key=source_key)


# ---------------------------------------------------------------------------
# day_index
# ---------------------------------------------------------------------------

def list_race_keys(page: RawPage) -> list[str]:
    """`<h3>中央</h3>` セクションから12桁の race_id を列挙する（JRA平地のみ）。

    同じページに地方競馬（`<h3>地方</h3>`）が並ぶが、ID体系が異なり
    対象外（`D-025`）。中央セクションの終端は次の `<h3>`（=地方の開始）。

    **障害レースもここで除外する。** 対象は「JRA平地全レースとJpnI」
    （`002-loader.md` 制約 / `D-025`）であり、障害はスコープ外。day_index
    ページは各レースの距離表記に "障" を接頭辞として持つため、archive
    ページを取得する前にここで弾ける（障害コースは芝とダートが混在し、
    `_parse_header` の距離正規表現が一致しないため、弾かずに進めると
    `races.distance` が `NULL` になり `NOT NULL` 制約違反で書き込みが落ちる）。
    """
    text = decode_best(page.body, page.encoding if page.encoding != "unknown" else None)
    text = _STRIP_RE.sub(" ", text)

    m_start = re.search(r"<h3>\s*中央\s*</h3>", text)
    if not m_start:
        return []
    m_next = re.search(r"<h3>", text[m_start.end():])
    end = m_start.end() + (m_next.start() if m_next else len(text) - m_start.end())
    segment = text[m_start.end():end]

    seen: dict[str, None] = {}
    for rid, surface_text in re.findall(
        r'/race/(\d{12})/"[^>]*title="[^"]*">.*?<div>\s*([^<]+?)\s*<', segment, re.S,
    ):
        if rid[4:6] not in _JRA_VENUE_CODES:
            continue
        if surface_text.startswith("障"):
            continue
        seen.setdefault(rid, None)
    return list(seen.keys())


def list_nar_race_keys(page: RawPage, course_filter: str) -> list[str]:
    """`<h3>地方</h3>` セクションから、`course_filter` に一致する会場のレースIDを
    列挙する（`Q-047` 段階②、`D-176`）。

    NAR側のday_indexは中央側（`list_race_keys`）と異なるHTML構造を持ち
    （`<dt class="race_top_hold_data"><p>会場名</p>` で会場ブロックが区切られる、
    `race_top_hold_data` に相当する要素が中央側には無い）、正規表現を共有できない
    ため別関数にした。段階①（`data/nar_stage1_sample.py`）のスタンドアロン実装を
    そのまま踏襲する。
    """
    text = decode_best(page.body, page.encoding if page.encoding != "unknown" else None)
    text = _STRIP_RE.sub(" ", text)

    m_start = re.search(r"<h3>\s*地方\s*</h3>", text)
    if not m_start:
        return []
    m_next = re.search(r"<h3>", text[m_start.end():])
    end = m_start.end() + (m_next.start() if m_next else len(text) - m_start.end())
    segment = text[m_start.end():end]

    ids: list[str] = []
    for course_block in re.split(r'(?=<dt class="race_top_hold_data">)', segment):
        cm = re.search(r'<dt class="race_top_hold_data">\s*<p>\s*([^<]+?)\s*</p>', course_block)
        if not cm or cm.group(1).strip() != course_filter:
            continue
        ids.extend(re.findall(r'/race/(\d{12})/"', course_block))
    return ids


# ---------------------------------------------------------------------------
# archive: レースヘッダ
# ---------------------------------------------------------------------------

_GRADE_MAP = {
    "GI": "G1", "GII": "G2", "GIII": "G3",
    "JpnI": "Jpn1", "JpnII": "Jpn2", "JpnIII": "Jpn3",
    "L": "L",
    # 「重賞」は地方競馬（NAR）の見出しで確認した（`Q-047` 段階②）。全国指定
    # （JpnI〜III）を持たない地方限定の重賞がこの表記になる（例:
    # 「第68回羽田盃競走(重賞)」——羽田盃自体は全国交流のある重賞だが見出しは
    # JpnI表記ではなくこちらだった）。JRAの見出しにこの表記は現れないため
    # JRA側の解析結果に影響しない
    "重賞": "重賞",
}


def _parse_header(text: str, race_id: int) -> dict:
    course_m = re.search(
        rf'<a href="/race/{race_id}/"[^>]*class="active"[^>]*>([^<]+)</a>', text,
    )
    course = course_m.group(1).strip() if course_m else None

    # レース見出し情報は `<dl class="racedata ...">` ブロック内に限定して探す。
    # ページ先頭のロゴ <h1> やヘッダーメニューの <span> を誤って拾わないため。
    block_m = re.search(r'<dl class="racedata[^"]*">(.*?)</dl>', text, re.S)
    block = block_m.group(1) if block_m else ""

    rn_m = re.search(r"<dt>\s*(\d{1,2})\s*R\s*</dt>", block)
    race_number = int(rn_m.group(1)) if rn_m else None

    h1_m = re.search(r"<h1>(.*?)</h1>", block, re.S)
    title = html.unescape(re.sub(r"<.*?>", "", h1_m.group(1))).strip() if h1_m else ""
    grade = None
    if h1_m:
        g_m = re.search(r"\((GI{1,3}|JpnI{1,3}|L|重賞)\)", h1_m.group(1))
        if g_m:
            grade = _GRADE_MAP.get(g_m.group(1), g_m.group(1))

    span_m = re.search(r"<span>\s*(.*?)\s*</span>\s*<br", block, re.S)
    surface = direction = weather = track_condition = post_time = None
    distance = None
    if span_m:
        info = html.unescape(re.sub(r"<.*?>", "", span_m.group(1)))
        # 距離表記の接頭辞は「ダ」の略記（track_condition 側は「ダート」表記）。
        # 001-schema.md の CHECK は 'ダート' を要求するため正規化する。
        #
        # 方向のあとにコース形状が挟まる。保存済みページの実測で確認した表記:
        # 「芝右」「芝左」「ダ右」「ダ左」「芝右 外」「芝右 外-内」
        # 「芝右 内2周」の7通り。
        #   - 「外-内」は阪神・京都の3200m（外回りから内回りへ）
        #   - 「内2周」は中山3600m（1周が短く、2周する）のように
        #     **形状の記述そのものに数字（周回数）が含まれる**
        # 「距離の直前は数字以外」という前提は「内2周」で破れる（2周の "2" が
        # ある）。形状の語を数え上げても新しい表記でまた漏れるため、
        # **「digit+m」が最初に現れる位置**を探す形にする。区切りの `/` を
        # 跨がせないことで、距離が無いときに天候側の数字を拾わないようにする。
        #
        # **`障` を意図的に読まない。** 障害は `D-025` のスコープ外で
        # `day_index` の段階で除外している（`D-047`）が、万一そこをすり抜けても
        # ここで距離が読めず `parse_error` になり、**静かにデータへ混入しない**。
        # 障害の表記は「障芝 ダート2910m」で `芝` も `ダ` も内側に含むため、
        # **先頭トークンに錨を打つ**ことでのみ弾ける（部分一致では拾ってしまう）。
        head = info.split("/")[0].strip()
        sd_m = re.match(r"(芝|ダ)(右|左|直線)?.*?(\d+)m", head)
        if sd_m:
            surface = {"芝": "芝", "ダ": "ダート"}[sd_m.group(1)]
            direction, distance = sd_m.group(2), int(sd_m.group(3))
        w_m = re.search(r"天候\s*:\s*(\S+?)\s*/", info)
        if w_m:
            weather = w_m.group(1)
        tc_m = re.search(r"(?:芝|ダート)\s*:\s*(\S+?)\s*/", info)
        if tc_m:
            track_condition = tc_m.group(1)
        pt_m = re.search(r"発走\s*:\s*(\d{1,2}:\d{2})", info)
        if pt_m:
            post_time = pt_m.group(1)

    date_m = re.search(r'class="smalltxt">\s*(\d{4})年(\d{2})月(\d{2})日', text)
    race_date = date(int(date_m.group(1)), int(date_m.group(2)), int(date_m.group(3))) \
        if date_m else None

    smalltxt_m = re.search(r'<p class="smalltxt">(.*?)</p>', text, re.S)
    smalltxt = (re.sub(r"\s+", " ",
                       html.unescape(re.sub(r"<.*?>", "", smalltxt_m.group(1))).replace("\xa0", " "))
                .strip() if smalltxt_m else "")
    race_class, weight_rule, meeting_no, meeting_day, meeting_course, class_unparsed = \
        _parse_smalltxt(smalltxt)

    # `Q-047` 段階②（`D-176`）: `course` の一次情報源（`class="active"` リンク）は
    # NARページでは会場名ではなくレース番号タブ（"7R" 等）を拾ってしまう
    # （実測、大井のアーカイブページで確認）。妥当な会場名に見えない場合だけ、
    # smalltxt の開催表記から取れた会場名で置き換える。JRAは常に active リンクが
    # 成功するため、この分岐はJRAの既存結果を変えない
    if not course or re.match(r"^\d+R$", course):
        course = meeting_course

    # `Q-047` 段階②（`D-176`）: 大井のクラス表記はJRAの語彙（`_RACE_CLASSES`）に
    # 一つも一致しないため、JRA解析が失敗した場合に限りNAR用の解析を試みる。
    # `course == "大井"` に絞るのは、他場の見出し書式を未確認のまま流用しないため
    # （段階③で他場に広げる際に検証してから対象を増やす）
    if class_unparsed and course == "大井":
        nar_class = _parse_nar_race_class(title)
        if nar_class is not None:
            race_class = nar_class
            class_unparsed = False

    return {
        "course": course, "race_number": race_number, "grade": grade,
        "surface": surface, "direction": direction, "distance": distance,
        "weather": weather, "track_condition": track_condition,
        "post_time": post_time, "date": race_date,
        "race_class": race_class, "weight_rule": weight_rule,
        "meeting_no": meeting_no, "meeting_day": meeting_day,
        "class_unparsed": class_unparsed, "smalltxt": smalltxt,
    }


# D-049: smalltxt の第2トークンが開催、第3トークン以降が条件。
#   例: "2023年05月28日 2回東京12日目 3歳オープン (国際) 牡・牝(指)(定量)"
# 保存済み3,606ページで下記の語彙に閉じることを確認した（2026-08-17）。
_RACE_CLASSES = ("新馬", "未勝利", "1勝クラス", "2勝クラス", "3勝クラス", "オープン")
_WEIGHT_RULES = ("馬齢", "定量", "別定", "ハンデ")
# `Q-047` 段階②: 「n回<会場名>m日目」の会場名を第2群として捕捉する。JRA・NAR
# 共通の書式（`D-176`）。当初は会場名を捕捉していなかったが、`course` の一次情報源
# （`class="active"` リンク）がNARページでは別の値（レース番号タブ）を拾ってしまう
# ため、こちらをフォールバックに使う
_MEETING_RE = re.compile(r"(\d+)回(\D+?)(\d+)日目")

# `Q-047` 段階②: 大井の見出しクラス表記（`D-176`）。JRAの「1勝クラス」等とは
# 全く別の体系——A1/A2/B1/B2/B3/C1/C2/C3のクラスラダーで、隣接クラスの併合が
# ある（例: B1B2, C1C2）。実見出し172件（大井2023年4-6月）で確認した2パターン:
#   括弧内にクラス表記がある: "ヘルメス賞(C1)" "南風賞競走(B1B2)" "さつき賞競走(A2)"
#     （括弧内に組番号が続くこともある: "(C2三)" → 有効なクラスは先頭の "C2"）
#   括弧が無く見出し先頭にクラス表記がある: "C3五　六" "B3四"
# 3歳馬の収得賞金帯レース（例: "3歳　164万"）はクラス文字を持たず、この関数では
# 拾わない——生の賞金帯テキストを race_class に流用すると年ごとに閾値が動き
# 事実上ユニークな値になってしまい（`F-304` の class_key を汚す）、それよりは
# 素直に未解析（`None`）として扱う方が安全と判断した
_NAR_CLASS_RE = re.compile(r"[ABC][1-3](?:[ABC][1-3])?")


def _parse_nar_race_class(title: str) -> str | None:
    """大井の見出し（`title`）からクラス表記を抜く。見つからなければ `None`
    （`D-049` と同じ寛容な失敗——レースは棄てず `class_unparsed` として記録する
    のみで、取り込み自体は止めない）。"""
    paren_m = re.search(r"\(([^)]*)\)\s*$", title)
    if paren_m:
        m = _NAR_CLASS_RE.match(paren_m.group(1))
        if m:
            return m.group(0)
    m = _NAR_CLASS_RE.match(title)
    if m:
        return m.group(0)
    if "新馬" in title:
        return "新馬"
    if "オープン" in title:
        return "オープン"
    return None


def _parse_smalltxt(
    s: str,
) -> tuple[str | None, str | None, int | None, int | None, str | None, bool]:
    """smalltxt から (race_class, weight_rule, meeting_no, meeting_day,
    meeting_course, 解釈失敗) を返す。`meeting_course` は `course` のフォールバック
    に使う会場名（`D-176`）。"""
    if not s:
        return None, None, None, None, None, False

    parts = s.split()
    meeting_no = meeting_day = None
    meeting_course = None
    if len(parts) >= 2:
        m = _MEETING_RE.match(parts[1])
        if m:
            meeting_no = int(m.group(1))
            meeting_course = m.group(2)
            meeting_day = int(m.group(3))

    # 条件部（第3トークン以降）。年齢条件が前置されるので部分一致で探す
    cond = " ".join(parts[2:])
    race_class = next((c for c in _RACE_CLASSES if c in cond), None)
    weight_rule = next((w for w in _WEIGHT_RULES if w in cond), None)

    # 条件部があるのにクラスが取れない場合だけ失敗として扱う
    class_unparsed = bool(cond) and race_class is None
    return race_class, weight_rule, meeting_no, meeting_day, meeting_course, class_unparsed


# ---------------------------------------------------------------------------
# archive: コーナー通過順位テーブル → corner_nos
# ---------------------------------------------------------------------------

def _parse_corner_nos(text: str) -> tuple[list[int] | None, str | None]:
    """行見出しから corner_nos を得る。返り値: (corner_nos, 棄却理由)。

    テーブルが無ければ直線競走の可能性と未取得の可能性を区別できないため
    None（`corner_nos = NULL`）を返す。`D-043`。
    """
    m = re.search(r'コーナー通過順位.*?</table>', text, re.S)
    if not m:
        return None, None
    rows = re.findall(r"<th>([^<]*)</th>\s*<td>", m.group(0))
    if not rows:
        return [], None  # 直線競走: caption はあるが行が無い
    nos: list[int] = []
    for label in rows:
        no_m = re.match(r"([1-4])コーナー", label.strip())
        if not no_m:
            return None, "corner_header_unparsed"
        nos.append(int(no_m.group(1)))
    return nos, None


# ---------------------------------------------------------------------------
# archive: 着順テーブル（race_table_01）
# ---------------------------------------------------------------------------

# D-048: 出走取消の表記「取」は P-0 の3年分取り込み（2022〜2024年、
# 137,575出走行）で220件確認した。失格の表記「失」は10年分拡張取り込み
# （2015〜2021年）で1件確認した（202005050104、Q-023 を解決）
_STATUS_MAP: dict[str, str] = {
    "中": "競走中止", "除": "競走除外", "取": "出走取消", "失": "失格",
}


def _cells(row_html: str) -> list[str]:
    return [
        html.unescape(re.sub(r"\s+", " ", re.sub(r"<.*?>", "", c))).strip()
        for c in re.findall(r"<t[hd][^>]*>(.*?)</t[hd]>", row_html, re.S)
    ]


def _parse_status(raw: str) -> tuple[str | None, int | None]:
    """着順欄 → (status, finish_pos)。表に無い値は (None, None)（棄却対象）。"""
    raw = raw.strip()
    m = re.match(r"^(\d+)\s*\(降\)$", raw)
    if m:
        return "降着", int(m.group(1))
    if raw in _STATUS_MAP:
        return _STATUS_MAP[raw], None
    if re.match(r"^\d+$", raw):
        return "出走", int(raw)
    return None, None  # _STATUS_MAP に無い未確認の表記


def _parse_weight(raw: str) -> tuple[int | None, int | None]:
    m = re.match(r"^(\d+)\(([+-]?\d+)\)$", raw)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None, None


def _parse_time_sec(raw: str) -> float | None:
    m = re.match(r"^(\d+):(\d+\.\d)$", raw)
    if m:
        return round(int(m.group(1)) * 60 + float(m.group(2)), 1)
    return None


# D-049: 調教師欄の先頭マーカー。F-703 の遠征判定に使う
_AFFILIATION_RE = re.compile(r"\[([東西地外])\]")


def _parse_affiliation(cell_html: str) -> str | None:
    m = _AFFILIATION_RE.search(cell_html)
    return m.group(1) if m else None


def _parse_link_name(cell_html: str) -> str | None:
    """リンクの表示名を取る。D-057 の騎手名・調教師名。

    調教師欄は先頭に所属マーカー（`[東]` など）が付くため、タグを外した
    セル全体ではなく **`<a>` のテキスト**を採る。
    """
    m = re.search(r"<a\b[^>]*>(.*?)</a>", cell_html, re.S)
    if not m:
        return None
    name = html.unescape(re.sub(r"<.*?>", "", m.group(1))).strip()
    return name or None


def _parse_id_link(cell_html: str) -> str | None:
    m = re.search(
        r'href="/(?:horse|jockey/result/recent|trainer/result/recent|owner/result/recent)'
        r'/(\w+)/?"',
        cell_html,
    )
    return m.group(1) if m else None


def _parse_finish_table(
    text: str, race_id: int, corner_nos: list[int] | None, fetched_at: datetime,
) -> tuple[list[dict], list[RejectedRow], int | None]:
    """戻り値: (runners, rejected, prize)。`prize` は1着馬の賞金列（円）。"""
    m = re.search(
        r'<table[^>]*class="[^"]*race_table_01[^"]*"[^>]*>(.*?)</table>', text, re.S,
    )
    if not m:
        return [], [], None
    rows_html = re.findall(r"<tr[^>]*>(.*?)</tr>", m.group(1), re.S)
    if not rows_html:
        return [], [], None
    header = _cells(rows_html[0])
    idx = {name: i for i, name in enumerate(header)}

    runners: list[dict] = []
    rejected: list[RejectedRow] = []
    prize: int | None = None
    n = len(corner_nos) if corner_nos else 0

    for i, row_html in enumerate(rows_html[1:], start=1):
        cells = _cells(row_html)
        tds = re.findall(r"<t[hd][^>]*>(.*?)</t[hd]>", row_html, re.S)
        if len(cells) <= idx.get("着順", -1):
            continue

        status, finish_pos = _parse_status(cells[idx["着順"]])
        number = int(cells[idx["馬番"]])
        if status is None:
            rejected.append(RejectedRow(
                source_key=str(race_id), row_ref=str(number),
                reason="unknown_finish_marker", raw=cells[idx["着順"]],
            ))
            continue

        horse_key = _parse_id_link(tds[idx["馬名"]])
        jockey_key = _parse_id_link(tds[idx["騎手"]]) if "騎手" in idx else None
        trainer_key = _parse_id_link(tds[idx["調教師"]]) if "調教師" in idx else None
        owner_key = _parse_id_link(tds[idx["馬主"]]) if "馬主" in idx else None

        sex_age = cells[idx["性齢"]]
        sex = sex_age[0] if sex_age else None
        age = int(sex_age[1:]) if len(sex_age) > 1 and sex_age[1:].isdigit() else None

        hw, wd = _parse_weight(cells[idx["馬体重"]]) if "馬体重" in idx else (None, None)
        odds_raw = cells[idx["単勝"]] if "単勝" in idx else ""
        odds_win = float(odds_raw) if odds_raw not in ("", "---") else None
        popularity = int(cells[idx["人気"]]) if idx.get("人気", -1) >= 0 and cells[idx["人気"]] else None
        time_sec = _parse_time_sec(cells[idx["タイム"]]) if "タイム" in idx else None
        last_3f = float(cells[idx["上り"]]) if idx.get("上り", -1) >= 0 and cells[idx["上り"]] else None

        # corners: D-044 の対応表
        corners: list[int] | None
        rejected_corners = False
        passage_raw = cells[idx["通過"]] if "通過" in idx else ""
        if corner_nos == []:
            corners = []  # 直線競走。通過列の値は採らない
        elif corner_nos is None:
            corners = None
        elif status not in ("出走", "降着"):
            corners = None  # 完走しなかった馬。正常（D-044）
        elif not passage_raw:
            corners = None  # 取得できていない
        else:
            parts = [p for p in passage_raw.split("-")]
            if len(parts) != n or not all(p.isdigit() for p in parts):
                corners = None
                rejected_corners = True
            else:
                corners = [int(p) for p in parts]

        if rejected_corners:
            rejected.append(RejectedRow(
                source_key=str(race_id), row_ref=str(number),
                reason="corners_length_mismatch", raw=passage_raw,
            ))

        runners.append({
            "race_id": race_id, "horse_source_key": horse_key, "number": number,
            "frame": int(cells[idx["枠番"]]) if "枠番" in idx and cells[idx["枠番"]] else None,
            "jockey_source_key": jockey_key, "trainer_source_key": trainer_key,
            "owner_source_key": owner_key,
            "jockey_name": (_parse_link_name(tds[idx["騎手"]]) if "騎手" in idx else None),
            "trainer_name": (_parse_link_name(tds[idx["調教師"]]) if "調教師" in idx else None),
            "owner_name": (_parse_link_name(tds[idx["馬主"]]) if "馬主" in idx else None),
            "horse_name": re.sub(r"<.*?>", "", tds[idx["馬名"]]).strip() if "馬名" in idx else None,
            "weight_carried": float(cells[idx["斤量"]]) if cells[idx.get("斤量", -1)] else None,
            "horse_weight": hw, "weight_diff": wd, "age": age, "sex": sex,
            "odds_win": odds_win, "popularity": popularity,
            "status": status, "finish_pos": finish_pos,
            "margin": cells[idx["着差"]] if idx.get("着差", -1) >= 0 and cells[idx["着差"]] else None,
            "time_sec": time_sec, "last_3f": last_3f, "corners": corners,
            "affiliation": (_parse_affiliation(tds[idx["調教師"]])
                            if "調教師" in idx else None),
            "fetched_at": fetched_at,
        })

        prize_raw = cells[idx["賞金(万円)"]] if "賞金(万円)" in idx else ""
        if finish_pos == 1 and prize_raw:
            # 万円 → 円。小数第1位までを最も近い円に丸める
            prize = round(float(prize_raw.replace(",", "")) * 10_000)

    return runners, rejected, prize


# ---------------------------------------------------------------------------
# archive: 払戻 (D-036 の正規形)
# ---------------------------------------------------------------------------

_ASC_TYPES = {"枠連", "馬連", "ワイド", "三連複"}
# 「枠単」は地方競馬（NAR）にあるがJRAには無い券種（`Q-047` 段階①で発見）。
# 「1着枠→2着枠」で `馬単` と同じ矢印区切りの並び順つき表記
_ORDERED_TYPES = {"馬単", "三連単", "枠単"}
_SPLIT_ARROW = re.compile(r"\s*→\s*")
_SPLIT_DASH = re.compile(r"\s*-\s*")


def _parse_combination(bet_type: str, raw: str) -> list[int]:
    if bet_type in _ORDERED_TYPES:
        return [int(x) for x in _SPLIT_ARROW.split(raw)]
    if bet_type in _ASC_TYPES:
        return sorted(int(x) for x in _SPLIT_DASH.split(raw))
    return [int(raw)]  # 単勝・複勝


def _parse_payouts(text: str, race_id: int, fetched_at: datetime) -> list[dict]:
    payouts: list[dict] = []
    for tbl_html in re.findall(
        r'<table[^>]*class="[^"]*pay_table_01[^"]*"[^>]*>(.*?)</table>', text, re.S,
    ):
        for row_html in re.findall(r"<tr[^>]*>(.*?)</tr>", tbl_html, re.S):
            cells = re.findall(r"<t[hd][^>]*>(.*?)</t[hd]>", row_html, re.S)
            if len(cells) < 4:
                continue
            bet_type = html.unescape(re.sub(r"<.*?>", "", cells[0])).strip()
            combos = [html.unescape(c).strip() for c in re.split(r"<br\s*/?>", cells[1])]
            pays = [html.unescape(re.sub(r"[,\s]", "", c)).strip()
                   for c in re.split(r"<br\s*/?>", cells[2])]
            pops = [html.unescape(c).strip() for c in re.split(r"<br\s*/?>", cells[3])]
            for combo, pay, pop in zip(combos, pays, pops):
                combo = re.sub(r"<.*?>", "", combo).strip()
                if not combo or not pay:
                    continue
                combination = _parse_combination(bet_type, combo)
                comb_key = "-".join(str(x) for x in combination)
                payouts.append({
                    "race_id": race_id, "bet_type": bet_type, "comb_key": comb_key,
                    "combination": combination, "payout": int(pay),
                    "popularity": int(pop) if pop.isdigit() else None,
                    "fetched_at": fetched_at,
                })
    return payouts


# ---------------------------------------------------------------------------
# archive: ラップタイム
# ---------------------------------------------------------------------------

def _parse_laps(text: str, race_id: int, fetched_at: datetime) -> list[dict]:
    m = re.search(r"ラップタイム.*?</table>", text, re.S)
    if not m:
        return []
    row_m = re.search(r"<th>ラップ</th>\s*<td[^>]*>([^<]*)</td>", m.group(0))
    if not row_m:
        return []
    values = [v.strip() for v in row_m.group(1).split("-") if v.strip()]
    return [
        {"race_id": race_id, "furlong_no": i, "lap_sec": float(v), "fetched_at": fetched_at}
        for i, v in enumerate(values, start=1)
    ]


# ---------------------------------------------------------------------------
# エントリポイント
# ---------------------------------------------------------------------------

def parse_archive(page: RawPage) -> ParsedRace:
    """`archive` ページを `ParsedRace` に変換する。"""
    race_id = int(page.source_key)
    text = decode_best(page.body, page.encoding if page.encoding != "unknown" else None)
    text = _STRIP_RE.sub(" ", text)
    fetched_at = page.fetched_at

    header = _parse_header(text, race_id)
    corner_nos, corner_reject_reason = _parse_corner_nos(text)
    runners, rejected, prize = _parse_finish_table(text, race_id, corner_nos, fetched_at)
    if corner_reject_reason:
        rejected.append(RejectedRow(
            source_key=str(race_id), row_ref=None,
            reason=corner_reject_reason, raw="",
        ))

    n_entries = len(runners)
    n_starters = sum(1 for r in runners if r["status"] in ("出走", "降着", "競走中止", "失格"))

    race = {
        "race_id": race_id,
        "date": header["date"],
        "course": header["course"],
        "race_number": header["race_number"],
        "post_time": header["post_time"],
        "distance": header["distance"],
        "surface": header["surface"],
        "direction": header["direction"],
        "grade": header["grade"],
        "track_condition": header["track_condition"],
        "weather": header["weather"],
        "weather_forecast": None,
        "n_entries": n_entries,
        "n_starters": n_starters,
        "prize": prize,
        "corner_nos": corner_nos,
        "race_class": header["race_class"],
        "weight_rule": header["weight_rule"],
        "meeting_no": header["meeting_no"],
        "meeting_day": header["meeting_day"],
        "fetched_at": fetched_at,
    }

    # D-049: クラスが対応表に無い場合はレースを取り込んだうえで記録する。
    # 行は捨てない
    if header["class_unparsed"]:
        rejected.append(RejectedRow(
            source_key=str(race_id), row_ref=None,
            reason="unknown_race_class", raw=header["smalltxt"][:200],
        ))

    payouts = _parse_payouts(text, race_id, fetched_at)
    laps = _parse_laps(text, race_id, fetched_at)

    return ParsedRace(race=race, runners=runners, payouts=payouts,
                      odds=[], laps=laps, rejected=rejected)


class NetkeibaJraSource:
    """`Source` プロトコルの実装（`D-009` の差し替え点）。`shutuba` は未実装。

    `list_race_keys` は内部で `fetcher` を使って `day_index` ページを取得する。
    `Fetcher` を構築時に渡すのは、`Source` プロトコルが `list_race_keys(day)` を
    `date` のみから解決させる契約になっているため（`002-loader.md`）。
    """

    name = SOURCE

    def __init__(self, fetcher: Fetcher) -> None:
        self._fetcher = fetcher

    def list_race_keys(self, day) -> list[str]:
        key = day.strftime("%Y%m%d")
        # `D-204`: 直近の日付は netkeiba 側の反映がレース終了後も遅れる
        # ことがある。早い時点で取得した空の結果が無期限キャッシュに
        # 固定される事故が2度起きたため（`D-188`、9/5当日）、直近
        # `_RECENT_DAYS`日以内は常にライブ取得する。過去の確定済み日は
        # 従来どおりキャッシュする（毎回全履歴を再取得しないため）
        is_recent = (date.today() - day).days <= _DAY_INDEX_RECENT_DAYS
        page = self._fetcher.get(url_for(key, "day_index"), source=self.name,
                                 page_kind="day_index", source_key=key,
                                 bypass_cache=is_recent)
        return list_race_keys(page)

    def url_for(self, source_key: str, page_kind: PageKind) -> str:
        return url_for(source_key, page_kind)

    def parse(self, page: RawPage) -> ParsedRace:
        if page.page_kind != "archive":
            raise NotImplementedError(
                f"page_kind={page.page_kind} のパースは P-0 の対象外（shutuba は未実装）"
            )
        return parse_archive(page)


class NetkeibaNarSource:
    """`Source` プロトコルの地方競馬（NAR）版実装（`Q-047` 段階②、`D-176`）。

    `parse()` は `NetkeibaJraSource` と共通（`parse_archive` をそのまま使う）。
    馬・騎手・調教師・馬主のID空間はJRA・NARで共有されている（同じ
    `db.netkeiba.com` の `/horse/{id}/` 等）ため、`name` も `SOURCE`
    （`"netkeiba_jra"`）を流用する——両者を交錯する馬・騎手のエンティティ解決を
    一致させるため。呼称と実態が食い違うが、`source_ids` テーブルの `source` は
    「データ提供元」を表す列であり「開催団体」ではないため、この流用は設計上
    正しい（`D-176`）。開催団体はレース自身の `course`（会場名）で判別する。

    段階②は大井1場に限定するため、`course_filter` はコンストラクタ引数として
    明示的に渡す（既定値を持たせない——対象場を無言で広げないため）。
    """

    name = SOURCE

    def __init__(self, fetcher: Fetcher, course_filter: str) -> None:
        self._fetcher = fetcher
        self._course_filter = course_filter

    def list_race_keys(self, day) -> list[str]:
        key = day.strftime("%Y%m%d")
        is_recent = (date.today() - day).days <= _DAY_INDEX_RECENT_DAYS  # `D-204`
        page = self._fetcher.get(url_for(key, "day_index"), source=self.name,
                                 page_kind="day_index", source_key=key,
                                 bypass_cache=is_recent)
        return list_nar_race_keys(page, self._course_filter)

    def url_for(self, source_key: str, page_kind: PageKind) -> str:
        return url_for(source_key, page_kind)

    def parse(self, page: RawPage) -> ParsedRace:
        if page.page_kind != "archive":
            raise NotImplementedError(
                f"page_kind={page.page_kind} のパースは段階②の対象外（shutuba は未実装）"
            )
        return parse_archive(page)


# ---------------------------------------------------------------------------
# horse_ped: 血統（D-050）
# ---------------------------------------------------------------------------

# 5代血統表は rowspan で世代を表す。父は 16 行、母も 16 行、母父は 8 行を占める。
# 実物（タスティエーラ 2020103532）で確認済み: 父サトノクラウン / 母パルティトゥーラ
# / 母父マンハッタンカフェ（2026-08-17）
_PED_CELL_RE = re.compile(r'<td[^>]*?rowspan="(\d+)"[^>]*>(.*?)</td>', re.S)
_PED_LINK_RE = re.compile(r'href="[^"]*?/horse/(\w+)/?"')


def parse_pedigree(page: RawPage) -> dict:
    """`horse_ped` ページから父・母・母父の source_key を採る。

    戻り値の3キーはいずれも取れなければ `None`。**推測で補わない**（`D-050`）。
    """
    text = decode_best(page.body, page.encoding if page.encoding != "unknown" else None)
    text = _STRIP_RE.sub(" ", text)

    m = re.search(r'<table[^>]*class="blood_table detail"[^>]*>(.*?)</table>', text, re.S)
    if not m:
        return {"horse_source_key": page.source_key,
                "sire_key": None, "dam_key": None, "damsire_key": None}

    cells: list[tuple[int, str]] = []
    for cm in _PED_CELL_RE.finditer(m.group(1)):
        link = _PED_LINK_RE.search(cm.group(2))
        if link:
            cells.append((int(cm.group(1)), link.group(1)))

    gen1 = [c for c in cells if c[0] == 16]
    sire_key = gen1[0][1] if len(gen1) >= 1 else None
    dam_key = gen1[1][1] if len(gen1) >= 2 else None

    damsire_key = None
    if dam_key is not None:
        i = cells.index(gen1[1])
        damsire_key = next((c[1] for c in cells[i + 1:] if c[0] == 8), None)

    return {"horse_source_key": page.source_key, "sire_key": sire_key,
            "dam_key": dam_key, "damsire_key": damsire_key}


# ---------------------------------------------------------------------------
# shutuba: 発走前の出馬表（推論パイプライン専用、`Q-048` 運用予測パス）
# ---------------------------------------------------------------------------
#
# `race.netkeiba.com` の `shutuba.html` は `db.netkeiba.com` の `archive`
# ページと**別サブドメイン・別テンプレート**（`RaceList_Item0*`/`RaceData0*`/
# `Shutuba_Table` の class 体系）で、archive 用の正規表現を流用できない。
# 実ページ（2026年皐月賞、既に確定済みのレースで検証——発走前のページは
# 発走後もURLがそのまま残り閲覧できることを確認した上で流用した）で
# 目視・regex照合して確認した（2026-08-30）。
#
# **オッズ・人気は静的HTMLに含まれない**（JavaScriptによる非同期更新の
# プレースホルダ "---.-" / "**" のみ）。対象レースのオッズを特徴量にしない
# （`D-002`）本プロジェクトの用途には元々不要なため、取得しない。

_GRADE_ICON_MAP = {"1": "G1", "2": "G2", "3": "G3"}


def _parse_shutuba_header(text: str, race_id: int) -> dict:
    # `_STRIP_RE`（呼び出し元）がHTMLコメントを除去済みのため、閉じ位置は
    # コメント（`<!-- /.RaceList_Item02 -->`）ではなく次の兄弟要素の開始
    # タグで判定する
    desc_m = re.search(r'<meta name="description" content="([^"]+)"', text)
    # 実測（2026年皐月賞）は月日をゼロ埋めしない表記（archiveページの
    # smalltxtとは書式が違う）
    date_m = re.search(r"(\d{4})年(\d{1,2})月(\d{1,2})日", desc_m.group(1)) if desc_m else None
    race_date = date(int(date_m.group(1)), int(date_m.group(2)), int(date_m.group(3))) \
        if date_m else None

    block_m = re.search(
        r'<div class="RaceList_Item02">(.*?)<div class="RaceList_Item03">', text, re.S,
    )
    block = block_m.group(1) if block_m else ""

    rn_m = re.search(r'<span class="RaceNum">.*?(\d{1,2})\s*R\s*</span>', text, re.S)
    race_number = int(rn_m.group(1)) if rn_m else None

    title_m = re.search(r'<h1 class="RaceName">\s*(.*?)\s*<span', block, re.S)
    title = html.unescape(title_m.group(1)).strip() if title_m else ""

    grade = None
    g_m = re.search(r"Icon_GradeType(\d+)", block)
    if g_m:
        grade = _GRADE_ICON_MAP.get(g_m.group(1))

    data01_m = re.search(r'<div class="RaceData01">(.*?)</div>', block, re.S)
    surface = direction = distance = weather = track_condition = post_time = None
    if data01_m:
        info = html.unescape(re.sub(r"<.*?>", " ", data01_m.group(1)))
        info = re.sub(r"\s+", " ", info).strip()
        pt_m = re.search(r"(\d{1,2}:\d{2})\s*発走", info)
        if pt_m:
            post_time = pt_m.group(1)
        sd_m = re.search(r"(芝|ダ)\s*(\d+)m\s*\(\s*(右|左|直線)?", info)
        if sd_m:
            surface = {"芝": "芝", "ダ": "ダート"}[sd_m.group(1)]
            distance = int(sd_m.group(2))
            direction = sd_m.group(3)
        w_m = re.search(r"天候\s*[:：]\s*(\S+)", info)
        if w_m:
            weather = w_m.group(1)
        tc_m = re.search(r"馬場\s*[:：]\s*(\S+)", info)
        if tc_m:
            # shutubaページは馬場状態を1文字に略す（archiveページは「稍重」
            # 「不良」とフル表記、`001-schema.md` の CHECK もフル表記を要求
            # するため展開する）。実測（2024年11月、東京・福島）で確認
            track_condition = {"良": "良", "稍": "稍重", "重": "重", "不": "不良"}.get(tc_m.group(1))

    data02_m = re.search(r'<div class="RaceData02">(.*?)</div>', block, re.S)
    meeting_no = meeting_day = None
    race_class = weight_rule = None
    n_entries = None
    course_from_data02 = None
    if data02_m:
        spans = [html.unescape(re.sub(r"<.*?>", "", s)).strip()
                 for s in re.findall(r"<span>(.*?)</span>", data02_m.group(1), re.S)]
        spans = [s for s in spans if s]
        mo_m = re.search(r"(\d+)回", spans[0]) if len(spans) > 0 else None
        meeting_no = int(mo_m.group(1)) if mo_m else None
        md_m = re.search(r"(\d+)日目", spans[2]) if len(spans) > 2 else None
        meeting_day = int(md_m.group(1)) if md_m else None
        race_class = next((c for c in _RACE_CLASSES if any(c in s for s in spans)), None)
        weight_rule = next((w for w in _WEIGHT_RULES if any(w in s for s in spans)), None)
        ne_m = re.search(r"(\d+)頭", " ".join(spans))
        n_entries = int(ne_m.group(1)) if ne_m else None
        # 「4回 中山 1日目」の会場名。`払戻一覧` リンクは結果確定後にしか
        # 存在しないため、発走前のページではこちらが唯一の取得元（`D-196`）
        flat = re.sub(r"\s+", " ", " ".join(spans))
        kaisai_m = re.search(r"\d+回\s*(\S+?)\s*\d+日目", flat)
        if kaisai_m:
            course_from_data02 = kaisai_m.group(1)

    # 結果確定後のページにだけある払戻リンク。発走前は `RaceData02` に頼る
    course_m = re.search(r">(\S+?)\s*払戻一覧</a>", text)
    course = course_m.group(1) if course_m else course_from_data02

    return {
        "date": race_date, "course": course, "race_number": race_number, "title": title,
        "grade": grade, "surface": surface, "direction": direction, "distance": distance,
        "weather": weather, "track_condition": track_condition, "post_time": post_time,
        "race_class": race_class, "weight_rule": weight_rule,
        "meeting_no": meeting_no, "meeting_day": meeting_day, "n_entries": n_entries,
    }


def _parse_shutuba_entries(text: str) -> list[dict]:
    tbl_m = re.search(r'<table class="Shutuba_Table[^"]*"', text)
    if not tbl_m:
        return []
    close_m = text.index("</table>", tbl_m.start())
    segment = text[tbl_m.start():close_m]

    rows = re.findall(r'<tr class="HorseList"[^>]*>(.*?)</tr>', segment, re.S)

    # 枠順抽選の前は馬名・騎手・斤量まで載るが、枠番・馬番のセルが空になる
    # （`<td class="Umaban Txt_C"></td>`）。`F-801`（枠順バイアス）が実際に
    # 効いている以上、この状態で予測しても意味が無いため、行が存在するのに
    # 馬番が1つも取れない場合は「未公開」ではなく専用の例外で区別する（`D-196`）
    if rows and not any(re.search(r'Umaban\d+[^>]*>\s*\d+\s*<', r) for r in rows):
        raise PostPositionsNotDrawn(
            f"出馬表は公開されていますが（{len(rows)}頭）、枠順抽選がまだ行われて"
            "いません。枠番・馬番が確定してから（通常は発走の1〜2日前）"
            "再取得してください。"
        )

    entries: list[dict] = []
    for row in rows:
        num_m = re.search(r'Umaban\d+[^>]*>\s*(\d+)\s*<', row)
        if not num_m:
            continue  # Shutuba_Table 内の他表記（レイアウト補助行）はここで弾く
        number = int(num_m.group(1))
        waku_m = re.search(r'"Waku\d+[^>]*>\s*<span>\s*(\d+)\s*</span>', row)
        frame = int(waku_m.group(1)) if waku_m else None

        # `(\w*)` — 海外調教馬など netkeiba に馬IDを持たない出走馬は
        # `href="…/horse/"`（ID部分が空）になる。`\w+` のままだと正規表現
        # 全体が不一致になり、**存在する馬名まで丸ごと拾えなくなる**
        # （`D-199`で実データ・セントウルS2026で発見: 外国馬「ファストネット
        # ワーク」の行がこれで名前ごと欠落した）
        horse_m = re.search(r'<span class="HorseName"><a href="[^"]*?/horse/(\w*)"[^>]*>([^<]+)', row)
        horse_key = horse_m.group(1) or None if horse_m else None
        horse_name = html.unescape(horse_m.group(2)).strip() if horse_m else None

        sa_m = re.search(r'"Barei Txt_C">\s*(\S+?)\s*</td>', row)
        sex_age = sa_m.group(1) if sa_m else ""
        sex = sex_age[0] if sex_age else None
        age = int(sex_age[1:]) if len(sex_age) > 1 and sex_age[1:].isdigit() else None

        wc_m = re.search(r'<td class="Txt_C">\s*(\d+\.\d)\s*</td>', row)
        weight_carried = float(wc_m.group(1)) if wc_m else None

        # ID部分を `(\w*)` にする理由は horse_m と同じ（`D-199`）。同じ
        # セントウルS2026の行で、この外国馬の調教師（表記「地方」）も
        # `href="…/trainer/result/recent//"`（ID空）で名前が欠落していた
        jockey_m = re.search(r'/jockey/result/recent/(\w*)/"[^>]*>\s*([^<]*?)\s*</a>', row)
        jockey_key = jockey_m.group(1) or None if jockey_m else None
        jockey_name = html.unescape(jockey_m.group(2)).strip() if jockey_m else None

        trainer_m = re.search(
            r'<td class="Trainer"><span class="Label\d">(\S+?)</span>'
            r'<a href="[^"]*?/trainer/result/recent/(\w*)/"[^>]*>\s*([^<]*?)\s*</a>', row,
        )
        affiliation = trainer_key = trainer_name = None
        if trainer_m:
            # 「地方」（地方競馬・海外所属）は東西どちらにも属さないため
            # `.get()` が意図どおり `None` を返す。新しいラベルを足す必要はない
            affiliation = {"美浦": "東", "栗東": "西"}.get(trainer_m.group(1))
            trainer_key = trainer_m.group(2) or None
            trainer_name = html.unescape(trainer_m.group(3)).strip()

        hw_m = re.search(r'<td class="Weight">\s*(\d+)<small>\(([+-]?\d+)\)</small>', row)
        horse_weight = int(hw_m.group(1)) if hw_m else None
        weight_diff = int(hw_m.group(2)) if hw_m else None

        entries.append({
            "number": number, "frame": frame,
            "horse_source_key": horse_key, "horse_name": horse_name,
            "sex": sex, "age": age, "weight_carried": weight_carried,
            "jockey_source_key": jockey_key, "jockey_name": jockey_name,
            "trainer_source_key": trainer_key, "trainer_name": trainer_name,
            "affiliation": affiliation,
            "horse_weight": horse_weight, "weight_diff": weight_diff,
        })
    return entries


def parse_shutuba(page: RawPage) -> ParsedShutuba:
    """`shutuba` ページ（発走前の出馬表）を `ParsedShutuba` に変換する。

    `parse_archive()` とはページ構造が全く別（`race.netkeiba.com` の別
    テンプレート）のため独立して実装する。結果・払戻・オッズは含まれない
    （発走前のため存在しない）。推論パイプライン専用で、本番の
    `ingest_race()`（`ParsedRace` 前提）には接続しない。
    """
    race_id = int(page.source_key)
    text = decode_best(page.body, page.encoding if page.encoding != "unknown" else None)
    text = _STRIP_RE.sub(" ", text)

    header = _parse_shutuba_header(text, race_id)
    entries = _parse_shutuba_entries(text)

    race = {"race_id": race_id, **header}
    return ParsedShutuba(race=race, entries=entries)
