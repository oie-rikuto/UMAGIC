"""`shutuba`（発走前の出馬表）ページの合成 fixture を組み立てるヘルパー。

`race.netkeiba.com` の実ページ（2026年皐月賞ほか、2026-08-30に目視・
regex照合して確認）の構造を再現する。`db.netkeiba.com` の `archive`
ページ（`build_archive.py`）とはテンプレートが別（R-017により実ページの
生HTMLはコミットしない）。
"""

from __future__ import annotations

HEADER_TMPL = """
<html><head><meta charset="utf-8">
<meta name="description" content="{date_y}年{date_m}月{date_d}日 {course}{race_number}R {title}{grade_paren}の出馬表です。">
</head><body>
<div class="RaceList_Item01">
<span class="RaceNum">
<span class="MyRace_Item MyRace-Item-{race_id}"></span>
{race_number}R
</span>
</div>
<div class="RaceList_Item02">
<h1 class="RaceName">{title}
{grade_icon}
</h1>
<div class="RaceData01">
{post_time}発走 /<span> {surface_token}{distance}m</span> ({direction}&nbsp;C)
/ 天候:{weather}<span class="Icon_Weather Weather01"></span>
<span class="Item03">/ 馬場:{track_condition_abbr}</span>
</div>
<div class="RaceData02">
<span>{meeting_no}回</span>
<span>{course}</span>
<span>{meeting_day}日目</span>
<span>{age_cond}</span>
<span>{race_class}</span>
<span>{weight_rule}</span>
<span>{n_entries}頭</span>
</div>
</div>
<div class="RaceList_Item03">
<a href="../top/payback_list.html?kaisai_date={date_y}{date_m:02d}{date_d:02d}&kaisai_id=x" class="LinkMore">{course}払戻一覧</a>
</div>
"""

TABLE_TMPL = """
<table class="Shutuba_Table RaceTable01 ShutubaTable">
<thead><tr class="Header"><th>...</th></tr></thead>
{rows}
</table>
"""

ROW_TMPL = """
<tr class="HorseList" id="tr_{number}">
<td class="Waku{frame} Txt_C"><span>{frame}</span></td>
<td class="Umaban{number} Txt_C">{number}</td>
<td class="CheckMark Horse_Select"></td>
<td class="HorseInfo">
<div><div>
<span class="HorseName"><a href="https://db.netkeiba.com/horse/{horse_key}" target="_blank" title="{horse_name}">{horse_name}</a></span>
</div></div>
</td>
<td class="Barei Txt_C">{sex_age}</td>
<td class="Txt_C">{weight_carried}</td>
<td class="Jockey">
<a href="https://db.netkeiba.com/jockey/result/recent/{jockey_key}/" target="_blank" title="j">{jockey_name}</a>
</td>
<td class="Trainer"><span class="Label{affiliation_label}">{affiliation_name}</span><a href="https://db.netkeiba.com/trainer/result/recent/{trainer_key}/" target="_blank" title="t">{trainer_name}</a></td>
<td class="Weight">{horse_weight}<small>({weight_diff})</small></td>
<td class="Txt_R Popular"><span id="odds-{number}_01" style="font-weight : bold">---.-</span></td>
<td class="Popular Popular_Ninki Txt_C"><span id="ninki-{number}_01">**</span></td>
</tr>
"""

_GRADE_ICON_NUM = {"G1": "1", "G2": "2", "G3": "3"}


def build_shutuba_html(
    *,
    race_id: int, date_y: int, date_m: int, date_d: int,
    course: str = "東京", race_number: int = 11, title: str = "テストレース",
    grade: str | None = None,
    surface: str = "芝", direction: str = "左", distance: int = 2000,
    weather: str = "晴", track_condition: str = "良", post_time: str = "15:40",
    meeting_no: int = 1, meeting_day: int = 1,
    age_cond: str = "3歳以上", race_class: str = "オープン", weight_rule: str = "馬齢",
    n_entries: int | None = None,
    entries: list[dict] | None = None,
) -> bytes:
    entries = entries or []
    n_entries = n_entries if n_entries is not None else len(entries)
    surface_token = {"芝": "芝", "ダート": "ダ"}[surface]
    tc_abbr = {"良": "良", "稍重": "稍", "重": "重", "不良": "不"}[track_condition]
    grade_icon = (
        f'<span class="Icon_GradeType Icon_GradeType{_GRADE_ICON_NUM[grade]}"></span>'
        if grade else ""
    )
    grade_paren = f"({grade})" if grade else ""

    header = HEADER_TMPL.format(
        race_id=race_id, date_y=date_y, date_m=date_m, date_d=date_d,
        course=course, race_number=race_number, title=title, grade_paren=grade_paren,
        grade_icon=grade_icon, post_time=post_time, surface_token=surface_token,
        distance=distance, direction=direction, weather=weather,
        track_condition_abbr=tc_abbr, meeting_no=meeting_no, meeting_day=meeting_day,
        age_cond=age_cond, race_class=race_class, weight_rule=weight_rule,
        n_entries=n_entries,
    )

    rows = []
    for i, e in enumerate(entries, start=1):
        row = dict(
            number=i, frame=1, horse_key=f"{2020000000 + i}", horse_name=f"馬{i}",
            sex_age="牡3", weight_carried="55.0", jockey_key="00000", jockey_name="騎手",
            affiliation_label="1", affiliation_name="美浦", trainer_key="00000",
            trainer_name="調教師", horse_weight="480", weight_diff="0",
        )
        row.update(e)
        rows.append(ROW_TMPL.format(**row))
    table = TABLE_TMPL.format(rows="".join(rows))

    return (header + table + "</body></html>").encode("utf-8")
