"""`archive` ページの合成 fixture を組み立てるヘルパー。

`tools/q011_feasibility/raw/` に保存された実ページを 2026-08-16 に
目視・regex 照合して確認した markup 構造に基づく（R-017 により実ページの
生HTMLはコミットしないため、構造だけを再現した合成データを使う）。
"""

from __future__ import annotations

RACE_HEADER_TMPL = """
<div class="race_head"><div class="race_head_inner">
<ul class="race_place fc">
<li><a href="/race/{race_id}/" class="active">{active_link_text}</a></li>
</ul>
</div></div>
<div class="mainrace_data fc"><div class="data_intro">
<dl class="racedata fc">
<dt> {race_number} R </dt>
<dd>
<h1>{title}</h1>
<p><span>
{dist_token}&nbsp;/&nbsp;
天候 : {weather}&nbsp;/&nbsp;
{surface_label} : {track_condition}&nbsp;&nbsp;/&nbsp;
発走 : {post_time}
</span><br />
</p>
</dd>
</dl>
<p class="smalltxt">{date_y}年{date_m:02d}月{date_d:02d}日 {meeting_no}回{course}{meeting_day}日目 {smalltxt_cond}</p>
</div></div>
"""

RUNNER_ROW_TMPL = """
<tr>
<td class="txt_r">{finish}</td>
<td class="w6ml"><span>{frame}</span></td>
<td class="txt_r">{number}</td>
<td class="txt_l"><a href="/horse/{horse_key}/" title="{name}">{name}</a></td>
<td class="txt_c">{sex_age}</td>
<td class="txt_c">{weight_carried}</td>
<td class="txt_l"><a href="/jockey/result/recent/{jockey_key}/" title="j">{jockey_key}</a></td>
<td class="txt_r">{time}</td>
<td class="txt_c">{margin}</td>
<td class="TimeIndexHeadCell01"></td>
<td class="TimeIndexMasterCell01"></td>
<td></td><td></td><td></td>
<td class="txt_c">{passage}</td>
<td>{last3f}</td>
<td>{odds}</td>
<td>{popularity}</td>
<td>{horse_weight}</td>
<td></td><td></td><td></td>
<td class="txt_l">{affiliation}<a href="/trainer/result/recent/{trainer_key}/" title="t">{trainer_key}</a></td>
<td class="txt_l"><a href="/owner/result/recent/{owner_key}/" title="o">{owner_name}</a></td>
<td>{prize}</td>
</tr>
"""

TABLE_HEADER = """
<tr>
<th>着順</th><th>枠番</th><th>馬番</th><th>馬名</th><th>性齢</th><th>斤量</th>
<th>騎手</th><th>タイム</th><th>着差</th>
<th>ﾀｲﾑ指数</th><th>ﾀｲﾑ指数M</th><th>ｽﾀｰﾄ指数</th><th>追走指数</th><th>上がり指数</th>
<th>通過</th><th>上り</th><th>単勝</th><th>人気</th><th>馬体重</th>
<th>調教ﾀｲﾑ</th><th>厩舎ｺﾒﾝﾄ</th><th>備考</th><th>調教師</th><th>馬主</th><th>賞金(万円)</th>
</tr>
"""


def build_archive_html(
    *,
    race_id: int,
    date_y: int, date_m: int, date_d: int,
    course: str = "東京", race_number: int = 11, title: str = "テストレース",
    # `class="active"` リンクの表示テキスト。既定は `course` と同じ（JRAの実挙動）。
    # NARページの再現（`Q-047` 段階②）ではレース番号タブが拾われる不具合を
    # 模すため "7R" 等を明示的に渡す
    active_link_text: str | None = None,
    surface: str = "芝", direction: str = "左", distance: int = 2000,
    course_shape: str = "",   # 「 外」「 外-内」など。実ページの表記を再現する
    meeting_no: int = 1, meeting_day: int = 1,
    # smalltxt の条件部。実ページは年齢条件が前置される（D-049）
    smalltxt_cond: str = "3歳未勝利  (混)[指](馬齢)",
    weather: str = "晴", track_condition: str = "良", post_time: str = "15:40",
    corner_nos: list[int] | None = None,
    runners: list[dict] | None = None,
    payouts: list[dict] | None = None,
    laps: list[float] | None = None,
    include_corner_table: bool = True,
    corner_row_labels: list[str] | None = None,
) -> bytes:
    surface_text = {"芝": "芝", "ダート": "ダ", "障害": "障"}[surface]
    surface_label = "芝" if surface == "芝" else "ダート"
    # 実ページの距離トークン。例: 芝右2000m / 芝右 外-内3200m / 障芝 ダート2910m
    dist_token = f"{surface_text}{direction or ''}{course_shape}{distance}m"

    header = RACE_HEADER_TMPL.format(
        race_id=race_id, course=course,
        active_link_text=active_link_text if active_link_text is not None else course,
        race_number=race_number, title=title,
        dist_token=dist_token, weather=weather,
        meeting_no=meeting_no, meeting_day=meeting_day, smalltxt_cond=smalltxt_cond,
        surface_label=surface_label, track_condition=track_condition,
        post_time=post_time, date_y=date_y, date_m=date_m, date_d=date_d,
    )

    rows = [TABLE_HEADER]
    for i, r in enumerate(runners or [], start=1):
        # horse_key は runners 内で自動的に一意にする。実データでは href が
        # 必ず馬ごとに異なるため、これを手で揃え忘れるテストの事故を防ぐ
        rows.append(RUNNER_ROW_TMPL.format(**{
            "finish": "", "frame": 1, "number": i, "horse_key": f"{9000000000 + i}",
            "name": "馬", "sex_age": "牡3", "weight_carried": "55",
            "jockey_key": "00000", "time": "", "margin": "", "passage": "",
            "last3f": "", "odds": "", "popularity": "", "horse_weight": "",
            "trainer_key": "00000", "prize": "", "affiliation": "[東]",
            "owner_key": "000000", "owner_name": "owner",
            **r,
        }))
    finish_table = (
        '<table class="race_table_01 nk_tb_common">' + "".join(rows) + "</table>"
    )

    corner_html = ""
    if include_corner_table:
        labels = corner_row_labels
        if labels is None:
            labels = [f"{n}コーナー" for n in (corner_nos or [])]
        body = "".join(f"<tr><th>{lb}</th><td>dummy</td></tr>" for lb in labels)
        corner_html = f'<table><caption>コーナー通過順位</caption>{body}</table>'

    lap_html = ""
    if laps:
        lap_str = " - ".join(f"{v:.1f}" for v in laps)
        lap_html = (
            '<table><caption>ラップタイム</caption>'
            f'<tr><th>ラップ</th><td class="race_lap_cell">{lap_str}</td></tr>'
            "</table>"
        )

    pay_html = ""
    if payouts:
        by_type: dict[str, list[dict]] = {}
        for p in payouts:
            by_type.setdefault(p["bet_type"], []).append(p)
        rows1 = []
        rows2 = []
        group1 = ["単勝", "複勝", "枠連", "馬連"]
        group2 = ["ワイド", "馬単", "三連複", "三連単"]
        for group, sink in ((group1, rows1), (group2, rows2)):
            for bt in group:
                items = by_type.get(bt, [])
                if not items:
                    continue
                combos = "<br />".join(i["combo"] for i in items)
                pays = "<br />".join(str(i["payout"]) for i in items)
                pops = "<br />".join(str(i.get("popularity", 1)) for i in items)
                sink.append(f"<tr><td>{bt}</td><td>{combos}</td><td>{pays}</td><td>{pops}</td></tr>")
        pay_html = (
            f'<table class="pay_table_01">{"".join(rows1)}</table>'
            f'<table class="pay_table_01">{"".join(rows2)}</table>'
        )

    html_doc = (
        '<html><head><meta charset="utf-8"><title>test</title></head><body>'
        + header + finish_table + corner_html + lap_html + pay_html
        + "</body></html>"
    )
    return html_doc.encode("utf-8")
