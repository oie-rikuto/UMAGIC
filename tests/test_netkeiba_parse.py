"""`002-loader.md` の単体テスト観点1〜10e、および day_index。"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from tests.fixtures.build_archive import build_archive_html
from umagic.sources.base import RawPage
from umagic.sources.netkeiba import list_race_keys, parse_archive

FETCHED_AT = datetime(2026, 8, 16, tzinfo=timezone.utc)


def _page(body: bytes, race_id: int = 1, page_kind="archive") -> RawPage:
    return RawPage(source="netkeiba_jra", page_kind=page_kind, source_key=str(race_id),
                   url="https://db.netkeiba.com/race/x/", body=body, encoding="unknown",
                   fetched_at=FETCHED_AT, from_cache=False)


def _runner(**kw):
    base = dict(finish="1", frame=1, number=1, name="馬", passage="")
    base.update(kw)
    return base


# --- 1〜4: 着順欄 -----------------------------------------------------------

def test_1_demotion_marker():
    html = build_archive_html(race_id=1, date_y=2023, date_m=1, date_d=1,
                              corner_nos=[1, 2, 3, 4],
                              runners=[_runner(finish="4(降)", passage="1-1-1-1")])
    pr = parse_archive(_page(html))
    assert pr.runners[0]["status"] == "降着"
    assert pr.runners[0]["finish_pos"] == 4


def test_2_dnf_marker():
    html = build_archive_html(race_id=1, date_y=2023, date_m=1, date_d=1,
                              corner_nos=[1, 2, 3, 4],
                              runners=[_runner(finish="中", passage="")])
    pr = parse_archive(_page(html))
    assert pr.runners[0]["status"] == "競走中止"
    assert pr.runners[0]["finish_pos"] is None


def test_3_scratched_marker():
    html = build_archive_html(race_id=1, date_y=2023, date_m=1, date_d=1,
                              corner_nos=[1, 2, 3, 4],
                              runners=[_runner(finish="除", passage="")])
    pr = parse_archive(_page(html))
    assert pr.runners[0]["status"] == "競走除外"
    assert pr.runners[0]["finish_pos"] is None


def test_4_unknown_marker_rejected():
    """失格の表記は Q-023 のとおり3年分の実データでも1件も観測されておらず未確認。"""
    html = build_archive_html(race_id=1, date_y=2023, date_m=1, date_d=1,
                              corner_nos=[1, 2, 3, 4],
                              runners=[_runner(finish="失", passage="")])
    pr = parse_archive(_page(html))
    assert len(pr.runners) == 0
    assert len(pr.rejected) == 1
    assert pr.rejected[0].reason == "unknown_finish_marker"


def test_3b_scratched_marker():
    """D-048: 出走取消は `取` の1文字。P-0 の3年分取り込みで220件確認した。"""
    html = build_archive_html(race_id=1, date_y=2023, date_m=1, date_d=1,
                              corner_nos=[1, 2, 3, 4],
                              runners=[_runner(finish="取", passage="")])
    pr = parse_archive(_page(html))
    assert pr.runners[0]["status"] == "出走取消"
    assert pr.runners[0]["finish_pos"] is None
    assert len(pr.rejected) == 0


# --- 5〜6: 馬体重 ------------------------------------------------------------

def test_5_weight_unmeasured():
    html = build_archive_html(race_id=1, date_y=2023, date_m=1, date_d=1,
                              corner_nos=[1, 2, 3, 4],
                              runners=[_runner(passage="1-1-1-1", horse_weight="計不")])
    pr = parse_archive(_page(html))
    assert pr.runners[0]["horse_weight"] is None
    assert pr.runners[0]["weight_diff"] is None


def test_6_weight_parsed():
    html = build_archive_html(race_id=1, date_y=2023, date_m=1, date_d=1,
                              corner_nos=[1, 2, 3, 4],
                              runners=[_runner(passage="1-1-1-1", horse_weight="490(-2)")])
    pr = parse_archive(_page(html))
    assert pr.runners[0]["horse_weight"] == 490
    assert pr.runners[0]["weight_diff"] == -2


# --- 7〜8: タイム・単勝 ------------------------------------------------------

def test_7_time_parsed():
    html = build_archive_html(race_id=1, date_y=2023, date_m=1, date_d=1,
                              corner_nos=[1, 2, 3, 4],
                              runners=[_runner(passage="1-1-1-1", time="1:08.0")])
    pr = parse_archive(_page(html))
    assert pr.runners[0]["time_sec"] == 68.0


def test_8_odds_dash_is_null():
    html = build_archive_html(race_id=1, date_y=2023, date_m=1, date_d=1,
                              corner_nos=[1, 2, 3, 4],
                              runners=[_runner(passage="1-1-1-1", odds="---")])
    pr = parse_archive(_page(html))
    assert pr.runners[0]["odds_win"] is None


# --- 9〜10e: コーナー通過順 --------------------------------------------------

def test_9_straight_course_no_corners():
    html = build_archive_html(race_id=1, date_y=2023, date_m=1, date_d=1,
                              corner_nos=[], include_corner_table=True,
                              runners=[_runner(passage="13")])
    pr = parse_archive(_page(html))
    assert pr.race["corner_nos"] == []
    assert pr.runners[0]["corners"] == []  # 通過列の "13" は採らない


def test_10_empty_passage_is_null():
    html = build_archive_html(race_id=1, date_y=2023, date_m=1, date_d=1,
                              corner_nos=[1, 2, 3, 4],
                              runners=[_runner(passage="")])
    pr = parse_archive(_page(html))
    assert pr.runners[0]["corners"] is None


def test_10b_length_mismatch_row_kept_corners_null():
    """D-044: 要素数不一致は行を棄却せず corners=NULL として取り込む。"""
    html = build_archive_html(race_id=1, date_y=2023, date_m=1, date_d=1,
                              corner_nos=[1, 2, 3, 4],
                              runners=[_runner(passage="3-2")])
    pr = parse_archive(_page(html))
    assert len(pr.runners) == 1
    assert pr.runners[0]["corners"] is None
    assert any(r.reason == "corners_length_mismatch" for r in pr.rejected)


def test_10c_two_corner_course():
    html = build_archive_html(race_id=1, date_y=2023, date_m=1, date_d=1,
                              corner_nos=[3, 4],
                              runners=[_runner(passage="3-2")])
    pr = parse_archive(_page(html))
    assert pr.race["corner_nos"] == [3, 4]
    assert pr.runners[0]["corners"] == [3, 2]


def test_10d_dnf_empty_passage_is_normal():
    """D-044: 完走しなかった馬の空 corners は正常。rejected_rows に乗らない。"""
    html = build_archive_html(race_id=1, date_y=2023, date_m=1, date_d=1,
                              corner_nos=[1, 2, 3, 4],
                              runners=[_runner(finish="中", passage="")])
    pr = parse_archive(_page(html))
    assert pr.runners[0]["corners"] is None
    assert pr.runners[0]["status"] == "競走中止"
    assert len(pr.rejected) == 0


def test_10e_unparsable_corner_header():
    html = build_archive_html(race_id=1, date_y=2023, date_m=1, date_d=1,
                              corner_nos=None, corner_row_labels=["謎コーナー"],
                              runners=[_runner(passage="1-1")])
    pr = parse_archive(_page(html))
    assert pr.race["corner_nos"] is None
    assert pr.runners[0]["corners"] is None
    assert any(r.reason == "corner_header_unparsed" for r in pr.rejected)


# --- 11: エンティティ同定は test_ids.py ------------------------------------
# --- 12: キャッシュヒットは test_cache.py -----------------------------------


# --- 直線競走・ダート・方向表記 ----------------------------------------------

@pytest.mark.parametrize("surface,direction,shape,distance,expect", [
    ("芝", "右", "", 2000, ("芝", "右", 2000)),
    ("芝", "左", "", 2400, ("芝", "左", 2400)),
    ("ダート", "右", "", 1800, ("ダート", "右", 1800)),
    ("芝", "直線", "", 1000, ("芝", "直線", 1000)),
    # 中山・新潟の外回り
    ("芝", "右", " 外", 1200, ("芝", "右", 1200)),
    # 阪神・京都の3200m は外回りから内回りへ入る。`外` か `内` の一方しか
    # 見ない正規表現では松籟S（202209010609）が読めず取り込みが落ちた
    ("芝", "右", " 外-内", 3200, ("芝", "右", 3200)),
    # 中山3600m は1周が短く2周する。「距離の直前は数字以外」という前提が
    # 周回数の "2" で破れ、ステイヤーズS（202206050111）が読めず落ちた
    ("芝", "右", " 内2周", 3600, ("芝", "右", 3600)),
])
def test_distance_notations(surface, direction, shape, distance, expect):
    html = build_archive_html(race_id=1, date_y=2023, date_m=1, date_d=1,
                              surface=surface, direction=direction,
                              course_shape=shape, distance=distance,
                              corner_nos=[1, 2, 3, 4],
                              runners=[_runner(passage="1-1-1-1")])
    r = parse_archive(_page(html)).race
    assert (r["surface"], r["direction"], r["distance"]) == expect


def test_jump_race_distance_is_not_parsed():
    """D-025 / D-047: 障害は day_index で除外するが、すり抜けてもここで読めない。

    表記「障芝 ダート2910m」は `芝` も `ダ` も内側に含むため、部分一致では
    誤って拾ってしまう。読めてしまうと `surface='芝'` の平地レースとして
    静かに学習データへ混入する。
    """
    html = build_archive_html(race_id=1, date_y=2023, date_m=1, date_d=1,
                              surface="障害", direction="", course_shape="芝 ダート",
                              distance=2910, corner_nos=[1, 2, 3, 4],
                              runners=[_runner(passage="1-1-1-1")])
    r = parse_archive(_page(html)).race
    assert r["distance"] is None
    assert r["surface"] is None


def test_dirt_and_direction_parsed():
    html = build_archive_html(race_id=1, date_y=2023, date_m=1, date_d=1,
                              surface="ダート", direction="左", distance=1600,
                              corner_nos=[3, 4], runners=[_runner(passage="1-1")])
    pr = parse_archive(_page(html))
    assert pr.race["surface"] == "ダート"
    assert pr.race["direction"] == "左"
    assert pr.race["distance"] == 1600


def test_header_fields_and_headcounts():
    html = build_archive_html(
        race_id=999, date_y=2024, date_m=3, date_d=10, course="中山", race_number=5,
        surface="芝", direction="右", distance=1200, weather="曇", track_condition="稍重",
        post_time="14:35", corner_nos=[3, 4],
        runners=[_runner(finish="1", number=1, passage="1-1"),
                 _runner(finish="中", number=2, passage="")],
    )
    pr = parse_archive(_page(html, race_id=999))
    r = pr.race
    assert (r["date"].year, r["date"].month, r["date"].day) == (2024, 3, 10)
    assert r["course"] == "中山"
    assert r["race_number"] == 5
    assert r["weather"] == "曇"
    assert r["track_condition"] == "稍重"
    assert r["post_time"] == "14:35"
    assert r["n_entries"] == 2
    assert r["n_starters"] == 2  # 競走中止もゲートは出ている


def test_empty_template_detected_no_race_table():
    html = b"<html><body>no race data here</body></html>"
    pr = parse_archive(_page(html))
    assert pr.race["course"] is None
    assert pr.runners == []


# --- payouts / laps ---------------------------------------------------------

def test_payouts_combination_normal_form():
    html = build_archive_html(
        race_id=1, date_y=2023, date_m=1, date_d=1, corner_nos=[1, 2, 3, 4],
        runners=[_runner(passage="1-1-1-1")],
        payouts=[
            {"bet_type": "単勝", "combo": "5", "payout": 800, "popularity": 2},
            {"bet_type": "馬連", "combo": "9 - 5", "payout": 690, "popularity": 3},
            {"bet_type": "馬単", "combo": "5 → 9", "payout": 2330, "popularity": 6},
            {"bet_type": "三連複", "combo": "9 - 5 - 3", "payout": 4700, "popularity": 12},
            {"bet_type": "三連単", "combo": "5 → 9 → 3", "payout": 29810, "popularity": 79},
        ],
    )
    pr = parse_archive(_page(html))
    by_type = {p["bet_type"]: p for p in pr.payouts}
    assert by_type["馬連"]["combination"] == [5, 9]        # 昇順に正規化
    assert by_type["馬単"]["combination"] == [5, 9]        # 着順のまま
    assert by_type["三連複"]["combination"] == [3, 5, 9]   # 昇順
    assert by_type["三連単"]["combination"] == [5, 9, 3]   # 着順のまま
    assert by_type["馬連"]["comb_key"] == "5-9"


def test_laps_parsed_in_order():
    html = build_archive_html(race_id=1, date_y=2023, date_m=1, date_d=1,
                              corner_nos=[1, 2, 3, 4], runners=[_runner(passage="1-1-1-1")],
                              laps=[12.6, 10.7, 12.0, 12.6])
    pr = parse_archive(_page(html))
    assert [l["furlong_no"] for l in pr.laps] == [1, 2, 3, 4]
    assert [l["lap_sec"] for l in pr.laps] == [12.6, 10.7, 12.0, 12.6]


# --- day_index ---------------------------------------------------------------

def _day_index_entry(race_id: str, title: str, surface_text: str) -> str:
    return (
        f'<dl><dt>1R</dt><dd><a href="/race/{race_id}/" title="{title}">{title}</a>'
        f'<br/><div>{surface_text}<br/></div></dd></dl>'
    )


def test_list_race_keys_jra_only():
    body = (
        '<html><head><meta charset="utf-8"></head><body>'
        '<h3>中央</h3><div>'
        + _day_index_entry("202305021201", "3歳未勝利", "ダ1600m")
        + _day_index_entry("202308011201", "3歳未勝利", "芝1600m")
        + "</div>"
        + '<h3>地方</h3><div>'
        + _day_index_entry("202336052801", "C2", "ダ850m")
        + "</div></body></html>"
    ).encode("utf-8")
    keys = list_race_keys(_page(body, page_kind="day_index"))
    assert keys == ["202305021201", "202308011201"]


def test_list_race_keys_excludes_jump_races():
    """D-025: 障害レースはスコープ外。day_index の距離表記 "障..." で弾く。"""
    body = (
        '<html><head><meta charset="utf-8"></head><body>'
        '<h3>中央</h3><div>'
        + _day_index_entry("202305021204", "3歳未勝利", "芝1600m")
        + _day_index_entry("202308011204", "障害4歳以上未勝利", "障2910m")
        + "</div></body></html>"
    ).encode("utf-8")
    keys = list_race_keys(_page(body, page_kind="day_index"))
    assert keys == ["202305021204"]


def test_list_race_keys_no_central_section():
    body = "<html><body>no races today</body></html>".encode("utf-8")
    assert list_race_keys(_page(body, page_kind="day_index")) == []


# --- D-049: smalltxt / 所属 -------------------------------------------------

@pytest.mark.parametrize("cond,expect_class,expect_rule", [
    ("3歳オープン  (国際) 牡・牝(指)(定量)", "オープン", "定量"),
    ("4歳以上1勝クラス  (混)[指](ハンデ)", "1勝クラス", "ハンデ"),
    ("3歳未勝利  (混)[指](馬齢)", "未勝利", "馬齢"),
    ("2歳新馬  (混)(馬齢)", "新馬", "馬齢"),
    ("3歳以上2勝クラス  (混)(特指)(定量)", "2勝クラス", "定量"),
    ("4歳以上3勝クラス  (混)(ハンデ)", "3勝クラス", "ハンデ"),
    ("3歳以上オープン  (国際)(特指)(別定)", "オープン", "別定"),
])
def test_3c_smalltxt_class_and_weight_rule(cond, expect_class, expect_rule):
    """年齢条件が前置されるため、部分一致でクラスを採る（D-049）。"""
    html = build_archive_html(race_id=1, date_y=2023, date_m=1, date_d=1,
                              corner_nos=[1, 2, 3, 4], smalltxt_cond=cond,
                              runners=[_runner(passage="1-1-1-1")])
    r = parse_archive(_page(html)).race
    assert r["race_class"] == expect_class
    assert r["weight_rule"] == expect_rule


def test_3d_meeting_no_and_day():
    html = build_archive_html(race_id=1, date_y=2023, date_m=1, date_d=1,
                              course="中山", meeting_no=2, meeting_day=12,
                              corner_nos=[1, 2, 3, 4],
                              runners=[_runner(passage="1-1-1-1")])
    r = parse_archive(_page(html)).race
    assert (r["meeting_no"], r["meeting_day"]) == (2, 12)


def test_3e_affiliation_parsed():
    html = build_archive_html(race_id=1, date_y=2023, date_m=1, date_d=1,
                              corner_nos=[1, 2, 3, 4],
                              runners=[_runner(passage="1-1-1-1", affiliation="[西]")])
    pr = parse_archive(_page(html))
    assert pr.runners[0]["affiliation"] == "西"


def test_affiliation_absent_is_null():
    html = build_archive_html(race_id=1, date_y=2023, date_m=1, date_d=1,
                              corner_nos=[1, 2, 3, 4],
                              runners=[_runner(passage="1-1-1-1", affiliation="")])
    pr = parse_archive(_page(html))
    assert pr.runners[0]["affiliation"] is None


def test_unknown_race_class_recorded_but_race_kept():
    """D-049: クラスが対応表に無くてもレースは取り込み、rejected_rows に残す。"""
    html = build_archive_html(race_id=1, date_y=2023, date_m=1, date_d=1,
                              corner_nos=[1, 2, 3, 4],
                              smalltxt_cond="3歳4勝クラス  (混)(定量)",
                              runners=[_runner(passage="1-1-1-1")])
    pr = parse_archive(_page(html))
    assert pr.race["race_class"] is None
    assert pr.race["weight_rule"] == "定量"      # 斤量条件は取れる
    assert len(pr.runners) == 1                   # 行は捨てない
    assert any(r.reason == "unknown_race_class" for r in pr.rejected)


# --- D-050: 血統 -------------------------------------------------------------

def _ped_html(pairs):
    """(rowspan, horse_key) の並びから血統表を組み立てる。"""
    cells = "".join(
        f'<td rowspan="{rs}" class="b_ml"><a href="https://db.netkeiba.com/horse/{k}/">馬{k}</a></td>'
        for rs, k in pairs
    )
    return ('<html><head><meta charset="utf-8"></head><body>'
            f'<table class="blood_table detail"><tr>{cells}</tr></table>'
            "</body></html>").encode("utf-8")


def _ped_page(body, key="2020103532"):
    return RawPage(source="netkeiba_jra", page_kind="horse_ped", source_key=key,
                   url="x", body=body, encoding="unknown",
                   fetched_at=FETCHED_AT, from_cache=False)


def test_pedigree_extracts_sire_dam_damsire():
    """1つ目の rowspan=16 が父、2つ目が母、母の後の最初の rowspan=8 が母父。"""
    from umagic.sources.netkeiba import parse_pedigree
    body = _ped_html([
        (16, "SIRE"), (8, "SIRE_S"), (4, "x1"), (2, "x2"),   # 父系
        (16, "DAM"), (8, "DAMSIRE"), (4, "y1"), (2, "y2"),   # 母系
    ])
    ped = parse_pedigree(_ped_page(body))
    assert ped["sire_key"] == "SIRE"
    assert ped["dam_key"] == "DAM"
    assert ped["damsire_key"] == "DAMSIRE"


def test_pedigree_damsire_taken_after_dam_not_before():
    """父側にも rowspan=8 があるため、母より前のものを拾ってはならない。"""
    from umagic.sources.netkeiba import parse_pedigree
    body = _ped_html([(16, "SIRE"), (8, "父の父"), (16, "DAM"), (8, "母の父")])
    ped = parse_pedigree(_ped_page(body))
    assert ped["damsire_key"] == "母の父"


def test_pedigree_missing_table_returns_nulls():
    """D-050: 解釈できなければ推測で補わず None を返す。"""
    from umagic.sources.netkeiba import parse_pedigree
    body = b'<html><head><meta charset="utf-8"></head><body>no pedigree</body></html>'
    ped = parse_pedigree(_ped_page(body))
    assert ped["sire_key"] is None and ped["dam_key"] is None and ped["damsire_key"] is None


def test_pedigree_foreign_ancestor_key():
    """外国産の先祖は 000a001fb6 のような非数値キーを持つ。"""
    from umagic.sources.netkeiba import parse_pedigree
    body = _ped_html([(16, "2012104668"), (8, "000a001fb6"), (16, "2014106097"), (8, "1998101554")])
    ped = parse_pedigree(_ped_page(body))
    assert ped["sire_key"] == "2012104668"
    assert ped["damsire_key"] == "1998101554"


# --- D-057: 騎手名・調教師名 ------------------------------------------------

def test_jockey_and_trainer_names_parsed():
    html = build_archive_html(race_id=1, date_y=2023, date_m=1, date_d=1,
                              corner_nos=[1, 2, 3, 4],
                              runners=[_runner(passage="1-1-1-1")])
    r = parse_archive(_page(html)).runners[0]
    assert r["jockey_name"] == "00000"      # fixture はキーを表示名にしている
    assert r["trainer_name"] == "00000"
    assert r["jockey_source_key"] == "00000"


def test_trainer_name_excludes_affiliation_marker():
    """調教師欄は先頭に [東] 等が付く。名前に混ぜない（D-057）。"""
    from umagic.sources.netkeiba import _parse_link_name
    cell = '[西]<a href="/trainer/result/recent/01070/" title="堀宣行">堀宣行</a>'
    assert _parse_link_name(cell) == "堀宣行"


def test_link_name_absent_is_null():
    from umagic.sources.netkeiba import _parse_link_name
    assert _parse_link_name("[東]") is None
