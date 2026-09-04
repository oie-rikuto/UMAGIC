"""`parse_shutuba()` の単体テスト（`Q-048` 運用予測パス）。

実ページ（2026年皐月賞ほか5レース）での検証結果（DBの確定値と一致）を
`data/`（コミット対象外）で個別に確認済み。ここでは合成fixtureで境界
ケース（等級無し・馬場状態の略記展開・欠測）を確認する。
"""

from __future__ import annotations

from datetime import date, datetime, timezone

import pytest

from tests.fixtures.build_shutuba import build_shutuba_html
from umagic.sources.base import RawPage
from umagic.sources.netkeiba import parse_shutuba

FETCHED_AT = datetime(2026, 8, 30, tzinfo=timezone.utc)


def _page(body: bytes, race_id: int = 202606030811) -> RawPage:
    return RawPage(source="netkeiba_jra", page_kind="shutuba", source_key=str(race_id),
                   url="https://race.netkeiba.com/race/shutuba.html?race_id=x",
                   body=body, encoding="unknown", fetched_at=FETCHED_AT, from_cache=False)


def test_header_fields_parsed():
    html = build_shutuba_html(
        race_id=202606030811, date_y=2026, date_m=4, date_d=19,
        course="中山", race_number=11, title="皐月賞", grade="G1",
        surface="芝", direction="右", distance=2000,
        weather="晴", track_condition="良", post_time="15:40",
        meeting_no=3, meeting_day=8, race_class="オープン", weight_rule="馬齢",
        entries=[{}, {}],
    )
    r = parse_shutuba(_page(html)).race
    assert r["date"] == date(2026, 4, 19)
    assert r["course"] == "中山"
    assert r["race_number"] == 11
    assert r["grade"] == "G1"
    assert r["surface"] == "芝"
    assert r["direction"] == "右"
    assert r["distance"] == 2000
    assert r["weather"] == "晴"
    assert r["track_condition"] == "良"
    assert r["post_time"] == "15:40"
    assert (r["meeting_no"], r["meeting_day"]) == (3, 8)
    assert r["race_class"] == "オープン"
    assert r["weight_rule"] == "馬齢"


def test_track_condition_abbreviation_expanded():
    """shutubaページは馬場状態を1文字に略す（稍→稍重、不→不良）。"""
    html = build_shutuba_html(race_id=1, date_y=2024, date_m=11, date_d=2,
                              track_condition="稍重", entries=[{}])
    assert parse_shutuba(_page(html, 1)).race["track_condition"] == "稍重"

    html2 = build_shutuba_html(race_id=1, date_y=2024, date_m=11, date_d=2,
                               track_condition="不良", entries=[{}])
    assert parse_shutuba(_page(html2, 1)).race["track_condition"] == "不良"


def test_grade_none_for_ungraded_race():
    html = build_shutuba_html(race_id=1, date_y=2024, date_m=11, date_d=2,
                              grade=None, race_class="未勝利", entries=[{}])
    r = parse_shutuba(_page(html, 1)).race
    assert r["grade"] is None
    assert r["race_class"] == "未勝利"


def test_entries_parsed_with_ids():
    html = build_shutuba_html(
        race_id=1, date_y=2024, date_m=11, date_d=2,
        entries=[
            {"horse_key": "2020123456", "horse_name": "テストホース", "sex_age": "牝4",
             "weight_carried": "54.0", "jockey_key": "01234", "jockey_name": "テスト騎手",
             "trainer_key": "05678", "trainer_name": "テスト調教師",
             "affiliation_label": "2", "affiliation_name": "栗東",
             "horse_weight": "460", "weight_diff": "-4"},
        ],
    )
    entries = parse_shutuba(_page(html, 1)).entries
    assert len(entries) == 1
    e = entries[0]
    assert e["number"] == 1
    assert e["horse_source_key"] == "2020123456"
    assert e["horse_name"] == "テストホース"
    assert e["sex"] == "牝"
    assert e["age"] == 4
    assert e["weight_carried"] == 54.0
    assert e["jockey_source_key"] == "01234"
    assert e["jockey_name"] == "テスト騎手"
    assert e["trainer_source_key"] == "05678"
    assert e["trainer_name"] == "テスト調教師"
    assert e["affiliation"] == "西"   # 栗東 → 西
    assert e["horse_weight"] == 460
    assert e["weight_diff"] == -4


def test_odds_and_popularity_not_present():
    """対象レースのオッズは特徴量にしない（`D-002`）。静的HTMLにも
    プレースホルダしか無く、そもそも取得対象ではない——`entries` に
    オッズ・人気のキーが無いことを確認する。"""
    html = build_shutuba_html(race_id=1, date_y=2024, date_m=11, date_d=2, entries=[{}])
    e = parse_shutuba(_page(html, 1)).entries[0]
    assert "odds_win" not in e
    assert "popularity" not in e


def test_n_entries_can_exceed_parsed_rows_when_scratched():
    """出馬表テーブルに現れない馬（取消等）がいても、ヘッダの `n_entries`
    はレース登録時点の頭数を保つ（実ページの実測どおり）。"""
    html = build_shutuba_html(race_id=1, date_y=2024, date_m=11, date_d=2,
                              n_entries=16, entries=[{}] * 15)
    parsed = parse_shutuba(_page(html, 1))
    assert parsed.race["n_entries"] == 16
    assert len(parsed.entries) == 15


def test_post_positions_not_drawn_raises_dedicated_error():
    """`D-196`: 枠順抽選前の出馬表は専用の例外で区別する。

    「ページが未公開」とは別の状態——待てば解消するので、呼び出し側が
    再取得を案内できるようにする。`F-801`（枠順バイアス）が確定しない
    まま予測しても意味が無い。
    """
    from umagic.sources.netkeiba import PostPositionsNotDrawn

    body = build_shutuba_html(
        race_id=202606040111, date_y=2026, date_m=9, date_d=5,
        course="中山", race_number=11, title="京成杯AH", grade="G3",
        entries=[{}, {}, {}], post_positions_drawn=False,
    )
    with pytest.raises(PostPositionsNotDrawn, match="枠順抽選"):
        parse_shutuba(_page(body, "202606040111"))


def test_course_falls_back_to_racedata02_before_payout_link_exists():
    """`D-196`: 発走前は払戻リンクが無いので `RaceData02` から会場を取る。"""
    body = build_shutuba_html(
        race_id=202606040211, date_y=2026, date_m=9, date_d=6,
        course="中山", race_number=11, title="紫苑S", grade="G2",
        meeting_no=4, meeting_day=2,
        entries=[{}, {}], with_payout_link=False,
    )
    out = parse_shutuba(_page(body, "202606040211"))
    assert out.race["course"] == "中山"
    assert out.race["meeting_no"] == 4
    assert out.race["meeting_day"] == 2


def test_foreign_trained_horse_with_empty_netkeiba_id_still_yields_name():
    """`D-199`: netkeibaに馬IDを持たない出走馬（海外調教馬等）は
    `href="…/horse/"`（ID部分が空）になり、`\\w+`前提の正規表現だと
    行全体が不一致になって**存在する馬名まで失われる**。

    実データ（セントウルS2026、外国馬「ファストネットワーク」・
    調教師「イプ」）で発見した欠陥の再現。同じ構造は
    `/trainer/result/recent//`（トレーナーID空）にも及ぶ。
    """
    body = build_shutuba_html(
        race_id=202609040211, date_y=2026, date_m=9, date_d=6,
        course="阪神", race_number=11, title="セントウルS", grade="G2",
        entries=[
            {"number": 1, "horse_key": "2020123456", "horse_name": "普通の馬",
             "trainer_key": "01234", "trainer_name": "普通の調教師"},
            {"number": 2, "horse_key": "", "horse_name": "ファストネットワーク",
             "sex_age": "セ6", "jockey_key": "05585", "jockey_name": "レーン",
             "affiliation_label": "3", "affiliation_name": "地方",
             "trainer_key": "", "trainer_name": "イプ"},
        ],
    )
    entries = parse_shutuba(_page(body, "202609040211")).entries
    assert entries[0]["horse_name"] == "普通の馬"
    assert entries[0]["horse_source_key"] == "2020123456"

    foreign = entries[1]
    assert foreign["horse_name"] == "ファストネットワーク"
    assert foreign["horse_source_key"] is None  # 空IDは None に正規化される
    assert foreign["trainer_name"] == "イプ"
    assert foreign["trainer_source_key"] is None
    assert foreign["jockey_name"] == "レーン"  # 通常どおりIDありのケース
    assert foreign["affiliation"] is None  # 「地方」は東西どちらにも属さない
