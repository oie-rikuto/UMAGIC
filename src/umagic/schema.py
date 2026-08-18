"""中間スキーマ（`docs/spec/001-schema.md`）。

ソース非依存の9テーブルを定義する（`D-009` / `D-057`）。作成順は外部キーの依存による:
`source_ids` → `horses` → `jockeys` → `trainers` → `races` → `runners` →
`payouts` → `odds` → `laps`（`D-057`）。

運用テーブル（`fetch_log` / `rejected_rows` / `quality_runs` / `quality_findings`）は
ここでは定義しない。`source` / `fetched_at` を持たないため対象外（`D-046`）。
"""

from __future__ import annotations

import duckdb

DDL_STATEMENTS: list[str] = [
    """
    CREATE TABLE source_ids (
        entity_type  VARCHAR   NOT NULL,   -- 'race' | 'horse' | 'jockey' | 'trainer'
        internal_id  BIGINT    NOT NULL,
        source       VARCHAR   NOT NULL,
        source_key   VARCHAR   NOT NULL,   -- ソース側のID。netkeiba のレースIDは12桁
        fetched_at   TIMESTAMP NOT NULL,
        PRIMARY KEY (entity_type, source, source_key),
        CHECK (entity_type IN ('race', 'horse', 'jockey', 'trainer'))
    )
    """,
    """
    CREATE UNIQUE INDEX ux_source_ids_internal
        ON source_ids (entity_type, internal_id, source)
    """,
    """
    CREATE TABLE horses (
        horse_id    BIGINT    PRIMARY KEY,
        name        VARCHAR   NOT NULL,
        birth       DATE,
        sire_id     BIGINT,
        dam_id      BIGINT,
        damsire_id  BIGINT,
        source      VARCHAR   NOT NULL,
        fetched_at  TIMESTAMP NOT NULL
    )
    """,
    """
    CREATE TABLE jockeys (
        jockey_id   BIGINT    PRIMARY KEY,
        name        VARCHAR   NOT NULL,
        source      VARCHAR   NOT NULL,
        fetched_at  TIMESTAMP NOT NULL
    )
    """,
    """
    CREATE TABLE trainers (
        trainer_id  BIGINT    PRIMARY KEY,
        name        VARCHAR   NOT NULL,
        source      VARCHAR   NOT NULL,
        fetched_at  TIMESTAMP NOT NULL
    )
    """,
    """
    CREATE TABLE races (
        race_id           BIGINT    PRIMARY KEY,
        date              DATE      NOT NULL,
        course            VARCHAR   NOT NULL,
        race_number       SMALLINT  NOT NULL,
        post_time         TIME,
        distance          SMALLINT  NOT NULL,
        surface           VARCHAR   NOT NULL,
        direction         VARCHAR,
        grade             VARCHAR,
        track_condition   VARCHAR,
        weather           VARCHAR,
        weather_forecast  VARCHAR,
        n_entries         SMALLINT  NOT NULL,
        n_starters        SMALLINT  NOT NULL,
        prize             BIGINT,
        corner_nos        SMALLINT[],
        race_class        VARCHAR,
        weight_rule       VARCHAR,
        meeting_no        SMALLINT,
        meeting_day       SMALLINT,
        source            VARCHAR   NOT NULL,
        fetched_at        TIMESTAMP NOT NULL,
        UNIQUE (date, course, race_number),
        CHECK (n_entries >= n_starters),
        CHECK (n_starters >= 0),
        CHECK (race_number >= 1),
        CHECK (distance > 0),
        CHECK (surface IN ('芝', 'ダート', '障害')),
        CHECK (direction IS NULL OR direction IN ('右', '左', '直線')),
        CHECK (track_condition IS NULL
               OR track_condition IN ('良', '稍重', '重', '不良')),
        CHECK (race_class IS NULL OR race_class IN
               ('新馬', '未勝利', '1勝クラス', '2勝クラス', '3勝クラス', 'オープン')),
        CHECK (weight_rule IS NULL OR weight_rule IN ('馬齢', '定量', '別定', 'ハンデ')),
        CHECK (meeting_no IS NULL OR meeting_no >= 1),
        CHECK (meeting_day IS NULL OR meeting_day >= 1)
    )
    """,
    """
    CREATE TABLE runners (
        race_id         BIGINT    NOT NULL REFERENCES races(race_id),
        horse_id        BIGINT    NOT NULL REFERENCES horses(horse_id),
        number          SMALLINT  NOT NULL,
        frame           SMALLINT,
        jockey_id       BIGINT,
        trainer_id      BIGINT,
        weight_carried  DECIMAL(4,1),
        horse_weight    SMALLINT,
        weight_diff     SMALLINT,
        age             SMALLINT,
        sex             VARCHAR,
        odds_win        DECIMAL(7,1),
        popularity      SMALLINT,
        status          VARCHAR   NOT NULL,
        finish_pos      SMALLINT,
        margin          VARCHAR,
        time_sec        DECIMAL(6,1),
        last_3f         DECIMAL(4,1),
        corners         SMALLINT[],
        affiliation     VARCHAR,
        source          VARCHAR   NOT NULL,
        fetched_at      TIMESTAMP NOT NULL,
        PRIMARY KEY (race_id, horse_id),
        UNIQUE (race_id, number),
        CHECK (status IN ('出走', '降着', '競走中止', '失格', '出走取消', '競走除外')),
        CHECK (number >= 1),
        CHECK (finish_pos IS NULL OR finish_pos >= 1),
        CHECK (affiliation IS NULL OR affiliation IN ('東', '西', '地', '外'))
    )
    """,
    """
    CREATE TABLE payouts (
        race_id      BIGINT     NOT NULL REFERENCES races(race_id),
        bet_type     VARCHAR    NOT NULL,
        comb_key     VARCHAR    NOT NULL,
        combination  SMALLINT[] NOT NULL,
        payout       INTEGER    NOT NULL,
        popularity   SMALLINT,
        source       VARCHAR    NOT NULL,
        fetched_at   TIMESTAMP  NOT NULL,
        PRIMARY KEY (race_id, bet_type, comb_key),
        CHECK (payout >= 0)
    )
    """,
    """
    CREATE TABLE odds (
        race_id      BIGINT       NOT NULL REFERENCES races(race_id),
        bet_type     VARCHAR      NOT NULL,
        comb_key     VARCHAR      NOT NULL,
        combination  SMALLINT[]   NOT NULL,
        odds_low     DECIMAL(8,1) NOT NULL,
        odds_high    DECIMAL(8,1) NOT NULL,
        as_of        TIMESTAMP    NOT NULL,
        source       VARCHAR      NOT NULL,
        fetched_at   TIMESTAMP    NOT NULL,
        PRIMARY KEY (race_id, bet_type, comb_key, as_of),
        CHECK (odds_high >= odds_low),
        CHECK (odds_low > 0)
    )
    """,
    """
    CREATE TABLE laps (
        race_id     BIGINT       NOT NULL REFERENCES races(race_id),
        furlong_no  SMALLINT     NOT NULL,
        lap_sec     DECIMAL(4,1) NOT NULL,
        source      VARCHAR      NOT NULL,
        fetched_at  TIMESTAMP    NOT NULL,
        PRIMARY KEY (race_id, furlong_no),
        CHECK (furlong_no >= 1),
        CHECK (lap_sec > 0)
    )
    """,
]

# 作成順どおりのテーブル名。テストと検査で参照する。
TABLE_NAMES: list[str] = [
    "source_ids", "horses", "jockeys", "trainers",
    "races", "runners", "payouts", "odds", "laps",
]


def create_schema(conn: duckdb.DuckDBPyConnection) -> None:
    """中間スキーマの全テーブルを作成する。既存テーブルがあればエラーになる。"""
    for stmt in DDL_STATEMENTS:
        conn.execute(stmt)
