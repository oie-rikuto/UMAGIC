"""データ品質検査（`docs/spec/012-data-quality.md`）。

`fail` 系9件・`warn` 系5件を実行し、`quality_runs` / `quality_findings` に
記録したうえで Markdown レポートと終了コードを返す。不良を見つけても
削除・上書きしない（`D-041`）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timezone

import duckdb

# --- fail 系（R-014 の8項目と対応。対応は各 SQL のコメントに明記） -----------

FAIL_CHECKS: dict[str, str] = {
    # R-014: 頭数の整合
    "headcount_starters": """
        SELECT r.race_id, NULL AS horse_id,
               'n_starters=' || r.n_starters || ' actual=' ||
               COUNT(ru.horse_id) FILTER (WHERE ru.status IN ('出走','降着','競走中止','失格')) AS detail
        FROM races r LEFT JOIN runners ru USING (race_id)
        GROUP BY r.race_id, r.n_starters
        HAVING r.n_starters <> COUNT(ru.horse_id)
               FILTER (WHERE ru.status IN ('出走','降着','競走中止','失格'))
    """,
    # R-014: 出馬表との整合
    "headcount_entries": """
        SELECT r.race_id, NULL AS horse_id,
               'n_entries=' || r.n_entries || ' actual=' || COUNT(ru.horse_id) AS detail
        FROM races r LEFT JOIN runners ru USING (race_id)
        GROUP BY r.race_id, r.n_entries
        HAVING r.n_entries <> COUNT(ru.horse_id)
    """,
    # R-014: 頭数の整合／出馬表との整合（runners が0行のレースはどちらも
    # INNER JOIN では検出できないため、この検査が拾う）
    "orphan_race": """
        SELECT r.race_id, NULL AS horse_id, 'runners が0行' AS detail
        FROM races r
        WHERE NOT EXISTS (SELECT 1 FROM runners ru WHERE ru.race_id = r.race_id)
    """,
    # R-014: 着順の整合
    "finish_pos_rank": """
        WITH ranked AS (
            SELECT race_id, horse_id, finish_pos,
                   RANK() OVER (PARTITION BY race_id ORDER BY finish_pos) AS expected
            FROM runners WHERE finish_pos IS NOT NULL
        )
        SELECT race_id, horse_id,
               'finish_pos=' || finish_pos || ' expected=' || expected AS detail
        FROM ranked WHERE finish_pos <> expected
    """,
    # R-014: 着順と状態の整合
    "status_columns": """
        SELECT race_id, horse_id, 'status=' || status AS detail
        FROM runners
        WHERE (status IN ('出走','降着') AND (finish_pos IS NULL OR time_sec IS NULL))
           OR (status IN ('競走中止','失格','出走取消','競走除外') AND finish_pos IS NOT NULL)
    """,
    # R-014: 着順マーカーの網羅
    "status_domain": """
        SELECT race_id, horse_id, 'status=' || status AS detail
        FROM runners
        WHERE status NOT IN ('出走','降着','競走中止','失格','出走取消','競走除外')
    """,
    # R-014: 払戻の整合
    "payout_horses": """
        SELECT p.race_id, NULL AS horse_id,
               p.bet_type || ' ' || p.comb_key || ' 馬番' || n AS detail
        FROM payouts p, UNNEST(p.combination) AS t(n)
        WHERE p.bet_type <> '枠連'
          AND NOT EXISTS (
              SELECT 1 FROM runners ru
              WHERE ru.race_id = p.race_id AND ru.number = n
                AND ru.status IN ('出走','降着','競走中止','失格'))
    """,
    # R-014: オッズと人気の単調性
    "odds_monotonic": """
        WITH o AS (
            SELECT race_id, horse_id, popularity, odds_win,
                   LAG(odds_win) OVER (PARTITION BY race_id ORDER BY popularity) AS prev
            FROM runners WHERE popularity IS NOT NULL AND odds_win IS NOT NULL
        )
        SELECT race_id, horse_id,
               'odds_win=' || odds_win || ' prev(popularity低)=' || prev AS detail
        FROM o WHERE prev IS NOT NULL AND odds_win < prev
    """,
    # R-014: コーナー通過順の整合。完走馬のみ対象（D-044）。
    # 要素数どうしではなく races.corner_nos との比較にする（D-043）
    "corners_uniform": """
        SELECT ru.race_id, ru.horse_id,
               'len(corners)=' || len(ru.corners) || ' len(corner_nos)=' || len(r.corner_nos) AS detail
        FROM runners ru JOIN races r USING (race_id)
        WHERE ru.corners IS NOT NULL AND r.corner_nos IS NOT NULL
          AND ru.status IN ('出走','降着') AND len(ru.corners) <> len(r.corner_nos)
    """,
}

# --- warn 系（分子・分母を明示。年代別にも分解する） -------------------------

WARN_CHECKS: dict[str, tuple[str, str]] = {
    "corners_missing": (
        """
        SELECT COUNT(*) FROM runners ru JOIN races r USING (race_id)
        WHERE ru.status IN ('出走','降着') AND r.corner_nos IS NOT NULL
          AND len(r.corner_nos) > 0 AND ru.corners IS NULL
        """,
        """
        SELECT COUNT(*) FROM runners ru JOIN races r USING (race_id)
        WHERE ru.status IN ('出走','降着') AND r.corner_nos IS NOT NULL
          AND len(r.corner_nos) > 0
        """,
    ),
    "rejected_rate": (
        "SELECT COUNT(*) FROM rejected_rows",
        "SELECT (SELECT COUNT(*) FROM runners) + (SELECT COUNT(*) FROM rejected_rows)",
    ),
    "fetch_incomplete": (
        "SELECT COUNT(*) FROM fetch_log WHERE outcome <> 'ok'",
        "SELECT COUNT(*) FROM fetch_log",
    ),
    "odds_coverage": (
        "SELECT COUNT(DISTINCT race_id) FROM odds",
        "SELECT COUNT(*) FROM races",
    ),
    "laps_coverage": (
        "SELECT COUNT(*) FROM races r WHERE NOT EXISTS "
        "(SELECT 1 FROM laps l WHERE l.race_id = r.race_id)",
        "SELECT COUNT(*) FROM races",
    ),
}


@dataclass
class QualityReport:
    run_id: int
    fail_counts: dict[str, int] = field(default_factory=dict)
    warn_rates: dict[str, tuple[int, int]] = field(default_factory=dict)
    n_races: int = 0
    n_runners: int = 0

    @property
    def n_fail(self) -> int:
        return sum(self.fail_counts.values())

    @property
    def exit_code(self) -> int:
        return 0 if self.n_fail == 0 else 1

    def to_markdown(self) -> str:
        lines = [f"# 品質レポート（run_id={self.run_id}）", ""]
        lines.append(f"対象: races={self.n_races} runners={self.n_runners}")
        lines.append("")
        lines.append("## fail")
        for check_id, n in self.fail_counts.items():
            lines.append(f"- `{check_id}`: {n}件")
        lines.append("")
        lines.append("## warn")
        for check_id, (num, den) in self.warn_rates.items():
            rate = f"{100 * num / den:.1f}%" if den else "n/a"
            lines.append(f"- `{check_id}`: {num}/{den} ({rate})")
        return "\n".join(lines)


def run_quality_checks(
    conn: duckdb.DuckDBPyConnection,
    *,
    scope_from: date | None = None,
    scope_to: date | None = None,
) -> QualityReport:
    started_at = datetime.now(timezone.utc)
    n_races = conn.execute("SELECT COUNT(*) FROM races").fetchone()[0]
    n_runners = conn.execute("SELECT COUNT(*) FROM runners").fetchone()[0]

    run_id = conn.execute("SELECT nextval('quality_run_id_seq')").fetchone()[0]
    conn.execute(
        "INSERT INTO quality_runs VALUES (?, ?, ?, ?, ?, ?)",
        [run_id, started_at, scope_from, scope_to, n_races, n_runners],
    )

    report = QualityReport(run_id=run_id, n_races=n_races, n_runners=n_runners)

    for check_id, sql in FAIL_CHECKS.items():
        rows = conn.execute(sql).fetchall()
        report.fail_counts[check_id] = len(rows)
        for race_id, horse_id, detail in rows[:20]:
            conn.execute(
                "INSERT INTO quality_findings VALUES (?, ?, 'fail', ?, ?, ?)",
                [run_id, check_id, race_id, horse_id, detail],
            )

    for check_id, (num_sql, den_sql) in WARN_CHECKS.items():
        num = conn.execute(num_sql).fetchone()[0]
        den = conn.execute(den_sql).fetchone()[0]
        report.warn_rates[check_id] = (num, den)
        conn.execute(
            "INSERT INTO quality_findings VALUES (?, ?, 'warn', NULL, NULL, ?)",
            [run_id, check_id, f"{num}/{den}"],
        )

    return report
