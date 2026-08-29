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
    # R-014: 払戻の整合。`枠連`/`枠単` は枠番（1〜8）の組み合わせで馬番では
    # ないため対象外（`枠単` は地方競馬（NAR）のみの券種、`Q-047` 段階②で
    # 実測して判明——`D-176`）
    "payout_horses": """
        SELECT p.race_id, NULL AS horse_id,
               p.bet_type || ' ' || p.comb_key || ' 馬番' || n AS detail
        FROM payouts p, UNNEST(p.combination) AS t(n)
        WHERE p.bet_type NOT IN ('枠連', '枠単')
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

# --- warn 系 -----------------------------------------------------------------
#
# 各検査は `(bucket, num, den)` を返すSQLを持つ。`by_year` は必須、`by_group`
# は次元がある検査のみ。全体の率は `by_year` を合算して求める。
#
# 年代別に出すのは `012-data-quality.md` の要求であり、`Q-020` が
# 「率だけでなく分布を見る必要がある」としているため。年代に偏りがあれば
# 単なる欠損ではなく取得ロジックの不具合を疑える。
#
# `rejected_rate` と `fetch_incomplete` の年は `source_key` の先頭4桁から取る。
# netkeiba の race_id は年で始まり、day_index の source_key は YYYYMMDD なので
# どちらも同じ規則で読める。**取り込みに失敗したレースは `races` に行が無い**
# ため、`races.date` から年を引くと失敗分が年代別集計から丸ごと消える。

WARN_CHECKS: dict[str, dict[str, str]] = {
    "corners_missing": {
        "by_year": """
            SELECT YEAR(r.date) AS bucket,
                   COUNT(*) FILTER (WHERE ru.corners IS NULL) AS num,
                   COUNT(*) AS den
            FROM runners ru JOIN races r USING (race_id)
            WHERE ru.status IN ('出走','降着')
              AND r.corner_nos IS NOT NULL AND len(r.corner_nos) > 0
            GROUP BY 1 ORDER BY 1
        """,
    },
    "rejected_rate": {
        "by_year": """
            WITH rej AS (
                SELECT TRY_CAST(SUBSTR(source_key, 1, 4) AS INTEGER) AS bucket,
                       COUNT(*) AS n
                FROM rejected_rows GROUP BY 1
            ), run AS (
                SELECT YEAR(r.date) AS bucket, COUNT(*) AS n
                FROM runners ru JOIN races r USING (race_id) GROUP BY 1
            )
            SELECT COALESCE(rej.bucket, run.bucket) AS bucket,
                   COALESCE(rej.n, 0) AS num,
                   COALESCE(rej.n, 0) + COALESCE(run.n, 0) AS den
            FROM rej FULL OUTER JOIN run ON rej.bucket = run.bucket
            ORDER BY 1
        """,
        # 未知の着順マーカーは `raw` まで出す。`Q-023`（出走取消・失格の表記が
        # 未観測）はここに実物が溜まることで閉じられる。他の理由は `raw` が
        # 通過順の文字列などで値が散らばるため理由だけにまとめる
        "by_group": """
            SELECT reason || CASE WHEN reason = 'unknown_finish_marker'
                                  THEN ' (' || raw || ')' ELSE '' END AS bucket,
                   COUNT(*) AS num, COUNT(*) AS den
            FROM rejected_rows GROUP BY 1 ORDER BY 2 DESC
        """,
    },
    # `day_index` の `empty` は「その日にJRA中央開催が無い」であり平日は
    # 大半がこれ。異常ではないので分子から外す。外さないと、開催の無い日が
    # 率を支配して実際の取得漏れが見えなくなる。
    "fetch_incomplete": {
        "by_year": """
            SELECT TRY_CAST(SUBSTR(source_key, 1, 4) AS INTEGER) AS bucket,
                   COUNT(*) FILTER (
                       WHERE outcome <> 'ok'
                         AND NOT (page_kind = 'day_index' AND outcome = 'empty')
                   ) AS num,
                   COUNT(*) AS den
            FROM fetch_log GROUP BY 1 ORDER BY 1
        """,
        "by_group": """
            SELECT page_kind || '/' || outcome AS bucket,
                   COUNT(*) AS num, COUNT(*) AS den
            FROM fetch_log GROUP BY 1 ORDER BY 2 DESC
        """,
    },
    "odds_coverage": {
        "by_year": """
            SELECT YEAR(r.date) AS bucket,
                   COUNT(DISTINCT o.race_id) AS num,
                   COUNT(DISTINCT r.race_id) AS den
            FROM races r LEFT JOIN odds o USING (race_id)
            GROUP BY 1 ORDER BY 1
        """,
        # Q-018: 単勝以外はほぼ0%になる見込み。券種別に出して実測する
        "by_group": """
            SELECT o.bet_type AS bucket,
                   COUNT(DISTINCT o.race_id) AS num,
                   (SELECT COUNT(*) FROM races) AS den
            FROM odds o GROUP BY 1 ORDER BY 2 DESC
        """,
    },
    "laps_coverage": {
        "by_year": """
            SELECT YEAR(r.date) AS bucket,
                   COUNT(*) FILTER (
                       WHERE NOT EXISTS (SELECT 1 FROM laps l WHERE l.race_id = r.race_id)
                   ) AS num,
                   COUNT(*) AS den
            FROM races r GROUP BY 1 ORDER BY 1
        """,
    },
}


@dataclass
class WarnResult:
    """1つの `warn` 検査の結果。全体の率に加え、年代別と（あれば）次元別を持つ。"""

    check_id: str
    by_year: list[tuple[object, int, int]] = field(default_factory=list)
    by_group: list[tuple[object, int, int]] = field(default_factory=list)

    @property
    def num(self) -> int:
        return sum(n for _, n, _ in self.by_year)

    @property
    def den(self) -> int:
        return sum(d for _, _, d in self.by_year)

    @property
    def rate(self) -> float | None:
        return (self.num / self.den) if self.den else None


@dataclass
class QualityReport:
    run_id: int
    fail_counts: dict[str, int] = field(default_factory=dict)
    warns: dict[str, WarnResult] = field(default_factory=dict)
    n_races: int = 0
    n_runners: int = 0

    @property
    def n_fail(self) -> int:
        return sum(self.fail_counts.values())

    @property
    def exit_code(self) -> int:
        return 0 if self.n_fail == 0 else 1

    def to_markdown(self) -> str:
        def pct(num: int, den: int) -> str:
            return f"{100 * num / den:.1f}%" if den else "n/a"

        lines = [f"# 品質レポート（run_id={self.run_id}）", ""]
        lines.append(f"対象: races={self.n_races} runners={self.n_runners}")
        lines.append("")
        lines.append("## fail")
        lines.append("")
        lines.append("| check_id | 違反件数 |")
        lines.append("|---|---|")
        for check_id, n in self.fail_counts.items():
            lines.append(f"| `{check_id}` | {n} |")
        lines.append("")
        lines.append(f"**fail 合計: {self.n_fail}件**"
                     + ("（`P-0` 完了条件を満たす）" if self.n_fail == 0 else "（`D-040` により `P-0` 未完了）"))

        lines.append("")
        lines.append("## warn")
        lines.append("")
        lines.append("閾値を設けない（`D-040`）。件数と年代別分布を出す。")
        for check_id, w in self.warns.items():
            lines.append("")
            lines.append(f"### `{check_id}`  {w.num}/{w.den} ({pct(w.num, w.den)})")
            if w.by_year:
                lines.append("")
                lines.append("| 年 | 分子 | 分母 | 率 |")
                lines.append("|---|---|---|---|")
                for bucket, num, den in w.by_year:
                    label = bucket if bucket is not None else "不明"
                    lines.append(f"| {label} | {num} | {den} | {pct(num, den)} |")
            if w.by_group:
                lines.append("")
                lines.append("| 内訳 | 件数 |")
                lines.append("|---|---|")
                for bucket, num, _den in w.by_group:
                    lines.append(f"| {bucket} | {num} |")
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

    for check_id, sqls in WARN_CHECKS.items():
        w = WarnResult(check_id=check_id)
        w.by_year = [tuple(r) for r in conn.execute(sqls["by_year"]).fetchall()]
        if "by_group" in sqls:
            w.by_group = [tuple(r) for r in conn.execute(sqls["by_group"]).fetchall()]
        report.warns[check_id] = w

        conn.execute(
            "INSERT INTO quality_findings VALUES (?, ?, 'warn', NULL, NULL, ?)",
            [run_id, check_id, f"{w.num}/{w.den}"],
        )
        # 年代別も残す。レポートは実行のたびに上書きされるが、
        # quality_findings は run_id で追える（D-041 の「印」）
        for bucket, num, den in w.by_year:
            conn.execute(
                "INSERT INTO quality_findings VALUES (?, ?, 'warn', NULL, NULL, ?)",
                [run_id, f"{check_id}:year={bucket}", f"{num}/{den}"],
            )

    return report
