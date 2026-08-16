#!/usr/bin/env python3
"""D-048 のバックフィル: `取` マーカーで棄却された行を再処理する。

`P-0` の3年分取り込みは `取`（出走取消）を未知マーカーとして棄却した
220行を含む。パーサーは `D-048` で対応済みだが、該当レースは
`fetch_log.outcome='ok'` のまま完了扱いなので、`scripts/ingest_range.py`
の再開ロジックでは触れられない。

キャッシュ済みページから該当レースだけを再パース・再書き込みする。
ネットワークへの新規アクセスは発生しない。
"""

from __future__ import annotations

import sys
from pathlib import Path

import duckdb

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from umagic.cache import LocalCacheFetcher
from umagic.loader import ingest_race
from umagic.sources.netkeiba import NetkeibaJraSource

UA = "UMAGIC-dev/0.1 (personal research; contact: repository owner)"


def main() -> int:
    db_path = Path("data/umagic.duckdb")
    conn = duckdb.connect(str(db_path))

    race_ids = [r[0] for r in conn.execute(
        "SELECT DISTINCT source_key FROM rejected_rows "
        "WHERE reason='unknown_finish_marker' AND raw='取'"
    ).fetchall()]
    print(f"対象レース: {len(race_ids)}件")

    fetcher = LocalCacheFetcher(cache_dir=Path("data/cache"), user_agent=UA)
    source = NetkeibaJraSource(fetcher)

    n_ok = n_still_missing = 0
    for rid in race_ids:
        out = ingest_race(conn, fetcher, source, rid)
        if out.outcome == "ok":
            n_ok += 1
        else:
            n_still_missing += 1
            print(f"  [{out.outcome}] {rid}: {out.detail}")

    print(f"再取り込み完了: ok={n_ok} 失敗={n_still_missing}")

    remaining = conn.execute(
        "SELECT COUNT(*) FROM rejected_rows WHERE reason='unknown_finish_marker' AND raw='取'"
    ).fetchone()[0]
    print(f"残存する `取` の未取り込み行: {remaining}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
