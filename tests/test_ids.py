"""`002-loader.md` 単体テスト観点11: エンティティ同定。"""

from __future__ import annotations

from tests.conftest import NOW
from umagic.ids import resolve


def test_11_resolve_is_idempotent(conn):
    id1 = resolve(conn, "horse", "netkeiba_jra", "2020103532", NOW)
    id2 = resolve(conn, "horse", "netkeiba_jra", "2020103532", NOW)
    assert id1 == id2
    rows = conn.execute("SELECT COUNT(*) FROM source_ids WHERE entity_type='horse'").fetchone()
    assert rows[0] == 1


def test_resolve_different_keys_get_different_ids(conn):
    id1 = resolve(conn, "horse", "netkeiba_jra", "A", NOW)
    id2 = resolve(conn, "horse", "netkeiba_jra", "B", NOW)
    assert id1 != id2


def test_resolve_scoped_by_entity_type(conn):
    """同じ source_key でも entity_type が違えば別々に解決される。"""
    hid = resolve(conn, "horse", "netkeiba_jra", "00001", NOW)
    jid = resolve(conn, "jockey", "netkeiba_jra", "00001", NOW)
    assert hid == jid == 1  # それぞれの entity_type 内で1番目
    rows = conn.execute("SELECT COUNT(*) FROM source_ids").fetchall()
    assert rows[0][0] == 2
