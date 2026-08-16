from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import pytest

sys.path.insert(0, str(Path(__file__).parent))

from umagic.ops_schema import create_ops_schema
from umagic.schema import create_schema


@pytest.fixture()
def conn():
    c = duckdb.connect()
    create_schema(c)
    create_ops_schema(c)
    yield c
    c.close()


NOW = datetime(2026, 8, 16, tzinfo=timezone.utc)
