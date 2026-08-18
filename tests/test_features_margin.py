"""着差の数値化（`D-064`）。"""

from __future__ import annotations

import pytest

from umagic.features.margin import parse_margin


@pytest.mark.parametrize(
    "text, expected",
    [
        ("1/2", 0.5),
        ("3/4", 0.75),
        ("1.1/4", 1.25),
        ("1.1/2", 1.5),
        ("1.3/4", 1.75),
        ("3.1/2", 3.5),
        ("1", 1.0),
        ("2", 2.0),
        ("9", 9.0),
        ("ハナ", 0.05),
        ("アタマ", 0.1),
        ("クビ", 0.2),
        ("大", 10.0),
        ("同着", 0.0),
    ],
)
def test_known_notations(text, expected):
    assert parse_margin(text) == expected


@pytest.mark.parametrize("text", [None, "", "  ", "不明", "1/0"])
def test_unknown_or_missing_returns_none(text):
    assert parse_margin(text) is None


def test_strips_whitespace():
    assert parse_margin(" クビ ") == 0.2
