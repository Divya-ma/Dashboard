"""Tests for core.user_settings validators."""

import pytest

from core.user_settings import (
    KEY_CORRELATION_WINDOW,
    KEY_MARGIN_LIMIT,
    KEY_PNL_STOP,
    KEY_ROLL_WARNING_DAYS,
    KEY_STALENESS,
    KEY_STRUCTURE_MAX_LOSS,
    KEY_VAR_CONFIDENCE,
    parse_number,
    validate_defaults,
    validate_thresholds,
    validate_token,
    validate_webhook_url,
)


# ---------- parse_number ----------


@pytest.mark.parametrize("bad", [None, "", "  ", True, "abc", float("nan"), float("inf"), [1]])
def test_parse_number_rejects_non_numbers(bad):
    with pytest.raises(ValueError):
        parse_number(bad, "Field")


def test_parse_number_accepts_numeric_strings_and_numbers():
    assert parse_number("12.5", "Field") == 12.5
    assert parse_number(3, "Field") == 3


def test_parse_number_bounds_and_integer():
    with pytest.raises(ValueError, match="at least"):
        parse_number(0, "Field", minimum=1)
    with pytest.raises(ValueError, match="at most"):
        parse_number(11, "Field", maximum=10)
    with pytest.raises(ValueError, match="whole number"):
        parse_number(2.5, "Field", integer=True)
    assert parse_number(5.0, "Field", integer=True) == 5
    assert isinstance(parse_number(5.0, "Field", integer=True), int)


# ---------- validate_token ----------


def test_token_is_trimmed_and_bearer_prefix_stripped():
    assert validate_token("  abc123  ") == "abc123"
    assert validate_token("Bearer abc123") == "abc123"
    assert validate_token("bearer   abc123") == "abc123"


@pytest.mark.parametrize("bad", [None, "", "   ", "Bearer ", 123])
def test_token_rejects_empty_or_non_string(bad):
    with pytest.raises(ValueError):
        validate_token(bad)


def test_token_rejects_embedded_whitespace_and_newlines():
    with pytest.raises(ValueError):
        validate_token("abc def")
    with pytest.raises(ValueError):
        validate_token("abc\r\nX-Injected: 1")


def test_token_error_message_never_contains_the_token():
    try:
        validate_token("secret value")
    except ValueError as exc:
        assert "secret" not in str(exc)


# ---------- validate_thresholds ----------


def valid_thresholds(**overrides):
    args = dict(
        pnl_stop=-50000, structure_max_loss=-10000, staleness_seconds=120,
        margin_limit=1_000_000, roll_warning_days=5,
    )
    args.update(overrides)
    return args


def test_thresholds_valid_returns_keyed_values():
    assert validate_thresholds(**valid_thresholds()) == {
        KEY_PNL_STOP: -50000.0,
        KEY_STRUCTURE_MAX_LOSS: -10000.0,
        KEY_STALENESS: 120.0,
        KEY_MARGIN_LIMIT: 1_000_000.0,
        KEY_ROLL_WARNING_DAYS: 5,
    }


def test_thresholds_positive_pnl_stop_rejected():
    with pytest.raises(ValueError, match="zero or negative"):
        validate_thresholds(**valid_thresholds(pnl_stop=50000))


def test_thresholds_positive_structure_max_loss_rejected():
    with pytest.raises(ValueError, match="zero or negative"):
        validate_thresholds(**valid_thresholds(structure_max_loss=10000))


@pytest.mark.parametrize(
    "field,value",
    [
        ("staleness_seconds", 0),
        ("staleness_seconds", 5000),
        ("margin_limit", -1),
        ("roll_warning_days", 2.5),
        ("roll_warning_days", -1),
        ("roll_warning_days", 91),
        ("pnl_stop", "lots"),
        ("margin_limit", None),
    ],
)
def test_thresholds_invalid_values_rejected(field, value):
    with pytest.raises(ValueError):
        validate_thresholds(**valid_thresholds(**{field: value}))


def test_thresholds_zero_stop_allowed():
    assert validate_thresholds(**valid_thresholds(pnl_stop=0))[KEY_PNL_STOP] == 0.0


# ---------- validate_defaults ----------


def test_defaults_valid():
    assert validate_defaults(0.99, 60) == {KEY_VAR_CONFIDENCE: 0.99, KEY_CORRELATION_WINDOW: 60}


@pytest.mark.parametrize("confidence", [0.9, 1.0, None, "high"])
def test_defaults_confidence_must_be_95_or_99(confidence):
    with pytest.raises(ValueError):
        validate_defaults(confidence, 60)


@pytest.mark.parametrize("window", [19, 501, 60.5, None, "x"])
def test_defaults_window_bounds(window):
    with pytest.raises(ValueError):
        validate_defaults(0.95, window)


# ---------- validate_webhook_url ----------


def test_webhook_empty_and_none_clear_the_setting():
    assert validate_webhook_url("") == ""
    assert validate_webhook_url("   ") == ""
    assert validate_webhook_url(None) == ""


def test_webhook_valid_https_is_trimmed():
    assert validate_webhook_url("  https://example.webhook.office.com/abc  ") == "https://example.webhook.office.com/abc"


@pytest.mark.parametrize(
    "bad",
    ["http://example.com/hook", "ftp://example.com", "example.com/hook", "https://", "https://a b.com/x", 42],
)
def test_webhook_rejects_non_https_or_malformed(bad):
    with pytest.raises(ValueError):
        validate_webhook_url(bad)
