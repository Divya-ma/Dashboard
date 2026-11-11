"""The one clock for what a person reads: London time (the trading day is London's), and UTC for anything that must sort.

    london_now() / london_today()   the wall clock a trader looks at: labels, export stamps, file dates, "this year"
    utc_now()                       file names and records that are compared or ordered (London repeats an hour when the clocks go back)

A naive `datetime.now()` is whatever time zone this PC happens to be set to, so the same dashboard moved to a server in another zone (or
a Docker container, which is UTC) would show a different day around midnight. Stored trade and audit times are already UTC.
"""

from datetime import date, datetime, timezone
from zoneinfo import ZoneInfo

LONDON = ZoneInfo("Europe/London")


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def london_now() -> datetime:
    return datetime.now(LONDON)


def london_today() -> date:
    return london_now().date()
