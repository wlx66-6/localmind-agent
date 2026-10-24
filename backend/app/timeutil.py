"""Timestamp presentation helpers.

SQLite has no timezone type, so SQLAlchemy's ``DateTime`` gives back a naive
datetime whose value is UTC (``utcnow`` writes aware UTC; the SQLite dialect
drops the offset). Everything user-visible therefore goes through these
helpers instead of trusting the stored value — otherwise a memory recorded at
02:39 local time would be shown to the user, and to the model, as "yesterday".
"""

from __future__ import annotations

from datetime import UTC, date, datetime


def as_utc(value: datetime) -> datetime:
    """Attach UTC to a naive stored timestamp; convert an aware one."""
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def utc_iso(value: datetime) -> str:
    """Serialize as ISO 8601 with an explicit offset, for JSON and prompts."""
    return as_utc(value).isoformat()


def local_date(value: datetime | str) -> date:
    """The calendar date the user actually experienced."""
    parsed = datetime.fromisoformat(value) if isinstance(value, str) else value
    return as_utc(parsed).astimezone().date()
