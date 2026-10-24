from __future__ import annotations

from datetime import UTC, date, datetime

from app.timeutil import as_utc, local_date, utc_iso


def test_utc_iso_marks_a_naive_stored_timestamp_as_utc() -> None:
    # SQLite hands back naive UTC; without the offset the browser would read it
    # as local time and could show the previous day.
    assert utc_iso(datetime(2026, 10, 24, 18, 39, 22)) == "2026-10-24T18:39:22+00:00"


def test_utc_iso_keeps_the_instant_of_an_aware_timestamp() -> None:
    aware = datetime(2026, 10, 25, 2, 39, 22, tzinfo=UTC)
    assert utc_iso(aware) == "2026-10-25T02:39:22+00:00"


def test_local_date_reads_iso_strings_and_naive_datetimes_identically() -> None:
    assert local_date("2026-10-24T18:39:22+00:00") == local_date(datetime(2026, 10, 24, 18, 39, 22))


def test_local_date_is_a_date() -> None:
    assert isinstance(local_date("2026-10-24T18:39:22+00:00"), date)


def test_as_utc_keeps_an_aware_value_untouched() -> None:
    aware = datetime(2026, 10, 25, 2, 39, tzinfo=UTC)
    assert as_utc(aware) is aware
