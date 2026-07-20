from __future__ import annotations

import re
from datetime import datetime, timedelta

DISPLAY_TIME_FORMAT = "%Y-%m-%d %H:%M:%S"
DISPLAY_TIMESTAMP_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$")


def display_datetime_label(moment: datetime | None = None) -> str:
    return (moment or datetime.now()).strftime(DISPLAY_TIME_FORMAT)


def normalize_display_timestamp(value: str, reference: datetime | None = None) -> str:
    if DISPLAY_TIMESTAMP_PATTERN.match(value):
        return value

    now = reference or datetime.now()
    clean = value.strip()

    time_match = re.fullmatch(r"(\d{1,2}):(\d{2})(?::(\d{2}))?", clean)
    if time_match:
        hour, minute, second = _time_parts(time_match)
        return now.replace(hour=hour, minute=minute, second=second, microsecond=0).strftime(
            DISPLAY_TIME_FORMAT
        )

    relative_match = re.fullmatch(r"(今天|昨天)\s*(?:(\d{1,2}):(\d{2})(?::(\d{2}))?)?", clean)
    if relative_match:
        day = now if relative_match.group(1) == "今天" else now - timedelta(days=1)
        hour = int(relative_match.group(2) or 0)
        minute = int(relative_match.group(3) or 0)
        second = int(relative_match.group(4) or 0)
        return day.replace(hour=hour, minute=minute, second=second, microsecond=0).strftime(
            DISPLAY_TIME_FORMAT
        )

    weekday_offsets = {
        "周一": 0,
        "周二": 1,
        "周三": 2,
        "周四": 3,
        "周五": 4,
        "周六": 5,
        "周日": 6,
        "周天": 6,
    }
    if clean in weekday_offsets:
        start_of_week = now - timedelta(days=now.weekday())
        target = start_of_week + timedelta(days=weekday_offsets[clean])
        return target.replace(hour=0, minute=0, second=0, microsecond=0).strftime(
            DISPLAY_TIME_FORMAT
        )

    month_day_match = re.fullmatch(r"(\d{1,2})/(\d{1,2})", clean)
    if month_day_match:
        month = int(month_day_match.group(1))
        day = int(month_day_match.group(2))
        return now.replace(
            month=month, day=day, hour=0, minute=0, second=0, microsecond=0
        ).strftime(DISPLAY_TIME_FORMAT)

    return clean


def _time_parts(match: re.Match[str]) -> tuple[int, int, int]:
    return int(match.group(1)), int(match.group(2)), int(match.group(3) or 0)
