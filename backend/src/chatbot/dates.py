"""Date text parsing: regex patterns with LLM fallback."""

import calendar
import re
from datetime import date, timedelta

from backend.src.chatbot.messages.resolve import DATE_NOT_UNDERSTOOD
from backend.src.chatbot.prompts.resolve import DATE_PARSE_PROMPT
from backend.src.chatbot.schemas import DateRange
from backend.src.chatbot.utils import llm

MONTHS = {m.lower(): i for i, m in enumerate(calendar.month_name) if m}


def parse_dates_regex(date_text: str) -> tuple[str, str] | None:
    """Try to parse date_text with regex patterns. Returns (date_from, date_to) or None."""
    today = date.today()
    text = date_text.strip().lower()

    # "in 2025" or just "2025"
    m = re.match(r"(?:in\s+)?(\d{4})$", text)
    if m:
        year = int(m.group(1))
        return f"{year}-01-01", f"{year}-12-31"

    # "last N months"
    m = re.match(r"last\s+(\d+)\s+months?$", text)
    if m:
        months = int(m.group(1))
        start = today - timedelta(days=months * 30)
        return start.strftime("%Y-%m-%d"), today.strftime("%Y-%m-%d")

    # "last N years"
    m = re.match(r"last\s+(\d+)\s+years?$", text)
    if m:
        years = int(m.group(1))
        start = today.replace(year=today.year - years)
        return start.strftime("%Y-%m-%d"), today.strftime("%Y-%m-%d")

    # "since <month> [year]"
    m = re.match(r"since\s+(\w+)(?:\s+(\d{4}))?$", text)
    if m:
        month_name = m.group(1).lower()
        if month_name in MONTHS:
            month_num = MONTHS[month_name]
            year = int(m.group(2)) if m.group(2) else today.year
            return f"{year}-{month_num:02d}-01", today.strftime("%Y-%m-%d")

    # "in <month> <year>"
    m = re.match(r"in\s+(\w+)\s+(\d{4})$", text)
    if m:
        month_name = m.group(1).lower()
        if month_name in MONTHS:
            month_num = MONTHS[month_name]
            year = int(m.group(2))
            last_day = calendar.monthrange(year, month_num)[1]
            return f"{year}-{month_num:02d}-01", f"{year}-{month_num:02d}-{last_day:02d}"

    return None


async def parse_dates_llm(date_text: str) -> tuple[str, str] | None:
    """LLM fallback for date expressions regex can't handle."""
    today = date.today().strftime("%Y-%m-%d")
    prompt = DATE_PARSE_PROMPT.format(today=today, date_text=date_text)
    try:
        parser = llm.with_structured_output(DateRange)
        result = await parser.ainvoke(prompt)
        return result.date_from, result.date_to
    except Exception:
        return None


async def parse_dates(date_text: str | None) -> tuple[str | None, str | None, str | None]:
    """Parse date_text into (date_from, date_to, ask_message).

    Returns ask_message if both tiers fail.
    """
    if not date_text:
        return None, None, None

    result = parse_dates_regex(date_text)
    if result:
        return result[0], result[1], None

    result = await parse_dates_llm(date_text)
    if result:
        return result[0], result[1], None

    return None, None, DATE_NOT_UNDERSTOOD.format(date_text=date_text)
