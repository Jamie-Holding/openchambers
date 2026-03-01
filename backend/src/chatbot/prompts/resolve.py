"""Resolve node prompts."""

DATE_PARSE_PROMPT = """Convert this date expression to a date range with date_from and date_to in YYYY-MM-DD format.

Today's date: {today}

Date expression: "{date_text}"
"""
