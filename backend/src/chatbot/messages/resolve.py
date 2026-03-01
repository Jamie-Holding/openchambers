"""User-facing message templates for the resolve node."""

PERSON_NOT_FOUND = (
    'I couldn\'t find an MP named "{name}". Could you check the spelling?'
)

PERSON_DISAMBIGUATION = (
    'I found multiple MPs named "{name}". Which one do you mean?\n'
    "\n{options}\n"
    "\nPlease reply with the number."
)

DATE_NOT_UNDERSTOOD = (
    'I couldn\'t understand the date "{date_text}". '
    "Could you rephrase it? (e.g. 'in 2025' or 'last 6 months')"
)


def format_person_options(matches: list[dict]) -> str:
    """Format a numbered list of person matches for disambiguation."""
    lines = []
    for i, m in enumerate(matches, 1):
        party = m["current_party"] or "unknown party"
        lines.append(f"{i}) {m['display_name']} — {party}")
    return "\n".join(lines)
