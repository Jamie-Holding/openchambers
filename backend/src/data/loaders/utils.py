"""Utility functions for data loaders."""


def extract_person_id(s: str | None) -> int | None:
    """Extract numeric person ID from a TheyWorkForYou person ID string.

    Args:
        s: Person ID string (e.g. 'uk.org.publicwhip/person/10001').

    Returns:
        Numeric ID extracted from the string, or None if input is None.
    """
    if s is None:
        return None
    return int(s.split("/")[-1])