"""Shared adapter utilities."""

import re
import html as html_mod


def strip_html(s: str) -> str:
    """Remove HTML tags and unescape entities."""
    s = re.sub(r"<[^>]+>", "", s)
    return html_mod.unescape(s).strip()
