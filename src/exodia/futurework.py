"""Mine the forward-looking sections of papers to estimate the next hot things.

Every paper's "Conclusion", "Future Work", "Discussion", "Limitations" and
"Open Questions" sections are where authors say, explicitly, what should come
next. This module locates those sections in the locally-cached full text
(:mod:`fulltext`) and returns just that text, so the curated concept gazetteer
(:mod:`concepts`) can be run over *only* the forward-looking prose — and trended
by year — to surface the directions the field keeps proposing.

The forward-looking sections live at the very end of a paper, so this depends on
``fulltext_max_chars`` being large enough to reach them (see :mod:`config`).
Detection is heading-anchored for precision: a section starts at a numbered
heading whose title contains a forward-looking keyword (``5 Conclusion and
Future Work``) or at a standalone heading line made only of heading words
(``Discussion and Conclusion``), and runs until the back-matter
(``References`` / ``Acknowledgements`` / ``Appendix``) or a per-section cap.

Analysis-only and offline, like the rest of the full-text pipeline: it reads the
git-ignored caches and never redistributes them.
"""

from __future__ import annotations

import re

from .config import Settings
from .fulltext import cached_pdf_text
from .models import Entry

# Forward-looking keywords (the things that mark a "what comes next" section).
_FWD = (
    r"(?:future\s+work|future\s+directions?|concluding\s+remarks|conclusions?|"
    r"limitations?|outlook|next\s+steps|open\s+(?:problems|questions|challenges))"
)
# Words a *combined* standalone heading line may be built from, e.g.
# "Discussion and Conclusion", "Broader impact, limitations, and future work".
_HEADING_WORD = (
    r"(?:discussion|summary|future|work|directions?|conclusions?|concluding|remarks|"
    r"limitations?|outlook|results?|broader\s+impact|next\s+steps|and|&|,)"
)

# A) Numbered section heading whose title (first ~55 chars) contains a fwd keyword:
#    "5 Conclusion and Future Work", "7.1 Open Questions ...".
_NUM_HEAD = re.compile(
    r"(?:\n|^)[ \t]*\d{1,2}(?:\.\d+)*\.?[ \t]+(?=[^\n]{0,55}?" + _FWD + r")",
    re.IGNORECASE,
)
# B) Standalone heading line made only of heading words/connectors, containing a
#    fwd keyword: "Conclusion", "Discussion and Conclusion".
_STD_HEAD = re.compile(
    r"(?:\n)[ \t]*"
    r"(?=(?:" + _HEADING_WORD + r")(?:[ \t]+(?:" + _HEADING_WORD + r"))*[ \t]*(?:\n|$))"
    r"(?=[^\n]*?" + _FWD + r")",
    re.IGNORECASE,
)
# Back-matter that ends a forward-looking section.
_STOP = re.compile(
    r"(?:\n|^)[ \t]*(?:\d{1,2}(?:\.\d+)*\.?[ \t]+)?"
    r"(?:references|acknowledge?ments?|bibliography|appendix|supplementary)\b",
    re.IGNORECASE,
)


def extract_future_sections(text: str, *, min_fraction: float = 0.4,
                            max_chars_per_section: int = 2400) -> str:
    """Return the concatenated forward-looking sections of a paper's full text.

    Only headings past ``min_fraction`` of the document are considered (these
    sections live near the end; the cutoff drops intro phrases like "limitations
    of prior work"). Each section runs until the next back-matter heading or
    ``max_chars_per_section``; overlapping/adjacent sections are merged.
    """
    if not text:
        return ""
    n = len(text)
    floor = min_fraction * n
    anchors = sorted(
        {m.start() for m in _NUM_HEAD.finditer(text)}
        | {m.start() for m in _STD_HEAD.finditer(text)}
    )
    spans: list[list[int]] = []
    for a in anchors:
        if a < floor:
            continue
        stop = _STOP.search(text, a + 5)
        end = min(stop.start() if stop else n, a + max_chars_per_section)
        spans.append([a, end])
    if not spans:
        return ""
    spans.sort()
    merged = [spans[0]]
    for s, e in spans[1:]:
        if s <= merged[-1][1] + 300:  # bridge tiny gaps between adjacent sections
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return " ".join(text[s:e].strip() for s, e in merged)


def future_work_text(entry: Entry, settings: Settings) -> str:
    """Forward-looking sections of an entry's cached full text (``""`` if none)."""
    if not settings.fulltext_analyze:
        return ""
    return extract_future_sections(cached_pdf_text(entry, settings))
