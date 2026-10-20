"""Helpers for chunking: a rough token count and GitHub heading anchors."""

import math
import re


def estimate_tokens(text: str) -> int:
    """Rough token count: about 4 characters per token for English text.

    Good enough for deciding chunk sizes. It doesn't need to be exact, because
    chunks only have to stay well under the embedding model's input limit
    (8,191 tokens for text-embedding-3-small).
    """
    return math.ceil(len(text) / 4)


def _slugify(heading: str) -> str:
    text = heading.strip().lower()
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)  # [text](url) -> text
    text = text.replace("`", "")
    text = re.sub(r"[^\w\- ]", "", text)  # drop punctuation, keep letters/digits/_/-
    return text.replace(" ", "-")


class AnchorMaker:
    """Turns heading text into the #anchor GitHub generates for it.

    GitHub gives repeated headings in the same file a suffix: the first
    "Example" is #example, the second is #example-1, and so on. So create ONE
    AnchorMaker per file, and call it for EVERY heading in that file, in order,
    including headings whose section ends up empty.

        anchors = AnchorMaker()
        anchors("New order (TRADE)")        # -> "new-order-trade"
        anchors("-1021 INVALID_TIMESTAMP")  # -> "-1021-invalid_timestamp"
        anchors("Example")                  # -> "example"
        anchors("Example")                  # -> "example-1"
    """

    def __init__(self) -> None:
        self._seen: dict[str, int] = {}

    def __call__(self, heading: str) -> str:
        base = _slugify(heading)
        slug = base
        while slug in self._seen:
            self._seen[base] += 1
            slug = f"{base}-{self._seen[base]}"
        self._seen[slug] = 0
        return slug
