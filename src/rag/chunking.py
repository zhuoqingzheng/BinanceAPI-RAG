"""Split a Markdown document into chunks for search.

>>> YOUR TASK: implement chunk_markdown() below. <<<

Run the tests with:   uv run pytest tests/test_chunking.py -v
Preview real chunks:  uv run rag chunks --file filters.md

You can use these helpers (already written, in markdown_utils.py):
  - estimate_tokens(text) -> int     rough token count (characters / 4)
  - AnchorMaker()                    turns heading text into GitHub #anchors
"""

import re
from dataclasses import dataclass

from rag.markdown_utils import AnchorMaker, estimate_tokens


@dataclass(frozen=True)
class Chunk:
    doc_path: str  # file the chunk came from, e.g. "filters.md"
    heading_path: str  # e.g. "Filters > Symbol filters > LOT_SIZE"
    anchor: str  # GitHub anchor of the section's heading, e.g. "lot_size"
    content: str  # the section's text, without the heading line itself
    part: int = 1  # 1, 2, 3... when a long section is split into several chunks

    @property
    def embedding_text(self) -> str:
        """What gets embedded and searched: where the chunk is, then its text.

        Including the file and heading path gives every chunk its context, so
        part 3 of a long section still "knows" which endpoint it belongs to.
        """
        return f"{self.doc_path} > {self.heading_path}\n\n{self.content}"


def chunk_markdown(text: str, doc_path: str, max_tokens: int = 800) -> list[Chunk]:
    """Split one Markdown file into a list of Chunks.

    Rules (each one has a test in tests/test_chunking.py):

    1. Table of contents. Remove the auto-generated table of contents: every
       line from the one containing "<!-- START doctoc" to the one containing
       "<!-- END doctoc", inclusive. (In these docs the TOC happens to sit
       before the first heading, so rule 3 would drop it anyway. But many
       Markdown files put it right after the title, so handle it explicitly.)

    2. Headings. A heading is a line that starts with 1 to 6 "#" characters
       followed by a space, e.g. "## Trading endpoints". But lines inside fenced
       code blocks are never headings: the docs contain shell code with comment
       lines like "# Sign the request". A line whose stripped text starts with
       ``` opens a code block, and the next such line closes it.

    3. Sections. Each heading starts a new section, which runs until the next
       heading of any level. Text before the first heading is ignored.

    4. Heading path. A chunk's heading_path is the titles of its heading and
       all its parent headings, from the top down, joined by " > ".
       A heading of level N closes any open headings of level N or deeper:

           # Filters                  -> "Filters"
           ## Symbol filters          -> "Filters > Symbol filters"
           ### PRICE_FILTER           -> "Filters > Symbol filters > PRICE_FILTER"
           ### LOT_SIZE               -> "Filters > Symbol filters > LOT_SIZE"
           ## Exchange Filters        -> "Filters > Exchange Filters"

    5. Content. Strip whitespace from the start and end of each section's
       content. A section with no content (a heading directly followed by
       another heading) produces no chunk, but its title still appears in its
       children's heading paths.

    6. Anchors. Create one AnchorMaker for the whole file and call it once for
       every heading, in order, including headings of empty sections.
       Otherwise repeated headings get the wrong "-1", "-2" suffixes.

    7. Long sections. If estimate_tokens(content) > max_tokens, split the
       content into parts:
         - Split only at blank lines that are outside code blocks, so every
           paragraph, list, table and code block stays whole. Call these
           pieces "blocks".
         - Fill each part with consecutive blocks for as long as the part stays
           within max_tokens (blocks joined with a blank line, "\\n\\n").
         - A single block bigger than max_tokens becomes a part on its own.
         - Number the parts 1, 2, 3...; they share the heading_path and anchor.
    """
    current_title = None
    current_lines = []
    chunks = []
    anchors = AnchorMaker()
    levels = []
    current_heading_path = None
    anchor = None
    in_code = False
    text = remove_table_contents(text)
    for line in text.splitlines():
        # Rule 2: a ``` line opens or closes a code block. Inside one, a line
        # starting with "#" is a code comment, not a heading.
        if line.strip().startswith("```"):
            in_code = not in_code
        m = None if in_code else re.match(r"(#{1,6}) (.*)", line)
        if m:
            chunks.extend(make_chunks(doc_path, current_heading_path, anchor, current_lines, max_tokens))
            current_title = m.group(2)
            current_level = len(m.group(1))
            current_lines = []
            anchor = anchors(current_title)
            current_heading_path = generate_heading_paths(levels, current_level, current_title)
        elif current_title is not None:
            current_lines.append(line)

    chunks.extend(make_chunks(doc_path, current_heading_path, anchor, current_lines, max_tokens))
    return chunks


# a helper function that creates the correct heading path for each chunk
def generate_heading_paths(levels, current_level, current_title) -> str:
    while levels and levels[-1][0] >= current_level:
        levels.pop()
    levels.append((current_level, current_title))

    return ' > '.join(title for _, title in levels)


# a helper function that removes the auto-generated table of contents (rule 1)
def remove_table_contents(text: str) -> str:
    kept = []
    in_toc = False
    for line in text.splitlines():
        if "<!-- START doctoc" in line:
            in_toc = True
        if not in_toc:
            kept.append(line)
        if "<!-- END doctoc" in line:
            in_toc = False
    return "\n".join(kept)


def make_chunks(doc_path, heading_path, anchor, lines, max_tokens) -> list[Chunk]:
    """Turn one section's lines into chunks: none if empty, several if long."""
    content = "\n".join(lines).strip()
    if not content:
        return []
    if estimate_tokens(content) <= max_tokens:
        return [Chunk(doc_path, heading_path, anchor, content)]
    parts = pack_blocks(split_into_blocks(content), max_tokens)
    return [
        Chunk(doc_path, heading_path, anchor, part_text, part_number)
        for part_number, part_text in enumerate(parts, start=1)
    ]


def split_into_blocks(content: str) -> list[str]:
    """Split at blank lines outside code blocks, so code blocks stay whole."""
    blocks = []
    current = []
    in_code = False
    for line in content.splitlines():
        if line.strip().startswith("```"):
            in_code = not in_code
        if line.strip() == "" and not in_code:
            if current:
                blocks.append("\n".join(current))
                current = []
        else:
            current.append(line)
    if current:
        blocks.append("\n".join(current))
    return blocks


def pack_blocks(blocks: list[str], max_tokens: int) -> list[str]:
    """Fill each part with consecutive blocks while it stays within max_tokens.

    A block bigger than max_tokens on its own becomes a part by itself.
    """
    parts = []
    current = []
    for block in blocks:
        if current and estimate_tokens("\n\n".join(current + [block])) > max_tokens:
            parts.append("\n\n".join(current))
            current = []
        current.append(block)
    if current:
        parts.append("\n\n".join(current))
    return parts