"""Split a Markdown document into chunks for search.

>>> YOUR TASK: implement chunk_markdown() below. <<<

Run the tests with:   uv run pytest tests/test_chunking.py -v
Preview real chunks:  uv run rag chunks --file filters.md

You can use these helpers (already written, in markdown_utils.py):
  - estimate_tokens(text) -> int     rough token count (characters / 4)
  - AnchorMaker()                    turns heading text into GitHub #anchors
"""

from dataclasses import dataclass

from rag.markdown_utils import AnchorMaker, estimate_tokens  # noqa: F401  (for you to use)

import re


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
    for line in text.splitlines():
        m = re.match(r"(#{1,6}) (.*)", line)
        if m:
            content = "\n".join(current_lines).strip()
            if content:
                chunck = Chunk(doc_path, current_heading_path, anchor, content)
                chunks.append(chunck)
            current_title = m.group(2)
            current_level = len(m.group(1))
            current_lines = []
            anchor = anchors(current_title)
            current_heading_path = generate_heading_paths(levels, current_level, current_title)
            
        elif current_title is not None:
            current_lines.append(line)

    content = "\n".join(current_lines).strip()
    if content:
        chunks.append(Chunk(doc_path, current_heading_path, anchor, content))

    return chunks

# a helper function that creates the correct heading path for each chunk
def generate_heading_paths(levels, current_level, current_title) -> str:
    if current_level <= len(levels):
        del levels[current_level - 1:]
    levels.append(current_title)

    return ' > '.join(levels)
