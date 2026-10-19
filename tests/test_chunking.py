"""Tests for chunk_markdown(). Each test matches a numbered rule in its docstring.

Run:  uv run pytest tests/test_chunking.py -v
"""

import pytest

from rag.chunking import Chunk, chunk_markdown
from rag.markdown_utils import estimate_tokens


# --- Rules 1-3: table of contents, headings, sections ------------------------


def test_simple_document():
    md = "# Title\n\nIntro text.\n\n## Section A\n\nText A.\n"
    assert chunk_markdown(md, "doc.md") == [
        Chunk("doc.md", "Title", "title", "Intro text.", 1),
        Chunk("doc.md", "Title > Section A", "section-a", "Text A.", 1),
    ]


def test_rule1_removes_table_of_contents():
    md = (
        "# Filters\n"
        "\n"
        "<!-- START doctoc generated TOC please keep comment here to allow auto update -->\n"
        "<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->\n"
        "- [Filters](#filters)\n"
        "  - [LOT_SIZE](#lot_size)\n"
        "<!-- END doctoc generated TOC please keep comment here to allow auto update -->\n"
        "\n"
        "Filters define trading rules.\n"
    )
    chunks = chunk_markdown(md, "filters.md")
    assert len(chunks) == 1
    assert chunks[0].content == "Filters define trading rules."


def test_rule2_hash_lines_inside_code_blocks_are_not_headings():
    md = (
        "# API\n"
        "\n"
        "## Signing\n"
        "\n"
        "Example:\n"
        "\n"
        "```bash\n"
        "# Sign the request\n"
        "signature=$(echo -n \"$params\" | openssl dgst -sha256)\n"
        "```\n"
        "\n"
        "After the code.\n"
    )
    chunks = chunk_markdown(md, "rest-api.md")
    assert [c.heading_path for c in chunks] == ["API > Signing"]
    assert "# Sign the request" in chunks[0].content
    assert chunks[0].content.endswith("After the code.")


def test_rule2_heading_needs_space_after_hashes():
    md = "# Title\n\n#hashtag is not a heading\n"
    chunks = chunk_markdown(md, "doc.md")
    assert len(chunks) == 1
    assert chunks[0].content == "#hashtag is not a heading"


def test_rule3_text_before_first_heading_is_ignored():
    md = '<a id="general-api-information"></a>\n\n# Public Rest API\n\nBody.\n'
    chunks = chunk_markdown(md, "rest-api.md")
    assert len(chunks) == 1
    assert chunks[0].content == "Body."


# --- Rule 4: heading paths ----------------------------------------------------


def test_rule4_heading_path_follows_levels():
    md = (
        "# Filters\n\nIntro.\n\n"
        "## Symbol filters\n\nAbout symbol filters.\n\n"
        "### PRICE_FILTER\n\nPrice rules.\n\n"
        "### LOT_SIZE\n\nQuantity rules.\n\n"
        "## Exchange Filters\n\nAbout exchange filters.\n"
    )
    assert [c.heading_path for c in chunk_markdown(md, "filters.md")] == [
        "Filters",
        "Filters > Symbol filters",
        "Filters > Symbol filters > PRICE_FILTER",
        "Filters > Symbol filters > LOT_SIZE",
        "Filters > Exchange Filters",
    ]


def test_rule4_skipping_levels_still_works():
    md = "# Doc\n\nIntro.\n\n### Deep\n\nDeep text.\n\n## Shallow\n\nShallow text.\n"
    assert [c.heading_path for c in chunk_markdown(md, "doc.md")] == [
        "Doc",
        "Doc > Deep",
        "Doc > Shallow",
    ]


def test_rule4_siblings_after_skipped_level():
    # Real docs do this: faqs/api_key_types.md goes from "#" straight to "###".
    md = "# Doc\n\nIntro.\n\n### A\n\nText A.\n\n### B\n\nText B.\n"
    assert [c.heading_path for c in chunk_markdown(md, "doc.md")] == [
        "Doc",
        "Doc > A",
        "Doc > B",
    ]


# --- Rule 5: content ------------------------------------------------------------


def test_rule5_content_is_stripped_and_empty_sections_skipped():
    md = "# Doc\n\n## Empty\n\n\n## Full\n\n\n   Some text.   \n\n\n"
    chunks = chunk_markdown(md, "doc.md")
    assert [c.heading_path for c in chunks] == ["Doc > Full"]
    assert chunks[0].content == "Some text."


def test_rule5_empty_parent_still_appears_in_child_path():
    md = "# Doc\n\n## Trading endpoints\n\n### New order (TRADE)\n\nPOST /api/v3/order\n"
    chunks = chunk_markdown(md, "rest-api.md")
    assert len(chunks) == 1
    assert chunks[0].heading_path == "Doc > Trading endpoints > New order (TRADE)"
    assert chunks[0].anchor == "new-order-trade"


# --- Rule 6: anchors ------------------------------------------------------------


def test_rule6_repeated_headings_get_numbered_anchors():
    md = "# Doc\n\n## Example\n\nA\n\n## Example\n\nB\n\n### Example\n\nC\n"
    chunks = chunk_markdown(md, "doc.md")
    assert [c.anchor for c in chunks] == ["example", "example-1", "example-2"]


def test_rule6_anchor_counter_includes_empty_sections():
    # The first "Example" has no content, so it produces no chunk, but GitHub
    # still gives it the anchor "example", so the second one is "example-1".
    md = "## Example\n\n### Example\n\nText.\n"
    chunks = chunk_markdown(md, "doc.md")
    assert len(chunks) == 1
    assert chunks[0].anchor == "example-1"


# --- Rule 7: long sections --------------------------------------------------------


def _paragraph(i: int) -> str:
    return f"Paragraph {i}: " + "word " * 30


def test_rule7_short_section_is_a_single_part():
    md = "# Doc\n\n" + _paragraph(1) + "\n"
    chunks = chunk_markdown(md, "doc.md", max_tokens=100)
    assert len(chunks) == 1
    assert chunks[0].part == 1


def test_rule7_long_section_is_packed_greedily_into_parts():
    paragraphs = [_paragraph(i).strip() for i in range(10)]
    md = "# Doc\n\n" + "\n\n".join(paragraphs) + "\n"
    max_tokens = 100
    chunks = chunk_markdown(md, "doc.md", max_tokens=max_tokens)

    assert len(chunks) > 1
    assert [c.part for c in chunks] == list(range(1, len(chunks) + 1))
    assert all(c.heading_path == "Doc" and c.anchor == "doc" for c in chunks)
    # Nothing lost or reordered: the parts join back into the original text.
    assert "\n\n".join(c.content for c in chunks) == "\n\n".join(paragraphs)
    for i, chunk in enumerate(chunks):
        # Each part fits...
        assert estimate_tokens(chunk.content) <= max_tokens
        # ...and is as full as it can be: adding the next paragraph would overflow.
        if i + 1 < len(chunks):
            next_block = chunks[i + 1].content.split("\n\n")[0]
            assert estimate_tokens(chunk.content + "\n\n" + next_block) > max_tokens


def test_rule7_code_block_with_blank_lines_is_never_split():
    code = "```python\n" + "\n\n".join(f"line_{i} = {i}" for i in range(40)) + "\n```"
    md = "# Doc\n\n" + _paragraph(1) + "\n\n" + code + "\n\n" + _paragraph(2) + "\n"
    chunks = chunk_markdown(md, "doc.md", max_tokens=60)
    with_code = [c for c in chunks if "```python" in c.content]
    assert len(with_code) == 1
    assert code in with_code[0].content


def test_rule7_oversized_block_becomes_its_own_part():
    small_1, small_2 = "Small one.", "Small two."
    big = "x" * 1000  # about 250 tokens, with no blank lines inside
    md = f"# Doc\n\n{small_1}\n\n{big}\n\n{small_2}\n"
    chunks = chunk_markdown(md, "doc.md", max_tokens=100)
    assert [c.content for c in chunks] == [small_1, big, small_2]
    assert [c.part for c in chunks] == [1, 2, 3]


# --- The real docs ------------------------------------------------------------------


def test_real_docs_smoke():
    """Runs on the actual Binance docs. Skipped until `uv run rag fetch-docs`."""
    from rag.config import load_settings
    from rag.sources import select_files

    docs_dir = load_settings().docs_dir
    if not docs_dir.exists():
        pytest.skip("docs not downloaded yet: run `uv run rag fetch-docs`")

    all_chunks = []
    for doc_path in select_files(docs_dir):
        text = (docs_dir / doc_path).read_text(encoding="utf-8")
        all_chunks.extend(chunk_markdown(text, doc_path))

    assert len(all_chunks) > 500
    assert all(c.content for c in all_chunks)
    assert not any("doctoc" in c.content for c in all_chunks)
    assert not any("Sign the request" in c.heading_path for c in all_chunks)
    # Must stay under the embedding model's input limit (8,191 tokens).
    assert max(estimate_tokens(c.embedding_text) for c in all_chunks) < 8191

    lot_size = [c for c in all_chunks if c.doc_path == "filters.md" and c.anchor == "lot_size"]
    assert len(lot_size) == 1
    assert lot_size[0].heading_path == "Filters > Symbol filters > LOT_SIZE"
    assert "stepSize" in lot_size[0].content
