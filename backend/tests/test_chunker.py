from __future__ import annotations

from app.ingestion.chunker import chunk_text


def test_markdown_sections_carry_heading_and_lines() -> None:
    text = (
        "# 标题一\n"
        "第一段内容足够长，可以成为一个分块。\n"
        "\n"
        "## 小节\n"
        "第二段内容也足够长，可以成为一个分块。\n"
    )
    chunks = chunk_text(text, max_chars=600)

    assert [chunk.heading for chunk in chunks] == ["标题一", "标题一 > 小节"]
    assert (chunks[0].start_line, chunks[0].end_line) == (2, 2)
    assert (chunks[1].start_line, chunks[1].end_line) == (5, 5)
    assert [chunk.ordinal for chunk in chunks] == [0, 1]


def test_paragraphs_are_packed_up_to_limit() -> None:
    text = "。".join("句子" * 10 for _ in range(20)) + "。"
    chunks = chunk_text(text, max_chars=100, overlap=20, mode="text")

    assert len(chunks) > 1
    assert all(len(chunk.text) <= 100 for chunk in chunks)


def test_long_paragraph_is_hard_split_with_overlap() -> None:
    text = "甲" * 1500
    chunks = chunk_text(text, max_chars=600, overlap=100, mode="text")

    assert len(chunks) == 3
    assert chunks[0].text == "甲" * 600
    assert chunks[1].text == "甲" * 600
    assert chunks[2].text == "甲" * 500
    assert all(chunk.start_line == 1 for chunk in chunks)


def test_plain_text_has_no_heading() -> None:
    chunks = chunk_text("这是一段普通的纯文本内容，没有标题。", mode="text")
    assert len(chunks) == 1
    assert chunks[0].heading == ""


def test_short_fragments_are_dropped() -> None:
    assert chunk_text("hi", mode="text") == []


def test_crlf_is_normalized() -> None:
    text = "# 标题\r\n这是第二行的正文内容。\r\n"
    chunks = chunk_text(text)
    assert len(chunks) == 1
    assert chunks[0].start_line == 2
    assert "\r" not in chunks[0].text


def test_embedding_text_includes_heading() -> None:
    chunks = chunk_text("# 标题\n正文内容足够长，可以形成分块。\n")
    assert chunks[0].embedding_text.startswith("标题\n")


def test_code_sections_start_at_top_level_symbols() -> None:
    text = (
        "import os\n"
        "from pathlib import Path\n"
        "\n"
        "def parse(path):\n"
        "    return path\n"
        "\n"
        "class Store:\n"
        "    def get(self):\n"
        "        return None\n"
    )
    chunks = chunk_text(text, mode="code")

    assert [chunk.heading for chunk in chunks] == ["", "def parse(path):", "class Store:"]
    # A preamble before the first symbol becomes its own section.
    assert chunks[0].text.startswith("import os")
    # The declaration line stays inside the body: it is part of the code.
    assert chunks[1].text.startswith("def parse(path):")
    # A nested method must not start a section of its own.
    assert chunks[2].text.count("def get") == 1
    assert chunks[2].start_line == 7


def test_code_heading_is_repeated_when_a_function_is_split() -> None:
    body = "\n".join(f"    value_{index} = {index}" for index in range(120))
    chunks = chunk_text(f"def big():\n{body}\n", max_chars=200, overlap=20, mode="code")

    assert len(chunks) > 1
    assert all(chunk.heading == "def big():" for chunk in chunks)


def test_page_starts_map_chunks_to_pages() -> None:
    page_one = "第一页的内容，讲的是检索。" * 3
    page_two = "第二页的内容，讲的是重排。" * 3
    text = f"{page_one}\n\n{page_two}"
    chunks = chunk_text(
        text, max_chars=40, overlap=0, mode="text", page_starts=(0, len(page_one) + 2)
    )

    assert len(chunks) == 2
    assert [(chunk.page, chunk.page_end) for chunk in chunks] == [(1, 1), (2, 2)]
    assert all(chunk.heading == "" for chunk in chunks)


def test_chunks_never_cross_a_page_break() -> None:
    # Short enough to fit one chunk, but the page boundary still splits it:
    # "第 1–2 页" would not tell the reader where the sentence is.
    text = "甲" * 40 + "\n\n" + "乙" * 40
    chunks = chunk_text(text, max_chars=200, overlap=0, mode="text", page_starts=(0, 42))

    assert len(chunks) == 2
    assert [(chunk.page, chunk.page_end) for chunk in chunks] == [(1, 1), (2, 2)]


def test_long_page_is_split_within_its_own_page() -> None:
    page_one = "甲" * 200
    page_two = "乙" * 200
    text = f"{page_one}\n\n{page_two}"
    chunks = chunk_text(text, max_chars=100, overlap=0, mode="text", page_starts=(0, 202))

    assert [chunk.page for chunk in chunks] == [1, 1, 2, 2]
    assert all(chunk.page == chunk.page_end for chunk in chunks)


def test_documents_without_page_starts_carry_no_page() -> None:
    chunks = chunk_text("# 标题\n正文内容足够长，可以形成分块。\n")
    assert chunks[0].page is None and chunks[0].page_end is None
