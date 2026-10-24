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
    chunks = chunk_text(text, max_chars=100, overlap=20, markdown=False)

    assert len(chunks) > 1
    assert all(len(chunk.text) <= 100 for chunk in chunks)


def test_long_paragraph_is_hard_split_with_overlap() -> None:
    text = "甲" * 1500
    chunks = chunk_text(text, max_chars=600, overlap=100, markdown=False)

    assert len(chunks) == 3
    assert chunks[0].text == "甲" * 600
    assert chunks[1].text == "甲" * 600
    assert chunks[2].text == "甲" * 500
    assert all(chunk.start_line == 1 for chunk in chunks)


def test_plain_text_has_no_heading() -> None:
    chunks = chunk_text("这是一段普通的纯文本内容，没有标题。", markdown=False)
    assert len(chunks) == 1
    assert chunks[0].heading == ""


def test_short_fragments_are_dropped() -> None:
    assert chunk_text("hi", markdown=False) == []


def test_crlf_is_normalized() -> None:
    text = "# 标题\r\n这是第二行的正文内容。\r\n"
    chunks = chunk_text(text)
    assert len(chunks) == 1
    assert chunks[0].start_line == 2
    assert "\r" not in chunks[0].text


def test_embedding_text_includes_heading() -> None:
    chunks = chunk_text("# 标题\n正文内容足够长，可以形成分块。\n")
    assert chunks[0].embedding_text.startswith("标题\n")
