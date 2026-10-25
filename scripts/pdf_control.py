"""Stage 6 control experiment: the same notes as markdown and as PDF.

The question is not "is PDF good" but "what does the PDF path cost": the same
content, the same questions, the same chunker — only the format and therefore
the chunk boundaries differ.

    .venv/Scripts/python.exe scripts/pdf_control.py build
    .venv/Scripts/python.exe scripts/pdf_control.py compare

``build``
    Converts examples/corpus/*.md into examples/pdf-corpus/*.pdf with page
    breaks that land mid-paragraph (as they do in real documents), then resolves
    every gold quote against both corpora and writes one question file per
    format. A quote that is not contained in exactly one chunk of *both*
    formats fails the build — that is the whole point of using verbatim quotes.

``compare``
    Reads the two eval reports produced from those question files and writes
    docs/eval/pdf-control-report.md.

Nothing here needs a network: PDFs are rendered locally with PyMuPDF.
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR / "backend"))

import pymupdf  # noqa: E402

from app.config import Settings, get_settings  # noqa: E402
from app.ingestion.chunker import ChunkData, chunk_text  # noqa: E402
from app.ingestion.parser import parse_document  # noqa: E402

CORPUS = BASE_DIR / "examples" / "corpus"
PDF_CORPUS = BASE_DIR / "examples" / "pdf-corpus"
QUESTIONS_SRC = BASE_DIR / "eval" / "stage6" / "questions.src.jsonl"
PAGE_WIDTH, PAGE_HEIGHT = 595, 842  # A4 points
MARGIN = 56
FONT_SIZE = 12.0
LINE_HEIGHT = 21.0
FONT_NAME = "china-s"

LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]+\)")
INLINE_RE = re.compile(r"[`*]{1,3}")
LIST_RE = re.compile(r"^\s*(?:[-*+]|\d+\.)\s+")


# --------------------------------------------------------------------------- #
# Markdown -> PDF
# --------------------------------------------------------------------------- #


def strip_markdown(line: str) -> str:
    """Flatten one markdown line into the text a rendered PDF would show."""
    line = LINK_RE.sub(r"\1", line)
    line = INLINE_RE.sub("", line)
    line = LIST_RE.sub("", line)
    if line.lstrip().startswith("#"):
        line = line.lstrip("#").strip()
    return line.rstrip()


def wrap_line(text: str, font: pymupdf.Font, width: float) -> list[str]:
    """Greedy character wrap measured with the real font, not an estimate."""
    if not text:
        return [""]
    lines: list[str] = []
    current = ""
    for char in text:
        candidate = current + char
        if font.text_length(candidate, fontsize=FONT_SIZE) > width and current:
            lines.append(current)
            current = char
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


def render_pdf(markdown: str, out_path: Path) -> int:
    """Flow the rendered text across pages; returns the page count.

    Page breaks are forced by vertical space, so a long paragraph is split
    between two pages exactly like in a printed book.
    """
    font = pymupdf.Font(FONT_NAME)
    usable_width = PAGE_WIDTH - 2 * MARGIN
    document = pymupdf.open()
    page = document.new_page(width=PAGE_WIDTH, height=PAGE_HEIGHT)
    y = MARGIN + FONT_SIZE

    for raw in markdown.splitlines():
        line = strip_markdown(raw)
        for piece in wrap_line(line, font, usable_width) if line else [""]:
            if y > PAGE_HEIGHT - MARGIN:
                page = document.new_page(width=PAGE_WIDTH, height=PAGE_HEIGHT)
                y = MARGIN + FONT_SIZE
            if piece:
                page.insert_text((MARGIN, y), piece, fontname=FONT_NAME, fontsize=FONT_SIZE)
            y += LINE_HEIGHT

    page_count = document.page_count
    document.save(str(out_path))
    document.close()
    return page_count


# --------------------------------------------------------------------------- #
# Question files
# --------------------------------------------------------------------------- #


def normalize(text: str) -> str:
    return re.sub(r"\s+", "", text)


def chunk_corpus(root: Path, settings: Settings) -> dict[str, list[ChunkData]]:
    cache: dict[str, list[ChunkData]] = {}
    for path in sorted(root.iterdir()):
        if not path.is_file():
            continue
        parsed = parse_document(path)
        cache[path.name] = chunk_text(
            parsed.text,
            max_chars=settings.chunk_max_chars,
            overlap=settings.chunk_overlap,
            mode=parsed.mode,
            page_starts=parsed.page_starts,
        )
    return cache


def resolve_quote(
    stem: str, quote: str, chunks: dict[str, list[ChunkData]], *, suffix: str
) -> tuple[ChunkData, str] | None:
    """Find the one chunk containing the quote in the corpus of this format."""
    path = f"{stem}{suffix}"
    needle = normalize(quote)
    matches = [chunk for chunk in chunks.get(path, []) if needle in normalize(chunk.text)]
    if len(matches) != 1:
        return None
    return matches[0], path


def build(args: argparse.Namespace) -> int:
    settings = get_settings()
    PDF_CORPUS.mkdir(parents=True, exist_ok=True)

    # 1. Convert every markdown note to a PDF with the same content.
    print(f"rendering {len(list(CORPUS.glob('*.md')))} notes into {PDF_CORPUS} ...")
    for source in sorted(CORPUS.glob("*.md")):
        out = PDF_CORPUS / f"{source.stem}.pdf"
        pages = render_pdf(source.read_text(encoding="utf-8"), out)
        print(
            f"  {source.name:26s} -> {out.name:28s} {pages} pages  {out.stat().st_size // 1024} KB"
        )

    # 2. Resolve the quotes against both formats.
    md_chunks = chunk_corpus(CORPUS, settings)
    pdf_chunks = chunk_corpus(PDF_CORPUS, settings)
    questions = [
        json.loads(line)
        for line in QUESTIONS_SRC.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    md_out: list[dict[str, Any]] = []
    pdf_out: list[dict[str, Any]] = []
    problems: list[str] = []
    cross_page_report: list[str] = []

    for question in questions:
        md_gold, md_ok = _resolve_all(question, md_chunks, ".md", problems)
        pdf_gold, pdf_ok = _resolve_all(question, pdf_chunks, ".pdf", problems)
        if not (md_ok and pdf_ok):
            continue
        md_out.append({**question, "gold": md_gold["gold"], "acceptable": md_gold["acceptable"]})
        pdf_out.append({**question, "gold": pdf_gold["gold"], "acceptable": pdf_gold["acceptable"]})
        if question["category"] == "cross_page":
            pages = [item["page"] for item in pdf_gold["gold"]]
            cross_page_report.append(f"  {question['id']}: pages {pages}")

    if problems:
        print("\nquote resolution failed:", file=sys.stderr)
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        return 2

    out_dir = BASE_DIR / "eval" / "stage6"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (("md-questions.jsonl", md_out), ("pdf-questions.jsonl", pdf_out)):
        path = out_dir / name
        path.write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8"
        )
        print(f"wrote {path.relative_to(BASE_DIR)} ({len(rows)} questions)")

    if cross_page_report:
        print("\ncross-page questions (gold quotes land on these PDF pages):")
        print("\n".join(cross_page_report))
    return 0


def _resolve_all(
    question: dict[str, Any],
    chunks: dict[str, list[ChunkData]],
    suffix: str,
    problems: list[str],
) -> tuple[dict[str, list[dict[str, Any]]], bool]:
    resolved: dict[str, list[dict[str, Any]]] = {"gold": [], "acceptable": []}
    ok = True
    for field in ("gold", "acceptable"):
        for quote in question.get(field, []):
            found = resolve_quote(question["file"], quote, chunks, suffix=suffix)
            if found is None:
                problems.append(
                    f"{question['id']} ({suffix}): quote not found in exactly one chunk: "
                    f"{quote[:30]}…"
                )
                ok = False
                continue
            chunk, path = found
            resolved[field].append(
                {
                    "path": path,
                    "heading": chunk.heading,
                    "quote": quote,
                    # Page is only meaningful for PDFs; kept for the report.
                    "page": chunk.page,
                }
            )
    return resolved, ok


# --------------------------------------------------------------------------- #
# Comparison
# --------------------------------------------------------------------------- #


def compare(args: argparse.Namespace) -> int:
    md_report = json.loads((BASE_DIR / args.md_report).read_text(encoding="utf-8"))
    pdf_report = json.loads((BASE_DIR / args.pdf_report).read_text(encoding="utf-8"))

    mode = args.mode
    md = md_report["modes"][mode]
    pdf = pdf_report["modes"][mode]

    by_category_md = _by_category(md)
    by_category_pdf = _by_category(pdf)

    lines = [
        "# PDF 控制实验：同一内容，两种格式",
        "",
        f"- 模式：`{mode}`，top_k = {md_report['top_k']}",
        f"- markdown 报告：`{args.md_report}`（集合 `{md_report['collection']}`，"
        f"{md_report['generated_at']}）",
        f"- PDF 报告：`{args.pdf_report}`（集合 `{pdf_report['collection']}`，"
        f"{pdf_report['generated_at']}）",
        "- 内容、题目、分块参数相同，唯一变量是格式：PDF 没有标题路径、按页切分、引用以页码定位。",
        "",
        "## 总体",
        "",
        "| 指标 | markdown | PDF | 差值 |",
        "| --- | --- | --- | --- |",
    ]
    for label, key in (
        ("Hit@k", "hit_at_k"),
        ("Recall@k", "recall_at_k"),
        ("MRR@k", "mrr_at_k"),
    ):
        md_value, pdf_value = md[key], pdf[key]
        lines.append(
            f"| {label} | {_pct(md_value)} | {_pct(pdf_value)} | "
            f"{(pdf_value - md_value) * 100:+.1f} pp |"
        )
    md_refusal = md["refusal_accuracy_unanswerable"]
    pdf_refusal = pdf["refusal_accuracy_unanswerable"]
    lines.append(
        f"| 无答案题拒答率 | {_pct(md_refusal)} | {_pct(pdf_refusal)} | "
        f"{(pdf_refusal - md_refusal) * 100:+.1f} pp |"
    )

    lines += [
        "",
        "## 按题型",
        "",
        "| 题型 | 题数 | markdown Recall@k | PDF Recall@k |",
        "| --- | --- | --- | --- |",
    ]
    for category in by_category_md:
        md_items = by_category_md[category]
        pdf_items = by_category_pdf[category]
        lines.append(
            f"| {category} | {len(md_items)} | "
            f"{_pct(_mean(item['recall'] for item in md_items))} | "
            f"{_pct(_mean(item['recall'] for item in pdf_items))} |"
        )

    md_stats = _index_stats(BASE_DIR / args.md_db)
    pdf_stats = _index_stats(BASE_DIR / args.pdf_db)
    if md_stats and pdf_stats:
        lines += [
            "",
            "## 两套索引的形状",
            "",
            "| | 文档 | 分块 | 平均每块字符 |",
            "| --- | --- | --- | --- |",
            f"| markdown | {md_stats['docs']} | {md_stats['chunks']} | {md_stats['avg_chars']} |",
            f"| PDF | {pdf_stats['docs']} | {pdf_stats['chunks']} | {pdf_stats['avg_chars']} |",
            "",
            "这一步很关键：两种格式的差异**不只是页边界**。markdown 按标题切节，"
            "小节各自成块，所以块更多、更短；PDF 没有标题信息，段落直接打包到上限，"
            '块更少、更长。下面的差值是两个因素合起来的效果，不能单独归因给"按页分块"。',
        ]

    lines += [
        "",
        "## 逐题",
        "",
        "| 题号 | 题型 | markdown 命中 | PDF 命中 | PDF 页码 |",
        "| --- | --- | --- | --- | --- |",
    ]
    pages = _pdf_pages(BASE_DIR / "eval" / "stage6" / "pdf-questions.jsonl")
    for md_item in md["per_question"]:
        pdf_item = next(item for item in pdf["per_question"] if item["id"] == md_item["id"])
        lines.append(
            f"| {md_item['id']} | {md_item['category']} | {_hit(md_item)} | {_hit(pdf_item)} | "
            f"{pages.get(md_item['id'], '—')} |"
        )

    lines += [
        "",
        "## 怎么读这张表",
        "",
        "- 这 10 道题只覆盖 control 语料的一小部分，单个题目的差异就是 10 个百分点，"
        "所以看差值的方向与题型分布，不要看小数位。",
        "- `cross_page` 两题的 gold 分列 PDF 的两页：如果每页单独分块伤害了召回，"
        "损失会集中出现在这一行。",
        "- PDF 丢失了 markdown 的标题路径，引用里没有「小节」这一层；这既是格式差异，"
        "也是解析器当前的行为（PDF 无标题信息）。",
        "",
    ]

    out = BASE_DIR / args.out
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out.relative_to(BASE_DIR)}")
    return 0


def _by_category(metrics: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in metrics["per_question"]:
        if item["answerable"]:
            grouped[item["category"]].append(item)
    return dict(grouped)


def _index_stats(db_path: Path) -> dict[str, int] | None:
    """Documents, chunks and average chunk size of one isolated eval index."""
    if not db_path.exists():
        return None
    connection = sqlite3.connect(db_path)
    try:
        docs = connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        chunks = connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        average = connection.execute("SELECT AVG(LENGTH(text)) FROM chunks").fetchone()[0]
    finally:
        connection.close()
    return {"docs": int(docs), "chunks": int(chunks), "avg_chars": round(float(average or 0))}


def _pdf_pages(path: Path) -> dict[str, str]:
    pages: dict[str, str] = {}
    if not path.exists():
        return pages
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        values = sorted({item["page"] for item in row["gold"] if item.get("page")})
        if values:
            pages[row["id"]] = ",".join(str(value) for value in values)
    return pages


def _hit(record: dict[str, Any]) -> str:
    if not record["answerable"]:
        return "—（无答案）"
    return "是" if record["hit"] else "否"


def _mean(values: Any) -> float:
    items = [float(value) for value in values]
    return sum(items) / len(items) if items else 0.0


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def main() -> int:
    parser = argparse.ArgumentParser(description="PDF vs markdown control experiment")
    sub = parser.add_subparsers(dest="command", required=True)

    build_parser = sub.add_parser("build", help="render PDFs and write both question files")
    build_parser.set_defaults(func=build)

    compare_parser = sub.add_parser("compare", help="write the comparison report")
    compare_parser.add_argument(
        "--md-report", default="docs/eval/pdf-control-md/stage2-report.json"
    )
    compare_parser.add_argument(
        "--pdf-report", default="docs/eval/pdf-control-pdf/stage2-report.json"
    )
    compare_parser.add_argument("--md-db", default="data/eval/md.db")
    compare_parser.add_argument("--pdf-db", default="data/eval/pdf.db")
    compare_parser.add_argument("--mode", default="hybrid_rerank")
    compare_parser.add_argument("--out", default="docs/eval/pdf-control-report.md")
    compare_parser.set_defaults(func=compare)

    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
