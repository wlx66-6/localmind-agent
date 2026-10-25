"""Summarise what the running system actually did, straight from SQLite.

Every answer already stores its telemetry in ``messages.metrics_json`` (track,
ttft, total, refusal layer, egress receipt, memory extraction status), so the
report is a read-only aggregation — no server needs to be running and nothing
leaves the machine.

Usage:
    .venv/Scripts/python.exe scripts/metrics_report.py
    .venv/Scripts/python.exe scripts/metrics_report.py --out docs/eval/stage5-metrics.md
    .venv/Scripts/python.exe scripts/metrics_report.py --days 7

Markdown goes to stdout unless ``--out`` is given, so it can be quoted in a
README or committed as an acceptance artifact.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.config import get_settings

BASE_DIR = Path(__file__).resolve().parents[1]


@dataclass
class TrackStats:
    messages: int = 0
    generated: int = 0
    refused_gate: int = 0
    refused_model: int = 0
    refused_other: int = 0
    skipped: int = 0
    blocked: int = 0
    ttft: list[float] = field(default_factory=list)
    total: list[float] = field(default_factory=list)
    rerank: list[float] = field(default_factory=list)
    egress_chunks: int = 0
    egress_chars: int = 0
    egress_memories: int = 0
    rerank_failed: int = 0
    # Snippets that were retrieved but stayed on this machine (local track or
    # "retrieve only"): useful to show the privacy claim is not an empty one.
    local_chunks: int = 0
    errors: int = 0


@dataclass
class MemoryStats:
    extraction: Counter[str] = field(default_factory=Counter)
    summary: Counter[str] = field(default_factory=Counter)
    extraction_ms: list[float] = field(default_factory=list)
    stored: int = 0
    superseded: int = 0


def load_rows(connection: sqlite3.Connection, *, since: datetime | None) -> list[dict[str, Any]]:
    query = (
        "SELECT id, role, track, metrics_json, citations_json, created_at "
        "FROM messages WHERE role = 'assistant'"
    )
    params: list[str] = []
    if since is not None:
        query += " AND created_at >= ?"
        params.append(since.strftime("%Y-%m-%d %H:%M:%S"))
    rows = connection.execute(query + " ORDER BY id", params).fetchall()
    records = []
    for message_id, role, track, metrics_json, citations_json, created_at in rows:
        try:
            metrics = json.loads(metrics_json or "{}")
        except json.JSONDecodeError:
            metrics = {}
        if not isinstance(metrics, dict):
            metrics = {}
        records.append(
            {
                "id": message_id,
                "role": role,
                "track": track or metrics.get("track") or "unknown",
                "metrics": metrics,
                "citations_json": citations_json,
                "created_at": created_at,
            }
        )
    return records


def collect(records: list[dict[str, Any]]) -> tuple[dict[str, TrackStats], MemoryStats]:
    tracks: dict[str, TrackStats] = defaultdict(TrackStats)
    memory = MemoryStats()

    for record in records:
        metrics = record["metrics"]
        stats = tracks[str(record["track"])]
        stats.messages += 1
        if metrics.get("skipped"):
            stats.skipped += 1
        if metrics.get("blocked"):
            stats.blocked += 1
        if metrics.get("refused"):
            stats.refused_model += 1
        ttft = metrics.get("ttft_ms")
        if isinstance(ttft, (int, float)):
            stats.generated += 1
            stats.ttft.append(float(ttft))
        total = metrics.get("total_ms")
        if isinstance(total, (int, float)):
            stats.total.append(float(total))

        retrieval = metrics.get("retrieval")
        if isinstance(retrieval, dict):
            timings = retrieval.get("timings")
            if isinstance(timings, dict):
                rerank_ms = timings.get("rerank_ms")
                if isinstance(rerank_ms, (int, float)) and rerank_ms > 0:
                    stats.rerank.append(float(rerank_ms))
            if retrieval.get("rerank_failed"):
                stats.rerank_failed += 1

        egress = metrics.get("egress")
        if isinstance(egress, dict):
            # The receipt always counts the snippets a turn used; `enabled`
            # says whether they actually left the machine. Local turns and
            # "retrieve only" turns carry chunks but are not egress.
            if egress.get("enabled"):
                stats.egress_chunks += int(egress.get("chunks") or 0)
                stats.egress_chars += int(egress.get("chars") or 0)
                stats.egress_memories += int(egress.get("memories") or 0)
            else:
                stats.local_chunks += int(egress.get("chunks") or 0)

        memory_metrics = metrics.get("memory")
        if isinstance(memory_metrics, dict):
            extraction = memory_metrics.get("extraction")
            if isinstance(extraction, dict):
                memory.extraction[str(extraction.get("status") or "unknown")] += 1
                stored = extraction.get("stored")
                if isinstance(stored, int):
                    memory.stored += stored
                superseded = extraction.get("superseded")
                if isinstance(superseded, int):
                    memory.superseded += superseded
                ms = extraction.get("ms")
                if isinstance(ms, (int, float)):
                    memory.extraction_ms.append(float(ms))
            summary = memory_metrics.get("summary")
            if isinstance(summary, dict):
                memory.summary[str(summary.get("status") or "unknown")] += 1

    # The gate refuses before a provider is called, so it has no ttft. Count it
    # from the stored refusal flag plus the absence of generation.
    for record in records:
        metrics = record["metrics"]
        if metrics.get("refused") and metrics.get("ttft_ms") is None:
            tracks[str(record["track"])].refused_gate += 1
            tracks[str(record["track"])].refused_model -= 1
    return dict(tracks), memory


def database_counts(connection: sqlite3.Connection) -> dict[str, Any]:
    def scalar(query: str) -> int:
        return int(connection.execute(query).fetchone()[0])

    suffixes: Counter[str] = Counter()
    for (rel_path,) in connection.execute("SELECT rel_path FROM documents"):
        suffixes[Path(str(rel_path)).suffix.lower() or "(none)"] += 1

    return {
        "sources": scalar("SELECT COUNT(*) FROM sources"),
        "documents": scalar("SELECT COUNT(*) FROM documents"),
        "chunks": scalar("SELECT COUNT(*) FROM chunks"),
        "pdf_documents": scalar("SELECT COUNT(*) FROM documents WHERE page_count IS NOT NULL"),
        "page_chunks": scalar("SELECT COUNT(*) FROM chunks WHERE page IS NOT NULL"),
        "conversations": scalar("SELECT COUNT(*) FROM conversations"),
        "messages": scalar("SELECT COUNT(*) FROM messages"),
        "memories_active": scalar("SELECT COUNT(*) FROM memories WHERE status = 'active'"),
        "memories_superseded": scalar("SELECT COUNT(*) FROM memories WHERE status = 'superseded'"),
        "memories_unembedded": scalar("SELECT COUNT(*) FROM memories WHERE embedding_ok = 0"),
        "documents_by_suffix": dict(sorted(suffixes.items())),
    }


def _percentiles(values: list[float]) -> str:
    if not values:
        return "—"
    ordered = sorted(values)
    p95 = ordered[min(len(ordered) - 1, round(0.95 * (len(ordered) - 1)))]
    return f"{statistics.median(ordered):.0f} / {p95:.0f}"


def render(
    *,
    counts: dict[str, Any],
    tracks: dict[str, TrackStats],
    memory: MemoryStats,
    since: datetime | None,
    database: Path,
) -> str:
    generated_at = datetime.now(UTC).astimezone().isoformat(timespec="seconds")
    scope = f"最近 {since:%Y-%m-%d %H:%M} 之后" if since else "全部历史"
    lines = [
        "# LocalMind 运行指标",
        "",
        f"- 生成时间：{generated_at}",
        f"- 数据来源：`{database}`（只读，`messages.metrics_json`）",
        f"- 统计范围：{scope}",
        "",
        "## 1. 数据规模",
        "",
        "| 项 | 数量 |",
        "| --- | --- |",
        f"| 资料源 | {counts['sources']} |",
        f"| 文档 | {counts['documents']}（其中 PDF {counts['pdf_documents']}） |",
        f"| 分块 | {counts['chunks']}（其中带页码 {counts['page_chunks']}） |",
        f"| 会话 / 消息 | {counts['conversations']} / {counts['messages']} |",
        f"| 记忆 | active {counts['memories_active']}，已替换 "
        f"{counts['memories_superseded']}，未入索引 {counts['memories_unembedded']} |",
        "",
        "按类型："
        + "，".join(
            f"`{suffix}` {count}" for suffix, count in counts["documents_by_suffix"].items()
        ),
        "",
        "## 2. 双轨表现",
        "",
        "| 轨道 | 回答数 | 首 token p50/p95 (ms) | 总耗时 p50/p95 (ms) "
        "| 重排 p50/p95 (ms) | 闸门拒答 | 生成拒答 | 仅检索 | 策略拦截 | 重排降级 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for track in sorted(tracks):
        stats = tracks[track]
        lines.append(
            f"| {track} | {stats.messages} | {_percentiles(stats.ttft)} | "
            f"{_percentiles(stats.total)} | {_percentiles(stats.rerank)} | "
            f"{stats.refused_gate} | {stats.refused_model} | {stats.skipped} | "
            f"{stats.blocked} | {stats.rerank_failed} |"
        )
    if not tracks:
        lines.append("| （没有记录） | | | | | | | | | |")

    lines += [
        "",
        "## 3. 出站",
        "",
        "| 轨道 | 实际出站片段 | 实际出站字符 | 实际出站记忆 | 未出站片段（本机） |",
        "| --- | --- | --- | --- | --- |",
    ]
    for track in sorted(tracks):
        stats = tracks[track]
        lines.append(
            f"| {track} | {stats.egress_chunks} | {stats.egress_chars} | "
            f"{stats.egress_memories} | {stats.local_chunks} |"
        )
    lines += [
        "",
        "「实际出站」只统计回执 `enabled=true` 的轮次：本地轨道与「仅检索」轮次"
        "即使命中了片段也不计入，它们只出现在最后一列。",
        "本地轨道实际出站应为 0。"
        if "local" not in tracks or tracks["local"].egress_chars == 0
        else "注意：本地轨道出现了非零出站，需要排查。",
        "",
        "## 4. 记忆抽取",
        "",
        "| 状态 | 次数 |",
        "| --- | --- |",
    ]
    for status, count in sorted(memory.extraction.items()):
        lines.append(f"| {status} | {count} |")
    if not memory.extraction:
        lines.append("| （没有记录） | |")
    lines += [
        "",
        f"- 抽取出的记忆：{memory.stored} 条，被覆盖：{memory.superseded} 条",
        f"- 抽取耗时 p50/p95：{_percentiles(memory.extraction_ms)} ms",
        "",
        "会话摘要状态："
        + (
            "，".join(f"`{status}` {count}" for status, count in sorted(memory.summary.items()))
            or "无记录"
        ),
        "",
        "## 5. 说明",
        "",
        "- 全部指标来自每条回答落库的 `metrics_json`，不需要重新跑任何请求。",
        "- 「闸门拒答」在调用生成模型之前发生，没有首 token；它与「生成拒答」分开统计。",
        "- 出站字符数只统计真正发往云端的片段与记忆正文，不含 prompt 模板。",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarise LocalMind runtime metrics")
    parser.add_argument("--days", type=int, default=0, help="limit to the last N days (0 = all)")
    parser.add_argument("--out", default="", help="write Markdown here instead of stdout")
    args = parser.parse_args()

    settings = get_settings()
    database = Path(settings.sqlite_path)
    if not database.exists():
        print(f"database not found: {database}", file=sys.stderr)
        return 2

    since = datetime.now(UTC) - timedelta(days=args.days) if args.days > 0 else None
    connection = sqlite3.connect(database)
    try:
        counts = database_counts(connection)
        records = load_rows(connection, since=since)
    finally:
        connection.close()

    tracks, memory = collect(records)
    report = render(counts=counts, tracks=tracks, memory=memory, since=since, database=database)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(report, encoding="utf-8")
        print(f"wrote {out_path}")
    else:
        print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
