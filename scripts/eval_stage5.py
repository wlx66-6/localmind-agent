"""Stage 5 evaluation: the same 40 questions on both tracks, plus memory scenarios.

Retrieval runs once per question (hybrid + rerank on the GPU reranker) and the
resulting top-k is fed to *both* providers, so the per-track differences come
from generation alone. Retrieval quality itself is covered by
``eval_retrieval.py``; measuring it twice would just add noise.

Memory scenarios talk to the running backend over HTTP, because extraction
happens in a background task after the SSE stream ends — the only faithful way
to test it is to wait for the real thing.

Usage:
    .venv/Scripts/python.exe scripts/eval_stage5.py --tracks cloud local
    .venv/Scripts/python.exe scripts/eval_stage5.py --tracks cloud --skip-memory

Prerequisites: Qdrant + Ollama running, the corpus indexed and nothing else
(the same index-scope guard as eval_retrieval.py), and for the memory part the
backend on 127.0.0.1:8000. Cloud track sends only retrieved snippets to
DeepSeek — never original files.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import urllib.request
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR / "backend"))
sys.path.insert(0, str(BASE_DIR / "scripts"))

import eval_retrieval as ev  # noqa: E402  (shares gold resolution with stage 2)

from app.config import Settings, get_settings  # noqa: E402
from app.generation.base import ChatProvider, ProviderError  # noqa: E402
from app.generation.deepseek import DeepSeekProvider  # noqa: E402
from app.generation.ollama import OllamaProvider  # noqa: E402
from app.generation.prompt import build_messages, is_refusal  # noqa: E402
from app.retrieval.embedder import OllamaEmbedder  # noqa: E402
from app.retrieval.pipeline import retrieve, should_refuse  # noqa: E402
from app.retrieval.reranker import CrossEncoderReranker  # noqa: E402
from app.retrieval.sparse import warmup as sparse_warmup  # noqa: E402
from app.retrieval.types import RetrievalHit  # noqa: E402
from app.retrieval.vector_store import QdrantStore  # noqa: E402
from app.services.chat import cited_indices  # noqa: E402

API = "http://127.0.0.1:8000"


# --------------------------------------------------------------------------- #
# Track comparison
# --------------------------------------------------------------------------- #


@dataclass
class TrackResult:
    track: str
    records: list[dict[str, Any]] = field(default_factory=list)
    retries: int = 0

    def summary(self, *, answerable: int) -> dict[str, Any]:
        answered = [
            record
            for record in self.records
            if record["answerable"] and not record["refused"] and not record["error"]
        ]
        unanswerable = [record for record in self.records if not record["answerable"]]
        return {
            "track": self.track,
            "questions": len(self.records),
            "generated": sum(1 for record in self.records if record["ttft_ms"] is not None),
            "gate_refusals": sum(1 for record in self.records if record["refused_by"] == "gate"),
            "generation_refusals": sum(
                1 for record in self.records if record["refused_by"] == "generation"
            ),
            "refused_unanswerable": sum(1 for record in unanswerable if record["refused"]),
            "unanswerable": len(unanswerable),
            "false_refusals": sum(1 for record in self.records if record["false_refusal"]),
            "false_answers": sum(1 for record in self.records if record["false_answer"]),
            "answerable": answerable,
            "citation_precision": _mean(
                record["citation_precision"]
                for record in answered
                if "citation_precision" in record
            ),
            "citation_recall": _mean(
                record["citation_recall"] for record in answered if "citation_recall" in record
            ),
            "ttft_ms": _percentiles([record["ttft_ms"] for record in self.records]),
            "total_ms": _percentiles([record["total_ms"] for record in self.records]),
            "retries": self.retries,
            "errors": [record for record in self.records if record["error"]],
        }


async def run_track(
    track: str,
    provider: ChatProvider,
    questions: list[ev.EvalQuestion],
    hits_by_question: dict[str, list[RetrievalHit]],
    gold: dict[str, set[ev.GOLD_KEY]],
    accepted: dict[str, set[ev.GOLD_KEY]],
    settings: Settings,
    *,
    retry: bool = True,
) -> TrackResult:
    result = TrackResult(track=track)

    for question in questions:
        hits = hits_by_question[question.id]
        record: dict[str, Any] = {
            "id": question.id,
            "category": question.category,
            "answerable": question.answerable,
        }
        gate_refused = should_refuse(
            hits,
            min_rerank_score=settings.retrieval_gate_rerank_score,
            min_dense_score=settings.retrieval_min_score,
        )
        if gate_refused:
            record.update(
                refused=True,
                refused_by="gate",
                false_refusal=question.answerable,
                false_answer=False,
                error=None,
                ttft_ms=None,
                total_ms=None,
                answer="",
                cited=[],
            )
            result.records.append(record)
            continue

        messages = build_messages(question.question, hits, [])
        started = time.perf_counter()
        parts, first_token_at, error = await _generate(provider, messages)
        if error is not None and retry:
            # Infra hiccups (Ollama's llama-server occasionally fails to
            # initialise CUDA on the first call) are not model behaviour;
            # retry once and count it, so the report shows both.
            print(f"  {track} {question.id}: retrying after error: {error[:80]}", flush=True)
            result.retries += 1
            started = time.perf_counter()
            parts, first_token_at, error = await _generate(provider, messages)
        total_ms = (time.perf_counter() - started) * 1000
        answer = "".join(parts)
        cited = cited_indices(answer, len(hits))
        model_refused = is_refusal(answer) if not error else True
        strict = gold.get(question.id, set())
        wide = accepted.get(question.id, set())
        cited_wide = sum(1 for index in cited if hit_key(hits, index) in wide)
        cited_strict = sum(1 for index in cited if hit_key(hits, index) in strict)
        gold_indices = [
            index for index, hit in enumerate(hits, start=1) if ev.hit_key(hit) in strict
        ]

        record.update(
            refused=model_refused,
            refused_by="generation" if model_refused else None,
            false_refusal=question.answerable and model_refused,
            false_answer=(not question.answerable) and not model_refused,
            error=error,
            answer=answer[:240],
            cited=cited,
            ttft_ms=round((first_token_at - started) * 1000, 2) if first_token_at else None,
            total_ms=round(total_ms, 2),
        )
        if question.answerable and not model_refused and not error:
            record["citation_precision"] = cited_wide / len(cited) if cited else 0.0
            record["citation_recall"] = (
                len(set(cited) & set(gold_indices)) / len(gold_indices) if gold_indices else 0.0
            )
            record["cited_strict"] = cited_strict
        result.records.append(record)
        print(f"  {track} {question.id}: {record['answer'][:40]!r}", flush=True)

    return result


async def _generate(
    provider: ChatProvider, messages: Any
) -> tuple[list[str], float | None, str | None]:
    """Stream one answer; return (parts, first_token_at, error)."""
    parts: list[str] = []
    first_token_at: float | None = None
    try:
        async for delta in provider.stream_chat(messages):
            if first_token_at is None:
                first_token_at = time.perf_counter()
            parts.append(delta)
    except ProviderError as exc:
        return parts, first_token_at, str(exc)
    return parts, first_token_at, None


def hit_key(hits: list[RetrievalHit], index: int) -> ev.GOLD_KEY:
    hit = hits[index - 1]
    return ev.hit_key(hit)


def _mean(values: Any) -> float:
    items = [float(value) for value in values]
    return sum(items) / len(items) if items else 0.0


def _percentiles(values: list[float | None]) -> dict[str, float]:
    ordered = sorted(value for value in values if isinstance(value, (int, float)))
    if not ordered:
        return {"p50": 0.0, "p95": 0.0, "mean": 0.0}
    p95 = ordered[min(len(ordered) - 1, round(0.95 * (len(ordered) - 1)))]
    return {
        "p50": round(_median(ordered), 2),
        "p95": round(p95, 2),
        "mean": round(sum(ordered) / len(ordered), 2),
    }


def _median(ordered: list[float]) -> float:
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2


# --------------------------------------------------------------------------- #
# Memory scenarios (HTTP, against the running backend)
# --------------------------------------------------------------------------- #


def post_json(path: str, payload: dict[str, Any], *, timeout: float = 300) -> Any:
    request = urllib.request.Request(
        f"{API}{path}",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read().decode("utf-8")
    return json.loads(body) if body else None


def get_json(path: str) -> Any:
    with urllib.request.urlopen(f"{API}{path}", timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def delete(path: str) -> int:
    request = urllib.request.Request(f"{API}{path}", method="DELETE")
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.status


def chat(message: str, *, track: str = "cloud") -> dict[str, Any]:
    """One turn; returns the citations event plus the final answer text."""
    request = urllib.request.Request(
        f"{API}/api/chat",
        data=json.dumps({"message": message, "track": track}, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    citations: dict[str, Any] = {}
    parts: list[str] = []
    event = ""
    with urllib.request.urlopen(request, timeout=600) as response:
        for raw in response:
            line = raw.decode("utf-8").strip()
            if line.startswith("event:"):
                event = line[6:].strip()
            elif line.startswith("data:"):
                payload = json.loads(line[5:].strip())
                if event == "citations":
                    citations = payload
                elif event == "token":
                    parts.append(str(payload.get("text", "")))
                elif event == "error":
                    citations["error"] = payload.get("message")
    return {"citations": citations, "answer": "".join(parts)}


def wait_for_memory(key: str, *, timeout: float = 45) -> dict[str, Any] | None:
    """Poll the panel API until a memory with this key is active."""
    return _poll_memories(lambda memory: memory["key"] == key, timeout=timeout)


def wait_for_memory_content(text: str, *, timeout: float = 45) -> dict[str, Any] | None:
    """Same, but for keys the extractor invents (it chooses its own wording)."""
    return _poll_memories(lambda memory: text in memory["content"], timeout=timeout)


def _poll_memories(predicate: Any, *, timeout: float) -> dict[str, Any] | None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for memory in get_json("/api/memories"):
            if predicate(memory):
                return memory
        time.sleep(2)
    return None


def run_memory_scenarios() -> list[dict[str, Any]]:
    scenarios: list[dict[str, Any]] = []

    # 1. Cross-conversation recall with a date.
    chat("记一下：我住在杭州，平时主要用 Python 写后端。")
    stored = wait_for_memory("所在城市")
    recall = chat("你还记得我住在哪个城市吗？")
    recalled = recall["citations"].get("memories", [])
    scenarios.append(
        {
            "id": "memory-1",
            "name": "跨会话召回事实",
            "passed": bool(stored)
            and any(memory["key"] == "所在城市" for memory in recalled)
            and "杭州" in recall["answer"],
            "evidence": {
                "stored": stored,
                "recalled": [f"{m['kind']}/{m['key']} {m['score']:.3f}" for m in recalled],
                "answer": recall["answer"][:160],
            },
        }
    )

    # 2. Same key supersedes the old value.
    chat("我换工作了，现在主要用 Go 写后端。")
    time.sleep(6)
    active = [m for m in get_json("/api/memories") if m["key"] == "技术栈偏好"]
    history = [
        m for m in get_json("/api/memories?include_superseded=true") if m["key"] == "技术栈偏好"
    ]
    recall = chat("我现在后端主要用什么语言？")
    recalled = recall["citations"].get("memories", [])
    scenarios.append(
        {
            "id": "memory-2",
            "name": "同 key 覆盖旧值",
            "passed": len(active) == 1
            and len(history) >= 2
            and any(m["key"] == "技术栈偏好" for m in recalled)
            and "Go" in recall["answer"],
            "evidence": {
                "active": [m["content"] for m in active],
                "history": [f"{m['content']} ({m['status']})" for m in history],
                "recalled": [f"{m['key']} {m['score']:.3f}" for m in recalled],
                "answer": recall["answer"][:160],
            },
        }
    )

    # 3. An unrelated question must not drag memories into the prompt.
    unrelated = chat("我们公司明年的预算是多少？")
    refused_answer = is_refusal(unrelated["answer"]) or bool(unrelated["citations"].get("refused"))
    scenarios.append(
        {
            "id": "memory-3",
            "name": "无关问题不召回记忆",
            "passed": not unrelated["citations"].get("memories") and refused_answer,
            "evidence": {
                "memories": unrelated["citations"].get("memories", []),
                "refused": unrelated["citations"].get("refused"),
                "answer": unrelated["answer"][:160],
            },
        }
    )

    # 4. Event date is kept separate from the recording date.
    chat("去年八月我开始学 Rust。")
    rust = wait_for_memory_content("Rust")
    scenarios.append(
        {
            "id": "memory-4",
            "name": "事件时间与记录时间分开",
            # "去年八月" relative to today must resolve to 2025-08, while
            # created_at today stays 2026-10.
            "passed": rust is not None and str(rust.get("occurred_at", "")).startswith("2025-08"),
            "evidence": {"stored": rust},
        }
    )

    # 5. Deleting a memory removes it from recall.
    target = next((m for m in get_json("/api/memories") if m["key"] == "所在城市"), None)
    deleted = delete(f"/api/memories/{target['id']}") if target else None
    after = chat("我住在哪个城市？")
    scenarios.append(
        {
            "id": "memory-5",
            "name": "删除后不再召回",
            "passed": target is not None
            and deleted == 204
            and not any(
                memory["key"] == "所在城市" for memory in after["citations"].get("memories", [])
            ),
            "evidence": {
                "deleted": target["content"] if target else None,
                "recalled": [m["key"] for m in after["citations"].get("memories", [])],
                "answer": after["answer"][:160],
            },
        }
    )
    return scenarios


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #


def render_markdown(report: dict[str, Any]) -> str:
    tracks = report["tracks"]
    lines = [
        "# 阶段 5 评测：同一批问题，两条轨道",
        "",
        f"- 生成时间：{report['generated_at']}",
        f"- 评测集：`{report['questions_file']}`（{report['questions']} 题："
        f"{report['answerable']} 有答案 / {report['unanswerable']} 无答案）",
        f"- 集合：`{report['collection']}`，top_k = {report['top_k']}",
        "- 方法：每题检索一次（hybrid + rerank），同一组片段分别交给两条轨道生成，"
        "因此差值只来自生成层；检索指标见 `docs/eval/stage2-report.md`。",
        "- 云端只接收命中的片段，不接收任何原始文件。",
        "",
        "## 双轨对比",
        "",
        "| 轨道 | 生成回答 | 首 token p50/p95 (ms) | 总耗时 p50/p95 (ms) | 闸门拒答 "
        "| 生成拒答 | 无答案题拒答 | 有答案题误拒 | 无答案题误答 | 引用精确率 | 引用召回 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for track in report["track_order"]:
        summary = tracks[track]
        lines.append(
            f"| {track} | {summary['generated']} | {_fmt_ms(summary['ttft_ms'])} | "
            f"{_fmt_ms(summary['total_ms'])} | {summary['gate_refusals']} | "
            f"{summary['generation_refusals']} | "
            f"{summary['refused_unanswerable']}/{summary['unanswerable']} | "
            f"{summary['false_refusals']}/{summary['answerable']} | {summary['false_answers']} | "
            f"{_pct(summary['citation_precision'])} | {_pct(summary['citation_recall'])} |"
        )

    retries = sum(tracks[track]["retries"] for track in report["track_order"])
    lines += [
        "",
        "本地轨道的首题包含模型加载（3～4 s），p95 会被它抬高；批处理里的后续题才代表稳态。",
        "",
        "两条轨道共用同一次检索（CUDA 重排），而生产环境的本地轨道把重排放到 CPU 以让出显存。"
        "在 8 GB 显存的机器上，4B 生成模型与 CUDA 重排共存会互相拖慢，"
        "因此本表的本地耗时是**悲观值**；生产路径见 `docs/spikes/stage3-acceptance.md`。",
    ]
    if retries:
        lines.append(
            f"本次共重试 {retries} 次：Ollama 首次调用偶发 CUDA 初始化失败，重试后成功，"
            "不属于模型行为。"
        )
    lines += [
        "",
        "## 逐题对照",
        "",
    ]
    header = "| 题号 | 类别 | 有答案 |"
    separator = "| --- | --- | --- |"
    for track in report["track_order"]:
        header += f" {track} 结果 |"
        separator += " --- |"
    lines += [header, separator]
    first = {record["id"]: record for record in tracks[report["track_order"][0]]["records"]}
    others = {
        track: {record["id"]: record for record in tracks[track]["records"]}
        for track in report["track_order"][1:]
    }
    for question_id, base in first.items():
        row = (
            f"| {question_id} | {base['category']} | {'是' if base['answerable'] else '否'} | "
            f"{_verdict(base)} |"
        )
        for track in report["track_order"][1:]:
            row += f" {_verdict(others[track][question_id])} |"
        lines.append(row)

    memory = report.get("memory")
    if memory:
        lines += [
            "",
            "## 记忆场景",
            "",
            "| 场景 | 结果 | 证据 |",
            "| --- | --- | --- |",
        ]
        for scenario in memory:
            evidence = json.dumps(scenario["evidence"], ensure_ascii=False)[:300]
            lines.append(
                f"| {scenario['name']} | {'通过' if scenario['passed'] else '未通过'} | "
                f"`{evidence}` |"
            )
        passed = sum(1 for item in memory if item["passed"])
        lines += ["", f"通过 {passed}/{len(memory)}。"]

    failed = []
    for track in report["track_order"]:
        for record in tracks[track]["records"]:
            if record["false_answer"] or record["false_refusal"] or record["error"]:
                failed.append(
                    f"- {track} {record['id']}: {_verdict(record)}；回答：{record['answer'][:120]}"
                )
    lines += ["", "## 失败样本", ""]
    lines += failed or ["（无）"]
    return "\n".join(lines) + "\n"


def _verdict(record: dict[str, Any]) -> str:
    if record["error"]:
        return f"出错：{record['error'][:40]}"
    if record["false_answer"]:
        return "误答"
    if record["false_refusal"]:
        return "误拒"
    if record["refused"]:
        return "正确拒答" if not record["answerable"] else "拒答"
    if not record["answerable"]:
        return "通过了闸门（由生成层判断）"
    cited = record.get("cited") or []
    return f"回答，引用 {len(cited)} 条"


def _fmt_ms(values: dict[str, float]) -> str:
    return f"{values['p50']:.0f} / {values['p95']:.0f}"


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


async def run(args: argparse.Namespace) -> int:
    settings = get_settings()
    questions = ev.load_questions(Path(args.questions))
    if args.limit:
        questions = questions[: args.limit]
    corpus_root = Path(args.corpus_root)

    outside = ev.documents_outside_corpus(settings, corpus_root)
    if outside and not args.allow_extra_documents:
        listing = "\n  ".join(outside[:5])
        print(
            "index contains documents outside the corpus root; results would not be "
            f"comparable. Remove those sources or pass --allow-extra-documents.\n  {listing}",
            file=sys.stderr,
        )
        return 2

    chunks = ev.chunk_corpus(corpus_root, settings)
    gold: dict[str, set[ev.GOLD_KEY]] = {}
    accepted: dict[str, set[ev.GOLD_KEY]] = {}
    problems: list[str] = []
    for question in questions:
        if not question.answerable:
            continue
        items, issue = ev.resolve_items(question.id, question.gold, chunks, label="gold")
        extras, extra_issue = ev.resolve_items(
            question.id, question.acceptable, chunks, label="acceptable"
        )
        gold[question.id] = {item.key for item in items}
        accepted[question.id] = {item.key for item in [*items, *extras]}
        problems.extend(issue + extra_issue)
    if problems:
        for problem in problems:
            print(f"annotation resolution failed: {problem}", file=sys.stderr)
        return 2

    tracks = [track for track in args.tracks if track in {"cloud", "local"}]
    if not tracks:
        print("no valid --tracks given", file=sys.stderr)
        return 2
    if "cloud" in tracks and not settings.has_cloud_key:
        print("cloud track requested but DEEPSEEK_API_KEY is empty", file=sys.stderr)
        return 2

    sparse_warmup()
    embedder = OllamaEmbedder(settings.ollama_host, settings.ollama_embed_model, settings.embed_dim)
    store = QdrantStore(settings.qdrant_url, settings.qdrant_collection, settings.embed_dim)
    reranker = CrossEncoderReranker(
        settings.rerank_model,
        device=settings.rerank_device,
        batch_size=settings.rerank_batch_size,
        cache_folder=settings.rerank_cache_folder,
    )
    await reranker.warmup()

    providers: dict[str, ChatProvider] = {}
    if "cloud" in tracks:
        providers["cloud"] = DeepSeekProvider(
            settings.deepseek_api_key, settings.deepseek_base_url, settings.deepseek_model
        )
    if "local" in tracks:
        providers["local"] = OllamaProvider(
            settings.ollama_host, settings.ollama_chat_model, timeout=settings.ollama_chat_timeout
        )

    try:
        print(f"retrieving {len(questions)} questions once ...", flush=True)
        hits_by_question: dict[str, list[RetrievalHit]] = {}
        retrieval_ms: list[float] = []
        for question in questions:
            started = time.perf_counter()
            result = await retrieve(
                embedder,
                store,
                reranker,
                question.question,
                mode="hybrid_rerank",
                top_k=args.top_k,
                top_n=settings.retrieval_top_n,
                rrf_k=settings.rrf_k,
            )
            retrieval_ms.append((time.perf_counter() - started) * 1000)
            hits_by_question[question.id] = result.hits

        results: list[TrackResult] = []
        for track in tracks:
            print(f"generating with {track} ...", flush=True)
            results.append(
                await run_track(
                    track,
                    providers[track],
                    questions,
                    hits_by_question,
                    gold,
                    accepted,
                    settings,
                    retry=not args.no_retry,
                )
            )
    finally:
        await embedder.aclose()
        await store.close()

    answerable = sum(1 for question in questions if question.answerable)
    report: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "questions_file": Path(args.questions).relative_to(BASE_DIR).as_posix(),
        "corpus_root": str(corpus_root),
        "collection": settings.qdrant_collection,
        "questions": len(questions),
        "answerable": answerable,
        "unanswerable": len(questions) - answerable,
        "top_k": args.top_k,
        "retrieval_ms": _percentiles(retrieval_ms),
        "track_order": tracks,
        "tracks": {
            result.track: {**result.summary(answerable=answerable), "records": result.records}
            for result in results
        },
    }

    if not args.skip_memory:
        print("running memory scenarios against the backend ...", flush=True)
        report["memory"] = run_memory_scenarios()
        for scenario in report["memory"]:
            print(
                f"  {scenario['id']}: {'PASS' if scenario['passed'] else 'FAIL'}",
                flush=True,
            )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "stage5-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "stage5-report.md").write_text(render_markdown(report), encoding="utf-8")
    print(f"wrote {out_dir / 'stage5-report.json'}")
    print(f"wrote {out_dir / 'stage5-report.md'}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="LocalMind stage 5 evaluation")
    parser.add_argument("--questions", default=str(BASE_DIR / "eval/stage2/questions.jsonl"))
    parser.add_argument("--corpus-root", default=str(BASE_DIR / "examples/corpus"))
    parser.add_argument("--tracks", nargs="+", default=["cloud", "local"])
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--out", default=str(BASE_DIR / "docs/eval"))
    parser.add_argument("--limit", type=int, default=0, help="only the first N questions")
    parser.add_argument("--skip-memory", action="store_true")
    parser.add_argument(
        "--no-retry",
        action="store_true",
        help="do not retry a failed provider call (default: one retry)",
    )
    parser.add_argument("--allow-extra-documents", action="store_true")
    return asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
