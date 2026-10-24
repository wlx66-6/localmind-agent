"""Stage 2 retrieval evaluation: vector vs hybrid vs hybrid+rerank.

The gold set stores a verbatim quote from the corpus, not a chunk id, so the
dataset survives re-chunking. Quotes are resolved to chunks at runtime by
re-parsing the corpus with the same chunker the ingester uses.

Each answerable question has two annotations:

* ``gold`` — the minimal passage(s) that contain the answer. Drives retrieval
  metrics (Hit/Recall/MRR) and citation recall.
* ``acceptable`` — additional passages that also support a correct answer.
  Only used to judge citation precision, so a multi-part question is not
  penalised for citing a second relevant snippet. Deliberately annotated from
  the corpus content alone, not from model output.

Usage:
    .venv/Scripts/python.exe scripts/eval_retrieval.py
    .venv/Scripts/python.exe scripts/eval_retrieval.py --full   # adds cloud generation

``--full`` sends the retrieved snippets to DeepSeek once per answerable
question. It never uploads original files, only the top-k chunks.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.config import RetrievalMode, Settings, get_settings
from app.generation.base import ProviderError
from app.generation.deepseek import DeepSeekProvider
from app.generation.prompt import build_messages, is_refusal
from app.ingestion.chunker import ChunkData, chunk_text
from app.ingestion.parser import parse_document
from app.retrieval.embedder import OllamaEmbedder
from app.retrieval.pipeline import RetrievalResult, retrieve, should_refuse
from app.retrieval.reranker import CrossEncoderReranker
from app.retrieval.sparse import warmup as sparse_warmup
from app.retrieval.types import RetrievalHit
from app.retrieval.vector_store import QdrantStore
from app.services.chat import cited_indices

BASE_DIR = Path(__file__).resolve().parents[1]
GOLD_KEY = tuple[str, str, int, int]


@dataclass(frozen=True)
class EvalQuestion:
    id: str
    category: str
    question: str
    gold: list[dict[str, str]]
    acceptable: list[dict[str, str]]

    @property
    def answerable(self) -> bool:
        return bool(self.gold)


@dataclass(frozen=True)
class ResolvedGold:
    key: GOLD_KEY
    text: str


def normalize(text: str) -> str:
    return re.sub(r"\s+", "", text)


def load_questions(path: Path) -> list[EvalQuestion]:
    questions = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        questions.append(
            EvalQuestion(
                id=raw["id"],
                category=raw["category"],
                question=raw["question"],
                gold=raw.get("gold", []),
                acceptable=raw.get("acceptable", []),
            )
        )
    return questions


def chunk_corpus(corpus_root: Path, settings: Settings) -> dict[str, list[ChunkData]]:
    cache: dict[str, list[ChunkData]] = {}
    for path in sorted(corpus_root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(corpus_root).as_posix()
        parsed = parse_document(path)
        cache[rel] = chunk_text(
            parsed.text,
            max_chars=settings.chunk_max_chars,
            overlap=settings.chunk_overlap,
            markdown=parsed.markdown,
        )
    return cache


def resolve_items(
    question_id: str,
    items: list[dict[str, str]],
    chunks: dict[str, list[ChunkData]],
    *,
    label: str,
) -> tuple[list[ResolvedGold], list[str]]:
    resolved: list[ResolvedGold] = []
    problems: list[str] = []
    for item in items:
        path = item["path"]
        quote = normalize(item["quote"])
        candidates = [
            chunk
            for chunk in chunks.get(path, [])
            if quote in normalize(chunk.text) and chunk.heading == item["heading"]
        ]
        if len(candidates) != 1:
            problems.append(
                f"{question_id} ({label}): expected 1 chunk for {path} / {item['heading']}, "
                f"found {len(candidates)}"
            )
            continue
        chunk = candidates[0]
        resolved.append(
            ResolvedGold(
                key=(path, chunk.heading, chunk.start_line, chunk.end_line),
                text=chunk.text,
            )
        )
    return resolved, problems


def hit_key(hit: RetrievalHit) -> GOLD_KEY:
    return (hit.path, hit.heading, hit.start_line, hit.end_line)


def score_question(
    gold: list[ResolvedGold], hits: list[RetrievalHit], *, top_k: int
) -> dict[str, Any]:
    gold_keys = {item.key for item in gold}
    top = hits[:top_k]
    ranks = [rank for rank, hit in enumerate(top, start=1) if hit_key(hit) in gold_keys]
    first_rank = ranks[0] if ranks else None
    matched = {hit_key(hit) for hit in top} & gold_keys
    return {
        "hit": first_rank is not None,
        "recall": len(matched) / len(gold_keys) if gold_keys else 0.0,
        "rrr": 1.0 / first_rank if first_rank else 0.0,
        "first_rank": first_rank,
    }


def top_score(hit: RetrievalHit | None, mode: RetrievalMode) -> float | None:
    if hit is None:
        return None
    # Mirrors should_refuse(): rerank score when present, cosine otherwise.
    # RRF fused scores are rank-derived and not comparable to a fixed threshold.
    if mode == "hybrid_rerank" and hit.rerank_score is not None:
        return hit.rerank_score
    return hit.dense_score if hit.dense_score is not None else hit.score


async def evaluate_mode(
    mode: RetrievalMode,
    questions: list[EvalQuestion],
    resolved: dict[str, list[ResolvedGold]],
    embedder: OllamaEmbedder,
    store: QdrantStore,
    reranker: CrossEncoderReranker,
    settings: Settings,
    *,
    top_k: int,
) -> dict[str, Any]:
    per_question: list[dict[str, Any]] = []
    latencies: list[float] = []

    for question in questions:
        started = time.perf_counter()
        result = await retrieve(
            embedder,
            store,
            reranker,
            question.question,
            mode=mode,
            top_k=top_k,
            top_n=settings.retrieval_top_n,
            rrf_k=settings.rrf_k,
        )
        latencies.append((time.perf_counter() - started) * 1000)
        per_question.append(_question_record(question, resolved, result, mode, top_k))

    answerable = [item for item in per_question if item["answerable"]]
    unanswerable = [item for item in per_question if not item["answerable"]]
    threshold = (
        settings.retrieval_gate_rerank_score
        if mode == "hybrid_rerank"
        else settings.retrieval_min_score
    )
    refused_unanswerable = [item for item in unanswerable if (item["top_score"] or 0.0) < threshold]
    false_refusals = [item for item in answerable if (item["top_score"] or 0.0) < threshold]

    return {
        "mode": mode,
        "questions": len(per_question),
        "answerable": len(answerable),
        "hit_at_k": _mean(item["hit"] for item in answerable),
        "recall_at_k": _mean(item["recall"] for item in answerable),
        "mrr_at_k": _mean(item["rrr"] for item in answerable),
        "refusal_accuracy_unanswerable": (
            len(refused_unanswerable) / len(unanswerable) if unanswerable else 0.0
        ),
        "false_refusal_rate": (len(false_refusals) / len(answerable) if answerable else 0.0),
        "latency_ms": _percentiles(latencies),
        "per_question": per_question,
    }


def _question_record(
    question: EvalQuestion,
    resolved: dict[str, list[ResolvedGold]],
    result: RetrievalResult,
    mode: RetrievalMode,
    top_k: int,
) -> dict[str, Any]:
    gold = resolved.get(question.id, [])
    record: dict[str, Any] = {
        "id": question.id,
        "category": question.category,
        "question": question.question,
        "answerable": question.answerable,
        "top_score": top_score(result.hits[0] if result.hits else None, mode),
        "hits": [
            {
                "rank": hit.index,
                "path": hit.path,
                "heading": hit.heading,
                "lines": [hit.start_line, hit.end_line],
                "dense_rank": hit.dense_rank,
                "sparse_rank": hit.sparse_rank,
                "fused_rank": hit.fused_rank,
                "dense_score": hit.dense_score,
                "rerank_score": hit.rerank_score,
                "gold": hit_key(hit) in {item.key for item in gold},
            }
            for hit in result.hits[:top_k]
        ],
        "timings": {
            "embed_ms": result.timings.embed_ms,
            "dense_ms": result.timings.dense_ms,
            "sparse_ms": result.timings.sparse_ms,
            "rerank_ms": result.timings.rerank_ms,
            "total_ms": result.timings.total_ms,
        },
        "rerank_failed": result.rerank_failed,
    }
    if question.answerable:
        record.update(score_question(gold, result.hits, top_k=top_k))
    return record


def _mean(values: Any) -> float:
    items = [float(value) for value in values]
    return sum(items) / len(items) if items else 0.0


def _percentiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {"p50": 0.0, "p95": 0.0, "mean": 0.0}
    ordered = sorted(values)
    p95_index = min(len(ordered) - 1, round(0.95 * (len(ordered) - 1)))
    return {
        "p50": round(statistics.median(ordered), 2),
        "p95": round(ordered[p95_index], 2),
        "mean": round(statistics.fmean(ordered), 2),
    }


def calibrate_threshold(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Sweep a single score threshold, balancing false refusals and false answers."""
    scores = [
        (item["top_score"], item["answerable"]) for item in records if item["top_score"] is not None
    ]
    answerable_total = sum(1 for _, answerable in scores if answerable)
    unanswerable_total = sum(1 for _, answerable in scores if not answerable)
    sweep = []
    for step in range(1, 19):
        threshold = round(step * 0.05, 2)
        kept = sum(1 for score, answerable in scores if answerable and score >= threshold)
        refused = sum(1 for score, answerable in scores if not answerable and score < threshold)
        recall = kept / answerable_total if answerable_total else 0.0
        specificity = refused / unanswerable_total if unanswerable_total else 0.0
        sweep.append(
            {
                "threshold": threshold,
                "answer_recall": round(recall, 4),
                "refusal_rate": round(specificity, 4),
                "balanced_accuracy": round((recall + specificity) / 2, 4),
            }
        )
    best = max(
        sweep,
        key=lambda item: (item["balanced_accuracy"], -abs(item["threshold"] - 0.35)),
    )
    answerable_scores = [score for score, answerable in scores if answerable]
    unanswerable_scores = [score for score, answerable in scores if not answerable]
    min_answerable = min(answerable_scores) if answerable_scores else None
    return {
        "sweep": sweep,
        "best": best,
        # The gate must not refuse an answerable question, so it has to sit
        # below the weakest answerable score. The gap to that boundary is the
        # safety margin; see ADR-0009.
        "min_answerable_top_score": round(min_answerable, 4)
        if min_answerable is not None
        else None,
        "max_unanswerable_top_score": (
            round(max(unanswerable_scores), 4) if unanswerable_scores else None
        ),
        "unanswerables_gate_refused": [
            item["id"]
            for item in records
            if not item["answerable"] and (item["top_score"] or 0.0) < min_answerable
        ]
        if min_answerable is not None
        else [],
    }


async def run_generation(
    questions: list[EvalQuestion],
    resolved: dict[str, list[ResolvedGold]],
    accepted: dict[str, list[ResolvedGold]],
    embedder: OllamaEmbedder,
    store: QdrantStore,
    reranker: CrossEncoderReranker,
    settings: Settings,
    *,
    top_k: int,
) -> dict[str, Any]:
    """End-to-end evaluation: retrieval gate first, generation refusal second.

    Unanswerable questions are included on purpose: the gate only catches the
    clearly-irrelevant cases, so this phase measures whether the generation
    layer refuses the topically-related-but-insufficient ones.
    """
    provider = DeepSeekProvider(
        settings.deepseek_api_key, settings.deepseek_base_url, settings.deepseek_model
    )
    records: list[dict[str, Any]] = []
    ttfts: list[float] = []
    totals: list[float] = []

    for question in questions:
        result = await retrieve(
            embedder,
            store,
            reranker,
            question.question,
            mode="hybrid_rerank",
            top_k=top_k,
            top_n=settings.retrieval_top_n,
            rrf_k=settings.rrf_k,
        )
        gold_keys = {item.key for item in resolved.get(question.id, [])}
        accepted_keys = {item.key for item in accepted.get(question.id, [])}
        gold_indices = {hit.index for hit in result.hits if hit_key(hit) in gold_keys}
        accepted_indices = {hit.index for hit in result.hits if hit_key(hit) in accepted_keys}
        record: dict[str, Any] = {
            "id": question.id,
            "category": question.category,
            "answerable": question.answerable,
            "top_score": top_score(result.hits[0] if result.hits else None, "hybrid_rerank"),
            "gold_indices": sorted(gold_indices),
            "accepted_indices": sorted(accepted_indices),
            # Keep the retrieved chunks so a saved report can be re-scored
            # against updated annotations without another cloud call.
            "hits": [
                {
                    "index": hit.index,
                    "path": hit.path,
                    "heading": hit.heading,
                    "lines": [hit.start_line, hit.end_line],
                    "gold": hit_key(hit) in gold_keys,
                    "accepted": hit_key(hit) in accepted_keys,
                }
                for hit in result.hits
            ],
        }

        if should_refuse(
            result.hits,
            min_rerank_score=settings.retrieval_gate_rerank_score,
            min_dense_score=settings.retrieval_min_score,
        ):
            record.update(
                refused=True,
                refused_by="gate",
                false_refusal=question.answerable,
                false_answer=False,
            )
            records.append(record)
            continue

        started = time.perf_counter()
        first_token_at: float | None = None
        parts: list[str] = []
        error: str | None = None
        try:
            async for delta in provider.stream_chat(
                build_messages(question.question, result.hits, [])
            ):
                if first_token_at is None:
                    first_token_at = time.perf_counter()
                parts.append(delta)
        except ProviderError as exc:
            error = str(exc)
        total = (time.perf_counter() - started) * 1000
        answer = "".join(parts)
        cited = cited_indices(answer, len(result.hits))
        cited_gold = set(cited) & gold_indices
        cited_accepted = set(cited) & accepted_indices
        ttft = (first_token_at - started) * 1000 if first_token_at else None
        if ttft is not None:
            ttfts.append(ttft)
        totals.append(total)

        model_refused = is_refusal(answer)
        record.update(
            refused=model_refused,
            refused_by="generation" if model_refused else None,
            false_refusal=question.answerable and model_refused,
            false_answer=(not question.answerable) and not model_refused,
            error=error,
            answer=answer[:300],
            cited=cited,
            ttft_ms=round(ttft, 2) if ttft is not None else None,
            total_ms=round(total, 2),
        )
        if question.answerable and not model_refused and not error:
            # Precision is judged against gold + acceptable passages; the
            # gold-only figure is kept as a conservative lower bound.
            record["citation_precision"] = len(cited_accepted) / len(cited) if cited else 0.0
            record["citation_precision_strict"] = len(cited_gold) / len(cited) if cited else 0.0
            record["citation_recall"] = len(cited_gold) / len(gold_indices) if gold_indices else 0.0
        records.append(record)

    answered = [
        item
        for item in records
        if item["answerable"] and not item["refused"] and not item.get("error")
    ]
    unanswerable = [item for item in records if not item["answerable"]]
    return {
        "questions": len(records),
        "answerable": sum(1 for item in records if item["answerable"]),
        "generated": sum(1 for item in records if "ttft_ms" in item),
        "gate_refusals": sum(1 for item in records if item["refused_by"] == "gate"),
        "generation_refusals": sum(1 for item in records if item["refused_by"] == "generation"),
        "refused_unanswerable": sum(1 for item in unanswerable if item["refused"]),
        "unanswerable": len(unanswerable),
        "false_refusals": sum(1 for item in records if item["false_refusal"]),
        "false_answers": sum(1 for item in records if item["false_answer"]),
        "citation_precision": _mean(
            item["citation_precision"] for item in answered if "citation_precision" in item
        ),
        "citation_precision_strict": _mean(
            item["citation_precision_strict"]
            for item in answered
            if "citation_precision_strict" in item
        ),
        "citation_recall": _mean(
            item["citation_recall"] for item in answered if "citation_recall" in item
        ),
        "ttft_ms": _percentiles(ttfts),
        "total_ms": _percentiles(totals),
        "errors": [
            {"id": item["id"], "error": item["error"]} for item in records if item.get("error")
        ],
        "per_question": records,
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Stage 2 retrieval evaluation",
        "",
        f"- generated_at: {report['generated_at']}",
        f"- collection: `{report['collection']}`",
        f"- questions: {report['questions']} "
        f"({report['answerable']} answerable / {report['unanswerable']} unanswerable)",
        f"- top_k: {report['top_k']}",
        "",
        "## Retrieval modes",
        "",
        "| mode | Hit@k | Recall@k | MRR@k | 拒答率 | 误拒率 | p50 ms | p95 ms |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for mode, metrics in report["modes"].items():
        lines.append(
            f"| {mode} | {_pct(metrics['hit_at_k'])} | {_pct(metrics['recall_at_k'])} | "
            f"{metrics['mrr_at_k']:.3f} | {_pct(metrics['refusal_accuracy_unanswerable'])} | "
            f"{_pct(metrics['false_refusal_rate'])} | "
            f"{metrics['latency_ms']['p50']:.0f} | {metrics['latency_ms']['p95']:.0f} |"
        )

    lines += ["", "## Threshold sweep (hybrid_rerank)", ""]
    sweep = report["threshold_sweep"]
    lines += [
        "| threshold | answer recall | refusal rate | balanced acc |",
        "| --- | --- | --- | --- |",
    ]
    for row in sweep["sweep"]:
        lines.append(
            f"| {row['threshold']:.2f} | {_pct(row['answer_recall'])} | "
            f"{_pct(row['refusal_rate'])} | {_pct(row['balanced_accuracy'])} |"
        )
    lines += [
        "",
        "单阈值策略的代价：balanced accuracy 最优阈值会误拒有答案的问题，"
        "因为 rerank 对“同义改写 / 需要推理”的问题给分很低（见 q12、q29）。"
        "因此生产链路采用两层拒答：",
        "",
        f"- 闸门阈值 `retrieval_gate_rerank_score` = "
        f"**{report['thresholds']['hybrid_rerank']:.3f}**：只拦完全无关的检索结果；",
        "- 生成层拒答：通过闸门的片段交给模型，由系统 prompt 按“资料不足”规则判断。",
        "",
        f"最弱有答案题 top1 = {sweep['min_answerable_top_score']:.4f}。",
        "闸门之下可无风险拒掉的无答案题：",
        *[f"- {qid}" for qid in sweep["unanswerables_gate_refused"]],
    ]
    if not sweep["unanswerables_gate_refused"]:
        lines.append("- 无")

    generation = report.get("generation")
    if generation:
        lines += [
            "",
            "## Generation (cloud, hybrid_rerank)",
            "",
            f"- provider calls: {generation['generated']}/{generation['questions']}",
            f"- gate refusals: {generation['gate_refusals']}，"
            f"generation refusals: {generation['generation_refusals']}",
            f"- unanswerable refused: {generation['refused_unanswerable']}"
            f"/{generation['unanswerable']}",
            f"- true answers falsely refused: {generation['false_refusals']}"
            f"/{generation['answerable']}",
            f"- unanswerable falsely answered: {generation['false_answers']}",
            f"- citation precision (gold + acceptable): {_pct(generation['citation_precision'])}",
            f"- citation precision (gold only, strict): "
            f"{_pct(generation['citation_precision_strict'])}",
            f"- citation recall (gold-based): {_pct(generation['citation_recall'])}",
            f"- first token p50 / p95: "
            f"{generation['ttft_ms']['p50']:.0f} / {generation['ttft_ms']['p95']:.0f} ms",
            f"- total p50 / p95: "
            f"{generation['total_ms']['p50']:.0f} / {generation['total_ms']['p95']:.0f} ms",
        ]
        missed = [
            f"- {item['id']}: {item['answer'][:120]}"
            for item in generation["per_question"]
            if item.get("false_answer")
        ]
        if missed:
            lines += ["", "### 未被拒答的无答案题", *missed]

    lines += ["", "## Per-question retrieval failures", ""]
    gate = report["thresholds"]["hybrid_rerank"]
    failures = []
    for mode, metrics in report["modes"].items():
        for item in metrics["per_question"]:
            if item["answerable"] and not item["hit"]:
                failures.append(f"- {mode} {item['id']}: no gold chunk in top-{report['top_k']}")
            elif not item["answerable"] and (item["top_score"] or 0.0) >= (
                gate if mode == "hybrid_rerank" else report["thresholds"][mode]
            ):
                failures.append(
                    f"- {mode} {item['id']}: unanswerable, passes gate "
                    f"(top_score={item['top_score']:.3f})"
                )
    lines += failures or ["（无）"]
    return "\n".join(lines) + "\n"


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


async def run(args: argparse.Namespace) -> int:
    settings = get_settings()
    questions = load_questions(Path(args.questions))
    corpus_root = Path(args.corpus_root)
    chunks = chunk_corpus(corpus_root, settings)

    resolved: dict[str, list[ResolvedGold]] = {}
    accepted: dict[str, list[ResolvedGold]] = {}
    problems: list[str] = []
    for question in questions:
        if not question.answerable:
            continue
        items, issue = resolve_items(question.id, question.gold, chunks, label="gold")
        extras, extra_issue = resolve_items(
            question.id, question.acceptable, chunks, label="acceptable"
        )
        resolved[question.id] = items
        by_key = {item.key: item for item in [*items, *extras]}
        accepted[question.id] = list(by_key.values())
        problems.extend(issue)
        problems.extend(extra_issue)
    if problems:
        for problem in problems:
            print(f"annotation resolution failed: {problem}", file=sys.stderr)
        return 2

    answerable = sum(1 for question in questions if question.answerable)
    print(
        f"questions={len(questions)} answerable={answerable} "
        f"unanswerable={len(questions) - answerable}"
    )

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

    modes: list[RetrievalMode] = list(args.modes)
    report: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "collection": settings.qdrant_collection,
        "corpus_root": str(corpus_root),
        "top_k": args.top_k,
        "questions": len(questions),
        "answerable": answerable,
        "unanswerable": len(questions) - answerable,
        "thresholds": {
            "vector": settings.retrieval_min_score,
            "hybrid": settings.retrieval_min_score,
            "hybrid_rerank": settings.retrieval_gate_rerank_score,
        },
        "modes": {},
    }

    try:
        for mode in modes:
            print(f"evaluating {mode} ...", flush=True)
            report["modes"][mode] = await evaluate_mode(
                mode,
                questions,
                resolved,
                embedder,
                store,
                reranker,
                settings,
                top_k=args.top_k,
            )
            metrics = report["modes"][mode]
            print(
                f"  hit@{args.top_k}={metrics['hit_at_k']:.3f} "
                f"recall@{args.top_k}={metrics['recall_at_k']:.3f} "
                f"mrr@{args.top_k}={metrics['mrr_at_k']:.3f}",
                flush=True,
            )

        report["threshold_sweep"] = calibrate_threshold(
            report["modes"]["hybrid_rerank"]["per_question"]
        )

        if args.full:
            if not settings.has_cloud_key:
                print("--full requested but DEEPSEEK_API_KEY is empty", file=sys.stderr)
                return 2
            print("running cloud generation eval (snippets only) ...", flush=True)
            report["generation"] = await run_generation(
                questions,
                resolved,
                accepted,
                embedder,
                store,
                reranker,
                settings,
                top_k=args.top_k,
            )
    finally:
        await embedder.aclose()
        await store.close()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "stage2-report.json"
    markdown_path = out_dir / "stage2-report.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    print(f"wrote {json_path}")
    print(f"wrote {markdown_path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="LocalMind Stage 2 evaluation")
    parser.add_argument("--questions", default=str(BASE_DIR / "eval/stage2/questions.jsonl"))
    parser.add_argument("--corpus-root", default=str(BASE_DIR / "examples/corpus"))
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["vector", "hybrid", "hybrid_rerank"],
        choices=["vector", "hybrid", "hybrid_rerank"],
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--out", default=str(BASE_DIR / "docs/eval"))
    parser.add_argument(
        "--full",
        action="store_true",
        help="also run the cloud generation eval (sends retrieved snippets to DeepSeek)",
    )
    return asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
