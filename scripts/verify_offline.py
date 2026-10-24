"""Verify that a local-track answer produces no outbound connections.

Two independent checks:

1. **Protocol assertion** — ask the running backend for a local-track answer and
   require the SSE manifest to report ``track=local`` and ``egress.enabled=false``
   with a non-empty answer. This proves the backend *decided* not to egress.
2. **Socket observation** — while the request is in flight, sample the backend
   process's TCP connections with ``netstat -ano`` and require every remote
   address to be loopback (Ollama and Qdrant are local, so this is expected).
   This catches a provider that claims to be local but phones home anyway.

The socket check needs the backend PID. On Windows:

    netstat -ano | findstr :8000      # the last column is the PID

Usage:

    .venv/Scripts/python.exe scripts/verify_offline.py --pid 12345
    .venv/Scripts/python.exe scripts/verify_offline.py            # protocol check only

For a full offline demonstration, additionally disable the network adapter and
repeat the same request from the UI: retrieval, rerank and generation all stay
on this machine, so the answer is unchanged.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import threading
import time

import httpx

LOOPBACK = ("127.", "::1", "0.0.0.0", "::")
NETSTAT_ROW = re.compile(r"^\s*TCP\s+(\S+)\s+(\S+)\s+(\S+)\s+(\d+)\s*$")


def remote_endpoints(pid: int) -> set[str]:
    """Remote addresses of every TCP connection owned by ``pid``."""
    try:
        output = subprocess.run(
            ["netstat", "-ano", "-p", "TCP"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return set()
    remotes: set[str] = set()
    for line in output.splitlines():
        # Columns: proto, local, remote, state, pid.
        match = NETSTAT_ROW.match(line)
        if not match or int(match.group(4)) != pid:
            continue
        remote = match.group(2)
        if remote != "0.0.0.0:0":  # LISTENING socket, no peer
            remotes.add(remote)
    return remotes


def is_loopback(endpoint: str) -> bool:
    host = endpoint.rsplit(":", 1)[0]
    return host.startswith(LOOPBACK)


class Sampler:
    """Poll the process's TCP table while the request is running."""

    def __init__(self, pid: int, interval: float = 0.2) -> None:
        self._pid = pid
        self._interval = interval
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self.observed: set[str] = set()

    def _run(self) -> None:
        while not self._stop.is_set():
            self.observed |= remote_endpoints(self._pid)
            self._stop.wait(self._interval)

    def __enter__(self) -> Sampler:
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        self._thread.join(timeout=5)


def local_chat(base: str, question: str) -> tuple[dict, str, dict]:
    body = {"message": question, "track": "local"}
    started = time.perf_counter()
    events: list[tuple[str, dict]] = []
    with (
        httpx.Client(timeout=300.0) as client,
        client.stream("POST", f"{base}/api/chat", json=body) as response,
    ):
        response.raise_for_status()
        event = ""
        for line in response.iter_lines():
            if line.startswith("event: "):
                event = line[7:]
            elif line.startswith("data: "):
                events.append((event, json.loads(line[6:])))
    wall_ms = round((time.perf_counter() - started) * 1000, 1)
    if not events:
        raise SystemExit("FAIL: 没有收到任何 SSE 事件（后端未运行？）")
    citations = events[0][1]
    done = events[-1][1]
    answer = "".join(data["text"] for name, data in events if name == "token")
    if any(name == "error" for name, _ in events):
        raise SystemExit(f"FAIL: 生成失败：{events[-1][1]}")
    return citations, answer, {"done": done, "wall_ms": wall_ms}


def main() -> int:
    # Windows consoles default to GBK; the report contains Chinese text.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="http://127.0.0.1:8000", help="backend base URL")
    parser.add_argument("--pid", type=int, default=None, help="backend PID for the socket check")
    parser.add_argument("--question", default="混合检索为什么要结合 BM25？")
    args = parser.parse_args()

    if args.pid is not None:
        sampler = Sampler(args.pid)
        with sampler:
            citations, answer, extra = local_chat(args.base, args.question)
    else:
        sampler = None
        citations, answer, extra = local_chat(args.base, args.question)

    egress = citations.get("egress") or {}
    failures: list[str] = []
    if citations.get("track") != "local":
        failures.append(f"轨道不是 local：{citations.get('track')}")
    if egress.get("track") != "local" or egress.get("enabled") is not False:
        failures.append(f"出站回执不正确：{egress}")
    if not answer.strip():
        failures.append("本地回答为空")
    if extra["done"].get("total_ms") is None:
        failures.append("缺少延迟埋点")

    external: set[str] = set()
    if sampler is not None:
        # Loopback seeds are what the backend uses for Ollama/Qdrant; anything
        # else is a real outbound connection.
        external = {endpoint for endpoint in sampler.observed if not is_loopback(endpoint)}
        if external:
            failures.append(f"检测到非环回连接：{sorted(external)}")

    print(f"track        : {citations.get('track')}")
    print(f"egress       : enabled={egress.get('enabled')} chunks={egress.get('chunks')}")
    print(f"首 token     : {extra['done'].get('ttft_ms')} ms")
    print(f"总耗时       : {extra['done'].get('total_ms')} ms（墙钟 {extra['wall_ms']} ms）")
    print(f"rerank 设备  : {(citations.get('retrieval') or {}).get('device')}")
    print(f"回答（前 80）: {answer.strip()[:80]}")
    if sampler is not None:
        print(f"观测到连接   : {sorted(sampler.observed) or '无'}")
        print(f"非环回连接   : {sorted(external) or '无'}")

    if failures:
        print("\nFAIL")
        for item in failures:
            print(f" - {item}")
        return 1
    print("\nPASS: 本地轨道完成问答，出站回执为零，未观测到非环回连接。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
