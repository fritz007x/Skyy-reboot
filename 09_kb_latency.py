"""Step 9 - which knowledge-base mode answers fastest? (off / prompt / rag)

For each mode, asks the same questions and measures the time from "question ready"
to the LLM's first token, including the embedding + Qdrant search in rag mode.
"prompt tok" is how many prompt tokens llama-server actually had to process: in
prompt mode it should be small, because the FAQ in the system prompt is cached.

  $env:SKYY_QDRANT_KEY = "<the Qdrant API key>"
  python 09_kb_latency.py --host 192.168.55.1
  python 09_kb_latency.py --host 192.168.55.1 --modes prompt rag --runs 5

Needs the main LLM (:8080), and for rag also EmbeddingGemma (:8082) and the Qdrant
collection from 08_kb_index.py.
"""
import argparse, asyncio, statistics, time

from openai import AsyncOpenAI

import kb
from skyy_prompt import SYSTEM

QUESTIONS = [
    "How do I get my official transcript?",
    "¿Cuántos créditos necesito para ser estudiante de tiempo completo?",
    "Comment puis-je demander une aide financière ?",
    "Can I drop a class over the phone?",
    "¿Cuándo es la graduación?",
]

ap = argparse.ArgumentParser()
ap.add_argument("--host", required=True, help="Jetson IP")
ap.add_argument("--port", type=int, default=8080)
ap.add_argument("--modes", nargs="+", default=["off", "prompt", "rag"], choices=["off", "prompt", "rag"])
ap.add_argument("--runs", type=int, default=3, help="timed runs per question")
ap.add_argument("--show", action="store_true", help="print the answers")
a = ap.parse_args()

llm = AsyncOpenAI(base_url=f"http://{a.host}:{a.port}/v1", api_key="none", timeout=120,
                  default_headers={"Connection": "close"})


async def first_token(system: str, user: str):
    """Returns (seconds to first token, prompt tokens processed, answer)."""
    t0 = time.perf_counter()
    first, text, prompt_n = None, "", None
    stream = await llm.chat.completions.create(
        model="gemma", stream=True, max_tokens=80, temperature=0.3,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
    async for chunk in stream:
        timings = (chunk.model_extra or {}).get("timings")
        if timings:
            prompt_n = timings.get("prompt_n")
        if chunk.choices and chunk.choices[0].delta.content:
            first = first or time.perf_counter() - t0
            text += chunk.choices[0].delta.content
    return first or time.perf_counter() - t0, prompt_n, text.strip()


async def run_mode(mode: str, base: kb.KnowledgeBase):
    system = kb.system_prompt(SYSTEM, mode)
    await first_token(system, "Hello")   # caches the system prompt, like warm_up() in step 7
    if mode == "rag":
        await base.search(QUESTIONS[0])
    totals, retrievals = [], []
    print(f"=== {mode} ===")
    for q in QUESTIONS:
        for _ in range(a.runs):
            user, r_ms = q, 0.0
            if mode == "rag":
                hits, embed_ms, search_ms = await base.search(q)
                user, r_ms = kb.with_notes(q, hits), embed_ms + search_ms
            ttft, prompt_n, answer = await first_token(system, user)
            totals.append(r_ms / 1000 + ttft)
            retrievals.append(r_ms)
        extra = f"retrieval {r_ms:.0f} ms + " if mode == "rag" else ""
        print(f"  {extra}first token {ttft * 1000:.0f} ms, prompt tok {prompt_n}  | {q}")
        if a.show:
            print(f"     -> {answer}")
    med = statistics.median(totals) * 1000
    print(f"  median question -> first token: {med:.0f} ms"
          + (f" (retrieval median {statistics.median(retrievals):.0f} ms)" if mode == "rag" else "")
          + "\n")
    return med


async def main():
    base = kb.KnowledgeBase(a.host)
    results = {m: await run_mode(m, base) for m in a.modes}
    await base.aclose()
    print("Summary: " + ", ".join(f"{m} {ms:.0f} ms" for m, ms in results.items()))
    with_kb = {m: ms for m, ms in results.items() if m != "off"}
    if with_kb:
        print(f"Fastest with the knowledge base: {min(with_kb, key=with_kb.get)} "
              "('off' has no knowledge base; it is only the floor to compare against)")


asyncio.run(main())
