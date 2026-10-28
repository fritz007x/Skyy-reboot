"""Knowledge base for Skyy: the MDC FAQ (mdc_faq_kb_multilingual.jsonl), as RAG or as one cached prompt.

Each FAQ is in the file three times (en, es, fr), with the same faq_id. In rag mode all
three are indexed, so a Spanish or French question finds the entry in its own language.
In prompt mode only the English entries go into the system prompt.

SKYY_KB_MODE picks how the LLM gets the FAQ:

  rag     Each question is embedded by EmbeddingGemma on the Jetson, Qdrant
          returns the closest FAQ entries (one per faq_id), and they are added to that question before
          it goes to the LLM. Costs per turn: one embedding call, one Qdrant search
          (tens of ms together) and ~70 extra prompt tokens per entry for the LLM.
  prompt  (default) The whole English FAQ (~4k tokens) goes into the system prompt. llama-server keeps
          the processed system prompt in its cache, so each turn only processes the new
          question: no retrieval round trips and no extra prompt tokens per turn. Usually
          the fastest option while the FAQ is this small. Compare with 09_kb_latency.py.
  off     No knowledge base.

Build the Qdrant collection once with 08_kb_index.py (needed for rag mode only).

KnowledgeInjector (between HistoryCap and the LLM) does the rag-mode retrieval.
warm_up() sends the system prompt to llama-server at startup, so the first visitor
doesn't wait for it to be processed.
"""
import json
import os
import re
import time
import uuid
from pathlib import Path

import httpx
from loguru import logger

from pipecat.frames.frames import Frame, LLMContextFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from skyy_processors import FILLERS

KB_PATH = Path(os.environ.get("SKYY_KB", Path(__file__).resolve().parent / "mdc_faq_kb_multilingual.jsonl"))
KB_MODE = os.environ.get("SKYY_KB_MODE", "prompt").lower()
COLLECTION = os.environ.get("SKYY_KB_COLLECTION", "mdc_faq")
EMBED_PORT = int(os.environ.get("SKYY_EMBED_PORT", "8082"))
QDRANT_PORT = int(os.environ.get("SKYY_QDRANT_PORT", "6333"))
QDRANT_KEY = os.environ.get("SKYY_QDRANT_KEY", "")
TOP_K = int(os.environ.get("SKYY_KB_TOP_K", "3"))
MIN_SCORE = float(os.environ.get("SKYY_KB_MIN_SCORE", "0"))  # calibrate with 08_kb_index.py --test

# Added to the system prompt in rag and prompt mode. The notes can be in another language
# than the visitor's, so remind the model that they are not a language signal (see the
# LANGUAGE RULE in the system prompt).
KB_RULES = (" Answer college questions using only the FAQ notes you are given. If the notes "
            "don't cover the question, say you don't have that information and suggest the "
            "College Information Center at 305-237-8888. The FAQ notes may be in a different language "
            "than the visitor's; they are not a language signal, so still reply in the language "
            "of the visitor's message.")

# EmbeddingGemma task prefixes (see README, Part 1 step 6)
QUERY_PREFIX = "task: search result | query: "


def load_kb(path=KB_PATH) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def faq_id(e: dict) -> str:
    return e.get("faq_id", e["id"])   # the older English-only file has no faq_id


def note(e: dict) -> str:
    return " ".join(e["text"].split())   # one line per note ("question\nanswer" in the file)


def doc_text(e: dict) -> str:
    """What gets embedded for an entry: its question (also in Spanish if present) and summary."""
    title = e["question"] + (f" / {e['question_es']}" if e.get("question_es") else "")
    return f"title: {title} | text: {e['text']}"


def point_id(e: dict) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, e["id"]))   # stable, so re-indexing overwrites


def kb_prompt(entries: list[dict]) -> str:
    """The whole English FAQ as one block for the system prompt (prompt mode)."""
    return "\n\nFAQ notes:\n" + "\n".join(f"- {note(e)}" for e in entries
                                         if e.get("lang", "en") == "en")


def system_prompt(base: str, mode: str = KB_MODE) -> str:
    if mode == "prompt":
        return base + KB_RULES + kb_prompt(load_kb())
    if mode == "rag":
        return base + KB_RULES
    return base


def query_text(text: str, strip_words=()) -> str:
    """Drop a leading wake word and fillers ("Hey Sky, ...") so they don't steer the search."""
    skip = FILLERS | {w.lower() for w in strip_words}
    words = text.split()
    while words and re.sub(r"[^\w']", "", words[0].lower()) in skip:
        words = words[1:]
    return " ".join(words)


class KnowledgeBase:
    """Async EmbeddingGemma + Qdrant client for the Jetson."""

    def __init__(self, host: str, timeout: float = 2.0):
        # Connection: close -> fresh connection per request; keep-alive connections over
        # the USB-C link caused random resets on Windows (see 05_llm.py)
        headers = {"Connection": "close"}
        self._embed = httpx.AsyncClient(base_url=f"http://{host}:{EMBED_PORT}", timeout=timeout,
                                        headers=headers)
        self._qdrant = httpx.AsyncClient(base_url=f"http://{host}:{QDRANT_PORT}", timeout=timeout,
                                         headers={**headers, "api-key": QDRANT_KEY})

    async def embed(self, texts: list[str]) -> list[list[float]]:
        r = await self._embed.post("/v1/embeddings", json={"input": texts})
        r.raise_for_status()
        return [d["embedding"] for d in sorted(r.json()["data"], key=lambda d: d["index"])]

    async def search(self, question: str, k: int = TOP_K, min_score: float = MIN_SCORE):
        """Returns (hits, embed_ms, search_ms); each hit is the entry's payload plus its score.

        Asks Qdrant for 3*k points, because one FAQ can match in several languages,
        and keeps the best-scoring language of each FAQ.
        """
        t0 = time.perf_counter()
        [vector] = await self.embed([QUERY_PREFIX + question])
        t1 = time.perf_counter()
        body = {"query": vector, "limit": 3 * k, "with_payload": True}
        if min_score:
            body["score_threshold"] = min_score
        r = await self._qdrant.post(f"/collections/{COLLECTION}/points/query", json=body)
        r.raise_for_status()
        t2 = time.perf_counter()
        hits, seen = [], set()
        for p in r.json()["result"]["points"]:   # sorted by score, best first
            fid = p["payload"].get("faq_id", p["payload"]["kb_id"])
            if fid not in seen and len(hits) < k:
                seen.add(fid)
                hits.append({**p["payload"], "score": p["score"]})
        return hits, (t1 - t0) * 1000, (t2 - t1) * 1000

    async def aclose(self):
        await self._embed.aclose()
        await self._qdrant.aclose()


def with_notes(question: str, hits: list[dict]) -> str:
    """The user message the LLM sees in rag mode: FAQ notes first, the visitor's words last."""
    notes = "\n".join(f"- {note(h)}" for h in hits)
    return f"FAQ notes:\n{notes}\n\nVisitor: {question}"


def without_notes(content: str) -> str:
    """Undo with_notes(): only the visitor's words."""
    if content.startswith("FAQ notes:") and "\n\nVisitor: " in content:
        return content.rsplit("\n\nVisitor: ", 1)[1]
    return content


class KnowledgeInjector(FrameProcessor):
    """rag mode: adds the closest FAQ entries to the visitor's latest message.

    Only the latest message keeps its notes; earlier questions go back to the visitor's
    words. Removing the previous question's notes changes that message, so llama-server
    reprocesses the history from there on every turn anyway. Without old notes that
    is a few hundred tokens instead of 2-3k (~1 s on the Jetson). Follow-up questions
    still see Skyy's earlier answers, just not the FAQ text behind them.
    If the Jetson is slow or down, the question goes through without notes.
    """

    def __init__(self, kb: KnowledgeBase, strip_words=(), min_words: int = 2, **kwargs):
        super().__init__(**kwargs)
        self._kb = kb
        self._strip = strip_words
        self._min_words = min_words   # "Sky." / "Hola Sky" -> greeting, no search needed

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, LLMContextFrame):
            await self._add_notes(frame.context)
        await self.push_frame(frame, direction)

    async def _add_notes(self, context):
        msgs = list(context.messages)
        i = len(msgs) - 1
        if i < 0 or not isinstance(msgs[i], dict) or msgs[i].get("role") != "user":
            return
        old = [j for j, m in enumerate(msgs[:i]) if isinstance(m, dict) and m.get("role") == "user"
               and isinstance(m.get("content"), str) and m["content"].startswith("FAQ notes:")]
        for j in old:   # notes of earlier questions: keep only the visitor's words
            msgs[j] = {**msgs[j], "content": without_notes(msgs[j]["content"])}
        if old:
            context.set_messages(msgs)
        content = msgs[i].get("content")
        if not isinstance(content, str) or content.startswith("FAQ notes:"):
            return   # not plain text, or already done (e.g. a speculative run of this turn)
        question = query_text(content, self._strip)
        if len(question.split()) < self._min_words:
            return
        try:
            hits, embed_ms, search_ms = await self._kb.search(question)
        except Exception as e:
            logger.warning(f"KB search failed, answering without notes: {e!r}")
            return
        logger.info(f"KB: {len(hits)} notes in {embed_ms + search_ms:.0f} ms "
                    f"(embed {embed_ms:.0f}, search {search_ms:.0f}): "
                    + ", ".join(f"{h['kb_id']} {h['score']:.2f}" for h in hits))
        if hits:
            msgs[i] = {**msgs[i], "content": with_notes(content, hits)}
            context.set_messages(msgs)

    async def cleanup(self):
        await self._kb.aclose()
        await super().cleanup()


# Extra request fields for llama-server. Gemma 4 otherwise writes an empty thought block
# ("<|channel>thought<channel|>", 3 tokens, ~0.13 s on the Orin) before every answer,
# even with --reasoning-budget 0. Pass as extra={"extra_body": NO_THINKING} in Pipecat.
# It also removes "<|think|>" from the start of the system prompt, so every request that
# should reuse the cached prompt (warm_up() included) must send it.
NO_THINKING = {"chat_template_kwargs": {"enable_thinking": False}}


def warm_up(host: str, port: int, system: str, timeout: float = 120.0):
    """Process the system prompt on llama-server once, before the first visitor.

    llama-server reuses the processed start of the previous prompt, so later requests
    with the same system prompt skip it. In prompt mode that is ~4k tokens of FAQ.
    """
    t0 = time.perf_counter()
    try:
        r = httpx.post(f"http://{host}:{port}/v1/chat/completions",
                       timeout=httpx.Timeout(timeout, connect=3.0),   # don't hang if the Jetson is off
                       headers={"Connection": "close"}, json={
                           "messages": [{"role": "system", "content": system},
                                        {"role": "user", "content": "Hello"}],
                           "max_tokens": 1, "cache_prompt": True,
                           # same template options as the voice loop, or the cached
                           # prompt differs from the first token on ("<|think|>")
                           **NO_THINKING})
        r.raise_for_status()
        n = r.json().get("timings", {}).get("prompt_n", "?")
        logger.info(f"LLM warmed up: {n} prompt tokens cached in {time.perf_counter() - t0:.1f} s")
    except Exception as e:
        logger.warning(f"LLM warm-up failed (first answer will be slower): {e!r}")
