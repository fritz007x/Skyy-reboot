"""Step 8 - load the MDC FAQ into Qdrant on the Jetson (needed for SKYY_KB_MODE=rag).

Embeds every entry with EmbeddingGemma (llama-server :8082), creates the collection
(cosine, size taken from the model) and stores the entries. Then runs test searches:
time for the embedding and the search, the entries found and their scores.

  $env:SKYY_QDRANT_KEY = "<the Qdrant API key>"
  python 08_kb_index.py --host 192.168.55.1           # (re)build the collection, then test
  python 08_kb_index.py --host 192.168.55.1 --test    # only test searches

Use the scores to choose SKYY_KB_MIN_SCORE: the off-topic questions at the end should
score below it, the FAQ questions above it.
"""
import argparse, asyncio, statistics

import httpx

import kb

TESTS = [
    "How do I get my official transcript?",
    "¿Cuántos créditos necesito para ser estudiante de tiempo completo?",
    "Comment puis-je demander une aide financière ?",
    "Can I drop a class over the phone?",
    "¿Cuándo es la graduación?",
    "Do you have free English classes?",
    # off-topic: should score low
    "What's the weather like today?",
    "Where can I park my car?",
]

ap = argparse.ArgumentParser()
ap.add_argument("--host", required=True, help="Jetson IP")
ap.add_argument("--test", action="store_true", help="skip indexing, only run test searches")
ap.add_argument("--runs", type=int, default=3, help="timed searches per test question")
a = ap.parse_args()


def build_index():
    entries = kb.load_kb()
    print(f"{len(entries)} entries from {kb.KB_PATH}")
    headers = {"Connection": "close"}
    r = httpx.post(f"http://{a.host}:{kb.EMBED_PORT}/v1/embeddings", headers=headers, timeout=120,
                   json={"input": [kb.doc_text(e) for e in entries]})
    r.raise_for_status()
    vectors = [d["embedding"] for d in sorted(r.json()["data"], key=lambda d: d["index"])]
    print(f"Embedded with EmbeddingGemma: {len(vectors[0])} dimensions")

    q = httpx.Client(base_url=f"http://{a.host}:{kb.QDRANT_PORT}", timeout=30,
                     headers={**headers, "api-key": kb.QDRANT_KEY})
    q.delete(f"/collections/{kb.COLLECTION}")   # start clean; 404 if it didn't exist
    q.put(f"/collections/{kb.COLLECTION}", json={
        "vectors": {"size": len(vectors[0]), "distance": "Cosine"}}).raise_for_status()
    q.put(f"/collections/{kb.COLLECTION}/points?wait=true", json={"points": [
        {"id": kb.point_id(e), "vector": v,
         "payload": {"kb_id": e["id"], "faq_id": kb.faq_id(e), "lang": e.get("lang", "en"),
                     "category": e["category"], "question": e["question"],
                     "text": e["text"], "source_url": e["source_url"]}}
        for e, v in zip(entries, vectors)]}).raise_for_status()
    n = q.post(f"/collections/{kb.COLLECTION}/points/count", json={"exact": True}).json()["result"]["count"]
    print(f"Qdrant collection '{kb.COLLECTION}': {n} points\n")


async def test():
    base = kb.KnowledgeBase(a.host)
    await base.search(TESTS[0])   # first call loads things on the Jetson; don't time it
    for question in TESTS:
        times = []
        for _ in range(a.runs):
            hits, embed_ms, search_ms = await base.search(question, min_score=0)
            times.append((embed_ms, search_ms))
        e = statistics.median(t[0] for t in times)
        s = statistics.median(t[1] for t in times)
        print(f"Q: {question}\n   embed {e:.0f} ms + search {s:.0f} ms = {e + s:.0f} ms (median of {a.runs})")
        for h in hits:
            print(f"   {h['score']:.3f}  {h['kb_id']:<13} {h['question']}")
        print()
    await base.aclose()


if not kb.QDRANT_KEY:
    print("Note: SKYY_QDRANT_KEY is not set; Qdrant will answer 401 if it uses an API key.\n")
if not a.test:
    build_index()
asyncio.run(test())
