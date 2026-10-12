"""Step 5 - Jetson LLM over the network: time to first token and tokens/s.

  python 05_llm.py --host 192.168.1.50            # Gemma on port 8080
  python 05_llm.py --host 192.168.1.50 --runs 5

On the Jetson, llama-server must listen on the network, not just localhost:
  llama-server ... --host 0.0.0.0 --port 8080
"""
import argparse, statistics, time
from openai import OpenAI

ap = argparse.ArgumentParser()
ap.add_argument("--host", required=True)
ap.add_argument("--port", type=int, default=8080)
ap.add_argument("--runs", type=int, default=3)
a = ap.parse_args()

# Connection: close -> a fresh connection per request. Reusing keep-alive connections
# to llama-server caused random "connection forcibly closed" errors on Windows.
client = OpenAI(base_url=f"http://{a.host}:{a.port}/v1", api_key="none", timeout=60,
                default_headers={"Connection": "close"})
SYSTEM = ("You are Skyy, a friendly college concierge. Answer in the same language the "
          "visitor uses, in one or two short sentences suitable for speaking aloud.")
PROMPTS = ["What can you help me with?", "¿Qué servicios ofrece la biblioteca?",
           "Quels sont les horaires de la bibliothèque ?"]

model = client.models.list().data[0].id
print(f"Connected to {a.host}:{a.port}, model: {model}\n")

for prompt in PROMPTS:
    ttfts, rates = [], []
    for i in range(a.runs):
        t0 = time.perf_counter()
        first, n, text, thinking = None, 0, "", 0
        stream = client.chat.completions.create(
            model=model, stream=True, max_tokens=120, temperature=0.3,
            messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}])
        for chunk in stream:
            if not chunk.choices:
                continue
            d = chunk.choices[0].delta
            if getattr(d, "reasoning_content", None):
                thinking += 1  # model is "thinking" before answering: adds delay
            delta = d.content or ""
            if delta:
                if first is None:
                    first = time.perf_counter() - t0
                n += 1
                text += delta
        total = time.perf_counter() - t0
        ttfts.append(first or total)
        if first and total > first and n > 1:
            rates.append((n - 1) / (total - first))
    print(f"Q: {prompt}\nA: {text.strip()}")
    print(f"   first token {statistics.median(ttfts)*1000:.0f} ms (median of {a.runs}), "
          f"~{statistics.median(rates) if rates else 0:.0f} tok/s\n")
    if thinking:
        print(f"   WARNING: {thinking} 'thinking' chunks before the answer. Turn thinking off "
              "for voice (see README) or the first word will be slow.\n")
print("Note: tok/s counts streamed chunks, which is close to tokens for llama-server.")
