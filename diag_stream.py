"""Network diagnostic: stream English + Spanish answers from the Jetson over fresh connections.

  python diag_stream.py [jetson-ip]
"""
import http.client, json, time, sys
host = sys.argv[1] if len(sys.argv) > 1 else "192.168.55.1"
tests = [("en", "What can you help me with?"), ("es", "¿Qué servicios ofrece la biblioteca?")] * 3
for lang, q in tests:
    body = json.dumps({"messages": [{"role": "user", "content": q}],
                       "max_tokens": 40, "stream": True}).encode("utf-8")
    c = http.client.HTTPConnection(host, 8080, timeout=60)
    t0, lines, done = time.time(), 0, False
    try:
        c.request("POST", "/v1/chat/completions", body, {"Content-Type": "application/json"})
        r = c.getresponse()
        for line in r:
            lines += 1
            done = done or line.strip() == b"data: [DONE]"
        print(f"{lang}: OK      {lines:3d} lines, [DONE]={done}, {time.time()-t0:.2f}s")
    except Exception as e:
        print(f"{lang}: FAILED  {lines:3d} lines, [DONE]={done}, {time.time()-t0:.2f}s  {e!r}")
    finally:
        c.close()
