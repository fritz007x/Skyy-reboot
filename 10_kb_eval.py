"""Step 10 - how good is the knowledge base? Retrieval and answer quality.

Uses the test set kb_eval_questions.jsonl: each of the 50 FAQs asked in a visitor's own
words in English, Spanish and French, misheard versions ("corner rule" for Gordon Rule),
and off-topic questions the FAQ doesn't cover. Each question lists key facts the
answer must contain (one alternative per group, e.g. ["5", "five", "cinco", "cinq"]).

Part A, retrieval (EmbeddingGemma + Qdrant, no LLM): is the right FAQ the top hit
(hit@1), in the top 3 (hit@3), and MRR; per language and per kind. Then a threshold
sweep over the top score of FAQ vs off-topic questions, to choose SKYY_KB_MIN_SCORE.

Part B, answers (Gemma, per mode): rag = notes from Qdrant + the question, prompt =
whole FAQ in the cached system prompt. Each answer is checked for the key facts, the
visitor's language, and for off-topic questions that it declines and gives the College
Information Center number. A judge then grades it against the FAQ entry (correct 0-2,
grounded): Gemma by default (grading its own answers, so lenient), or Gemini with
--judge gemini. The fact score is the objective one.

Judge test (--judge-test): how good are the judges? Takes answers from an earlier run
that have every key fact, and plants known errors in copies of them: a changed number,
another FAQ's answer, "I don't have that information", an invented sentence. Unchanged
answers and the same answer in another language must pass. Reports, per judge, how
many planted errors it caught and how many good answers it wrongly marked down, and
where two judges disagree. No hand labels needed.

  $env:SKYY_QDRANT_KEY = "<the Qdrant API key>"
  python 10_kb_eval.py --host 192.168.55.1                    # everything (~25 min)
  python 10_kb_eval.py --host 192.168.55.1 --retrieval-only   # Part A only (~1 min)
  python 10_kb_eval.py --host 192.168.55.1 --modes rag --limit 20 --judge none
  python 10_kb_eval.py --host 192.168.55.1 --judge gemini     # needs $env:GEMINI_API_KEY
  python 10_kb_eval.py --host 192.168.55.1 --judge-test eval_results\\kb_eval_<...>.jsonl --judges gemma gemini-2.5-flash gemini-3.5-flash

Privacy: --judge gemini / --judges gemini send the test questions, Skyy's answers and
FAQ entries to Google. Fine for this written test set and the public FAQ; don't use it
on questions from real visitors. The Gemma judge stays on the Jetson.

Needs the Qdrant collection from 08_kb_index.py, EmbeddingGemma (:8082) and the main
LLM (:8080). Per-question results go to eval_results/kb_eval_<date>_<time>.jsonl.
"""
import argparse, asyncio, json, os, re, statistics, time, unicodedata, zlib
from datetime import datetime
from pathlib import Path

import httpx
from openai import AsyncOpenAI

import kb
from bilingual_tts import detect_language
from skyy_prompt import SYSTEM

HERE = Path(__file__).resolve().parent
PHONE = "305-237-8888"   # College Information Center: what kb.KB_RULES says to offer when unsure

ap = argparse.ArgumentParser()
ap.add_argument("--host", required=True, help="Jetson IP")
ap.add_argument("--port", type=int, default=8080)
ap.add_argument("--questions", default=str(HERE / "kb_eval_questions.jsonl"))
ap.add_argument("--modes", nargs="+", default=["rag", "prompt"], choices=["rag", "prompt"])
ap.add_argument("--retrieval-only", action="store_true", help="skip Part B (answers)")
ap.add_argument("--judge", default="gemma", choices=["gemma", "gemini", "none"],
                help="who grades the answers (gemini sends them to Google, see Privacy above)")
ap.add_argument("--no-judge", action="store_true", help="same as --judge none")
ap.add_argument("--gemini-model", default=os.environ.get("SKYY_GEMINI_MODEL", "gemini-3.5-flash"),
                help="Gemini model for --judge gemini / --judges gemini")
ap.add_argument("--judge-test", metavar="RESULTS",
                help="planted-error test of the judges, on a results .jsonl from an earlier run")
ap.add_argument("--judges", nargs="+", default=["gemma"],
                help="judges to compare in --judge-test: gemma, gemini (= --gemini-model) "
                     "or a Gemini model name, e.g. gemma gemini-2.5-flash gemini-3.5-flash")
ap.add_argument("--judge-test-n", type=int, default=30, help="good answers to plant errors in")
ap.add_argument("--limit", type=int, help="use only N questions, spread over the whole set")
ap.add_argument("--min-score", type=float,
                help="rag threshold for Part B (default: the one Part A recommends)")
ap.add_argument("--top-k", type=int, default=kb.TOP_K, help="notes per question in rag mode")
a = ap.parse_args()

llm = AsyncOpenAI(base_url=f"http://{a.host}:{a.port}/v1", api_key="none", timeout=120,
                  default_headers={"Connection": "close"})
report: list[str] = []

# Gemini's OpenAI-compatible endpoint: the same client library, no extra package
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"


def make_judge(name: str) -> dict:
    """A judge = an OpenAI-compatible client, a model and request options."""
    if name == "gemma":
        return {"name": "gemma", "client": llm, "model": "gemma",
                "kw": {"max_tokens": 160, "extra_body": kb.NO_THINKING}}
    if not name.startswith("gemini"):
        raise SystemExit(f"Unknown judge '{name}': use gemma, gemini or a gemini-* model name.")
    model = a.gemini_model if name == "gemini" else name
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise SystemExit("Set $env:GEMINI_API_KEY to use the Gemini judge "
                         "(create a key at https://aistudio.google.com/apikey).")
    client = AsyncOpenAI(base_url=GEMINI_URL, api_key=key, timeout=60, max_retries=6)
    # Gemini 2.5 thinks before answering and counts that against max_tokens: leave room
    return {"name": model, "client": client, "model": model,
            "kw": {"max_tokens": 4096, "reasoning_effort": "low"}}


def say(line: str = ""):
    print(line)
    report.append(line)


def norm(s: str) -> str:
    """Lower case, no accents, plain apostrophes: "Démarche’s" -> "demarche's"."""
    s = unicodedata.normalize("NFKD", s.lower().replace("’", "'"))
    return re.sub(r"\s+", " ", "".join(c for c in s if not unicodedata.combining(c)))


def has(text: str, alt: str) -> bool:
    """`alt` appears in `text` starting at a word boundary ("math" matches "mathematics")."""
    return re.search(r"(?<!\w)" + re.escape(norm(alt)), norm(text)) is not None


def fact_score(answer: str, facts) -> float:
    return sum(any(has(answer, alt) for alt in group) for group in facts) / len(facts) if facts else 1.0


def pct(xs) -> str:
    xs = list(xs)
    return f"{100 * sum(xs) / len(xs):5.1f}%" if xs else "   - "


# ---------------------------------------------------------------- Part A: retrieval

async def retrieval(base: kb.KnowledgeBase, qs):
    await base.search(qs[0]["question"])   # first call loads things on the Jetson
    for q in qs:
        hits, embed_ms, search_ms = await base.search(q["question"], k=5, min_score=0)
        q["hits"] = [{"faq_id": h.get("faq_id", h["kb_id"]), "lang": h.get("lang"),
                      "score": round(h["score"], 4)} for h in hits]
        q["retrieval_ms"] = embed_ms + search_ms
        ids = [h["faq_id"] for h in q["hits"]]
        q["rank"] = ids.index(q["faq_id"]) + 1 if q["faq_id"] in ids else None
        q["top_score"] = q["hits"][0]["score"] if q["hits"] else 0.0

    faq = [q for q in qs if q["faq_id"]]
    off = [q for q in qs if not q["faq_id"]]
    say("=== Part A: retrieval (EmbeddingGemma + Qdrant) ===")
    say(f"{'':16} {'n':>4} {'hit@1':>7} {'hit@3':>7} {'MRR':>6}  same-language top hit")
    groups = [("all", faq)] + [(f"lang {l}", [q for q in faq if q["lang"] == l]) for l in ("en", "es", "fr")] \
        + [(f"kind {k}", [q for q in faq if q["kind"] == k]) for k in ("paraphrase", "asr")]
    for name, g in groups:
        if not g:
            continue
        mrr = statistics.mean(1 / q["rank"] if q["rank"] else 0 for q in g)
        say(f"{name:16} {len(g):4} {pct(q['rank'] == 1 for q in g):>7} "
            f"{pct((q['rank'] or 9) <= 3 for q in g):>7} {mrr:6.3f}  "
            f"{pct(q['hits'] and q['hits'][0]['lang'] == q['lang'] for q in g)}")
    say(f"median embed + search: {statistics.median(q['retrieval_ms'] for q in qs):.0f} ms")

    # Threshold: keep FAQ questions whose top hit is right, reject off-topic ones
    best = (kb.MIN_SCORE, -1.0)
    rows = []
    for i in range(50, 86):
        t = i / 100
        kept = statistics.mean(q["top_score"] >= t and q["rank"] == 1 for q in faq)
        rejected = statistics.mean(q["top_score"] < t for q in off) if off else 1.0
        rows.append((t, kept, rejected))
        if kept + rejected > best[1] + 1e-9:
            best = (t, kept + rejected)
    say("\nThreshold sweep (top-1 score):  threshold  FAQ kept with right hit  off-topic rejected")
    for t, kept, rejected in rows:
        if round(t * 100) % 5 == 0 or t == best[0]:
            say(f"{'':33}{t:.2f}{'  <- best' if t == best[0] else '':9} {100 * kept:6.1f}%  {100 * rejected:15.1f}%")
    say(f"top-1 score: FAQ questions median {statistics.median(q['top_score'] for q in faq):.3f} "
        f"(min {min(q['top_score'] for q in faq):.3f}), off-topic median "
        f"{statistics.median(q['top_score'] for q in off):.3f} (max {max(q['top_score'] for q in off):.3f})"
        if off else "")
    say(f"Recommended SKYY_KB_MIN_SCORE = {best[0]:.2f}")

    misses = [q for q in faq if q["rank"] != 1]
    say(f"\nMisses ({len(misses)}: right FAQ not the top hit):")
    for q in misses:
        got = ", ".join(f"{h['faq_id']} {h['score']:.2f}" for h in q["hits"][:3])
        say(f"  [{q['lang']}/{q['kind']}] {q['question']}\n      want {q['faq_id']} (rank {q['rank'] or '>5'}), got {got}")
    return best[0]


# ---------------------------------------------------------------- Part B: answers

async def ask(system: str, user: str, max_tokens: int = 250, json_mode: bool = False):
    """Returns (answer, seconds to first token)."""
    t0, first, text = time.perf_counter(), None, ""
    extra = {"response_format": {"type": "json_object"}} if json_mode else {}
    stream = await llm.chat.completions.create(
        model="gemma", stream=True, max_tokens=max_tokens, temperature=0,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        extra_body=kb.NO_THINKING, **extra)
    async for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            first = first or time.perf_counter() - t0
            text += chunk.choices[0].delta.content
    return text.strip(), first or time.perf_counter() - t0


JUDGE = """You grade answers from a college voice assistant. Its answers are spoken aloud, so they are meant to be short (one to four sentences): leaving out secondary details of the FAQ entry is fine and expected. You get the FAQ entry that answers the visitor's question, related FAQ entries, the question and the assistant's answer. The answer may be in English, Spanish or French and the FAQ in another language: judge the content, not the language.
Check two things first:
(a) Yes or no: if the main FAQ entry answers the question with yes or no, the answer must give the same yes or no. The opposite (or a yes/no that the rest of the answer contradicts) is correct 0.
(b) What the visitor needs to act: the specific place, system, page, office, step, number, date or amount the FAQ gives for this question. If the answer only says something general instead (for example "it depends", "you can find it online" or "check the requirements") without saying where or how, correct is 1.
(c) Openers: a neutral first word such as "Sure.", "Okay.", "Claro." or "Bien sûr." is fine. \
A "Yes"/"No" (or "Sí", "Oui", "Non") opener on a question that is not a yes-or-no question is \
misleading when spoken: correct is at most 1.
Reply with JSON only: {"correct": 2, "grounded": true, "reason": "one short sentence"}.
correct: 2 = the answer gives the main information the visitor asked for, as the main FAQ entry states it; 1 = partly: something the visitor needs in order to act is missing or vague; 0 = wrong, contradicts the FAQ, off-topic or no answer.
grounded: true if every factual claim in the answer (including names of offices, systems, websites, exams and numbers) is supported by the main or a related FAQ entry. Pointing the visitor to the College Information Center at 305-237-8888 is always allowed."""

JUDGE_OFFTOPIC = """You grade answers from a college voice assistant. The visitor's question is \
NOT covered by the college FAQ, so the assistant should say it doesn't have that information \
(it may refer the visitor to the College Information Center at 305-237-8888).
Reply with JSON only: {"correct": 2, "grounded": true, "reason": "one short sentence"}.
correct: 2 = politely declines, 1 = declines but also guesses, 0 = answers as if it knew.
grounded: true if the answer makes no factual claims about the college or the topic."""


def related_ids(q) -> list[str]:
    """The closest other FAQs from Part A. An answer may correctly use one of them (a veteran
    asking about credit also needs the JST from the military-training FAQ)."""
    return [h["faq_id"] for h in q.get("hits", []) if h["faq_id"] != q["faq_id"]][:3]


async def judge(j: dict, q, answer: str, faq_text: dict, related: list[str]):
    """Grade one answer with judge `j` (from make_judge)."""
    if q["faq_id"]:
        system, user = JUDGE, (f"Main FAQ entry:\n{faq_text[q['faq_id']]}\n\nRelated FAQ entries:\n"
                               + "\n".join(f"- {faq_text[f]}" for f in related)
                               + f"\n\nQuestion: {q['question']}\n\nAnswer: {answer}")
    else:
        system, user = JUDGE_OFFTOPIC, f"Question: {q['question']}\n\nAnswer: {answer}"
    try:
        r = await j["client"].chat.completions.create(
            model=j["model"], temperature=0, response_format={"type": "json_object"},
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            **j["kw"])
        raw = (r.choices[0].message.content or "").strip()
    except Exception as e:
        return {"correct": None, "grounded": None, "reason": f"error: {e!r}"[:160]}
    try:
        v = json.loads(raw[raw.find("{"):raw.rfind("}") + 1])
        return {"correct": int(v.get("correct", 0)), "grounded": bool(v.get("grounded")),
                "reason": str(v.get("reason", ""))}
    except (ValueError, TypeError):
        return {"correct": None, "grounded": None, "reason": f"unparsable: {raw[:80]}"}


async def answers(mode: str, base: kb.KnowledgeBase, qs, threshold: float, faq_text: dict, out,
                  j: dict | None):
    system = kb.system_prompt(SYSTEM, mode)
    await ask(system, "Hello", max_tokens=1)   # cache the system prompt, like kb.warm_up()
    rows = []
    for i, q in enumerate(qs, 1):
        user, notes = q["question"], []
        if mode == "rag":
            hits, _, _ = await base.search(q["question"], k=a.top_k, min_score=threshold)
            notes = [h.get("faq_id", h["kb_id"]) for h in hits]
            if hits:
                user = kb.with_notes(q["question"], hits)
        answer, ttft = await ask(system, user)
        r = {"mode": mode, "id": q["id"], "lang": q["lang"], "kind": q["kind"], "faq_id": q["faq_id"],
             "question": q["question"], "notes": notes, "answer": answer, "ttft_s": round(ttft, 3),
             "facts": fact_score(answer, q["facts"]),
             "language_ok": detect_language(answer, default=q["lang"]) == q["lang"],
             "phone": PHONE in answer.replace(" ", "-"), "related": related_ids(q)}
        if j:
            r["judge"] = await judge(j, q, answer, faq_text, r["related"])
            r["judge_by"] = j["name"]
        rows.append(r)
        out.write(json.dumps(r, ensure_ascii=False) + "\n")
        out.flush()
        if i % 20 == 0 or i == len(qs):
            print(f"  {mode}: {i}/{len(qs)} answered")
    return rows


def summarize(mode: str, rows):
    faq = [r for r in rows if r["faq_id"]]
    off = [r for r in rows if not r["faq_id"]]
    judged = [r for r in faq if r.get("judge", {}).get("correct") is not None]
    say(f"\n--- {mode}: {len(faq)} FAQ questions, {len(off)} off-topic ---")
    say(f"key facts, mean            {100 * statistics.mean(r['facts'] for r in faq):5.1f}%")
    say(f"all key facts present      {pct(r['facts'] == 1 for r in faq)}")
    if judged:
        say(f"judge: fully correct (2)   {pct(r['judge']['correct'] == 2 for r in judged)}")
        say(f"judge: grounded            {pct(r['judge']['grounded'] for r in judged)}")
    say(f"answer in visitor language {pct(r['language_ok'] for r in rows)}")
    if off:
        say(f"off-topic: gives {PHONE} {pct(r['phone'] for r in off)}")
        oj = [r for r in off if r.get("judge", {}).get("correct") is not None]
        if oj:
            say(f"off-topic: declines (judge) {pct(r['judge']['correct'] == 2 for r in oj)}")
    say(f"median first token         {1000 * statistics.median(r['ttft_s'] for r in rows):.0f} ms")
    for lang in ("en", "es", "fr"):
        g = [r for r in faq if r["lang"] == lang]
        if g:
            say(f"  {lang}: facts {100 * statistics.mean(r['facts'] for r in g):5.1f}%"
                + (f", judge 2 {pct(r['judge']['correct'] == 2 for r in g if r.get('judge', {}).get('correct') is not None)}"
                   if judged else ""))
    lost = [r for r in rows if r["facts"] < 1 or not r["language_ok"]
            or (r.get("judge") or {}).get("correct", 2) != 2 or (not r["faq_id"] and not r["phone"])]
    worst = sorted(lost, key=lambda r: ((r.get("judge") or {}).get("correct") or 0, r["facts"]))[:10]
    say(f"answers that lost points ({len(lost)}, worst 10 shown):" if lost else "no answer lost points")
    for r in worst:
        j = r.get("judge") or {}
        say(f"  [{r['lang']}/{r['kind']}] {r['question']}\n      facts {r['facts']:.2f}, "
            f"judge {j.get('correct')}: {j.get('reason', '')}\n      -> {r['answer'][:200]}")
    return {"facts": statistics.mean(r["facts"] for r in faq),
            "correct": statistics.mean(r["judge"]["correct"] == 2 for r in judged) if judged else None}


# ---------------------------------------------------------------- judge test (planted errors)

DECLINE = {
    "en": "I'm sorry, I don't have that information. Please contact the College Information Center at 305-237-8888.",
    "es": "Lo siento, no tengo esa información. Comuníquese con el Centro de Información del Colegio al 305-237-8888.",
    "fr": "Je suis désolée, je n'ai pas cette information. Contactez le College Information Center au 305-237-8888."}
INVENTED = {   # a claim no FAQ entry supports
    "en": "You will also receive a free laptop from the Student Center once you do this.",
    "es": "Además, recibirá una laptop gratis del Student Center cuando lo haga.",
    "fr": "Vous recevrez aussi un ordinateur portable gratuit du Student Center une fois cela fait."}
# planted variant -> should a good judge pass it (correct 2 and grounded)?
SHOULD_PASS = {"unchanged": True, "other_language": True, "key_only": True,
               "number": False, "wrong_topic": False, "decline": False, "invented": False,
               "wrong_place": False, "step_removed": False, "negated": False}
# Plausible but wrong office, system or website: the first one found in an answer is swapped
WRONG_PLACE = [
    ("Admissions & Registration", "Financial Aid"), ("Admisiones y Registro", "Ayuda Financiera"),
    ("Advisement and Career Services", "Admissions"), ("Advisement & Career Services", "Admissions"),
    ("MDC Navigate", "Blackboard"), ("MDConnect", "Canvas"), ("Academic Record", "Financial Aid"),
    ("Admissions", "Financial Aid"), ("Admisiones", "Ayuda Financiera"), ("bookstore", "library"),
    ("librería", "biblioteca"), ("librairie", "bibliothèque"), ("Bursar", "Registrar"),
    ("Brainfuse", "Khan Academy"), ("Bank Mobile", "PayPal"), ("FAFSA", "CSS Profile"),
    ("Joint Services Transcript", "DD-214"), ("REVEST", "ACCESS"), ("CLEP", "SAT")]
FLIP = {"en": {"Yes": "No", "No": "Yes"}, "es": {"Sí": "No", "Si": "No", "No": "Sí"},
        "fr": {"Oui": "Non", "Non": "Oui"}}


def sentences(text: str) -> list[str]:
    return [x for x in re.split(r"(?<=[.!?])\s+", text.strip()) if x]


def key_split(answer: str, facts):
    """(sentences with a key fact, the other sentences)."""
    s = sentences(answer)
    key = [x for x in s if any(has(x, alt) for group in facts for alt in group)]
    return key, [x for x in s if x not in key]


def key_only(answer: str, facts) -> str | None:
    """Good, shorter answer: only the sentences with key facts (all of them still there)."""
    key, rest = key_split(answer, facts)
    if key and rest and fact_score(" ".join(key), facts) == 1 and len(" ".join(rest).split()) >= 4:
        return " ".join(key)
    return None


def step_removed(answer: str, facts) -> str | None:
    """The sentences with the key facts dropped; what is left still sounds like an answer."""
    key, rest = key_split(answer, facts)
    left = " ".join(rest)
    if key and len(left.split()) >= 6 and fact_score(left, facts) < 1:
        return left
    return None


def wrong_place(answer: str, facts) -> str | None:
    for right, wrong in WRONG_PLACE:
        if right in answer:
            return answer.replace(right, wrong, 1)
    return None


def is_yes_no(faq_note: str) -> bool:
    """The FAQ answers yes or no (the note is "question answer")."""
    return bool(re.search(r"\?\s+(Yes|No)\b", faq_note))


def negated(answer: str, lang: str) -> str | None:
    """Opening yes/no flipped ("Yes, you may..." -> "No, you may...")."""
    m = re.match(r"(\w+)[.,!]", answer.strip())
    if m and m[1] in FLIP[lang]:
        return FLIP[lang][m[1]] + answer.strip()[m.end(1):]
    return None


def change_number(answer: str, facts) -> str | None:
    """Change the first number that is a key fact (12 -> 19, 3.50 -> 4.50). None if there is none."""
    keys = {norm(alt) for group in facts for alt in group}
    for m in re.finditer(r"(?<![\d-])(\d+)([.,]\d+)?(?![\d-])", answer):   # skips 305-237-8888
        if {m.group(0), m.group(1), m.group(0).rstrip("0").rstrip(".,")} & keys:
            n = int(m.group(1))
            new = f"{n + 1}{m.group(2)}" if m.group(2) else str(n + 7)
            return answer[:m.start()] + new + answer[m.end():]
    return None


def plant(r, pool: list, qs_by_id: dict):
    """Yield (variant, answer) pairs for one good answer `r`."""
    q = qs_by_id[r["id"]]
    yield "unchanged", r["answer"]
    other = [p for p in pool if p["faq_id"] == r["faq_id"] and p["lang"] != r["lang"]]
    if other:
        yield "other_language", other[0]["answer"]
    same_lang = [p for p in pool if p["lang"] == r["lang"] and p["faq_id"].split("-")[0] != r["faq_id"].split("-")[0]]
    if same_lang:   # an answer from another FAQ category, in the same language
        yield "wrong_topic", same_lang[zlib.crc32(r["id"].encode()) % len(same_lang)]["answer"]
    yield "decline", DECLINE[r["lang"]]
    yield "invented", f"{r['answer'].rstrip()} {INVENTED[r['lang']]}"


async def judge_test(path: str, faq_text: dict):
    """Run each judge on good answers with planted errors; compare with the known verdicts."""
    judges = [make_judge(name) for name in a.judges]   # fail now if a key is missing
    qs_by_id = {q["id"]: q for q in map(json.loads, open(a.questions, encoding="utf-8"))}
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    seen, pool = set(), []
    for r in rows:   # each FAQ question once, only answers with every key fact
        if r["faq_id"] and r["id"] in qs_by_id and r["id"] not in seen \
                and fact_score(r["answer"], qs_by_id[r["id"]]["facts"]) == 1:
            seen.add(r["id"])
            pool.append(r)
    n = min(a.judge_test_n, len(pool))
    picked = [pool[round(i * len(pool) / n)] for i in range(n)]
    # Variants that only some answers allow (a key number, a yes/no opener, ...): drawn from
    # the whole pool, up to n each, spread over it
    sparse = {"number": lambda r, q: change_number(r["answer"], q["facts"]),
              "key_only": lambda r, q: key_only(r["answer"], q["facts"]),
              "step_removed": lambda r, q: step_removed(r["answer"], q["facts"]),
              "wrong_place": lambda r, q: wrong_place(r["answer"], q["facts"]),
              "negated": lambda r, q: (negated(r["answer"], r["lang"])
                                       if is_yes_no(faq_text[r["faq_id"]])
                                       # the FAQ asks "any campus?" (yes), the test set "my own campus?" (no)
                                       and r["faq_id"] != "finaid-10" else None)}
    work = [(r, list(plant(r, pool, qs_by_id))) for r in picked]
    extra = 0
    for variant, make in sparse.items():
        cands = [(r, v) for r in pool if (v := make(r, qs_by_id[r["id"]]))]
        k = min(n, len(cands))
        work += [(r, [(variant, v)]) for r, v in (cands[round(i * len(cands) / k)] for i in range(k))]
        extra += k
    base = None
    if any("related" not in r for r, _ in work):   # results from before "related" was saved
        base = kb.KnowledgeBase(a.host, timeout=10)
    items = []
    for r, variants in work:
        related = r.get("related")
        if related is None:
            hits, _, _ = await base.search(r["question"], k=4, min_score=0)
            related = [h.get("faq_id", h["kb_id"]) for h in hits if h.get("faq_id") != r["faq_id"]][:3]
        for variant, answer in variants:
            items.append({"id": r["id"], "lang": r["lang"], "variant": variant, "should_pass": SHOULD_PASS[variant],
                          "question": r["question"], "faq_id": r["faq_id"], "answer": answer, "related": related})
    if base:
        await base.aclose()

    say(f"Judge test, {datetime.now():%Y-%m-%d %H:%M}: {n} good answers (+{extra} for the targeted variants) "
        f"from {Path(path).name}, "
        f"{len(items)} items with planted errors, judges: {', '.join(j['name'] for j in judges)}\n")
    for j in judges:
        for i, it in enumerate(items, 1):
            v = await judge(j, qs_by_id[it["id"]], it["answer"], faq_text, it["related"])
            it[j["name"]] = v
            it[j["name"] + " passed"] = (None if v["correct"] is None
                                         else v["correct"] == 2 and v["grounded"] is True)
            if i % 40 == 0 or i == len(items):
                print(f"  {j['name']}: {i}/{len(items)} graded")

    say(f"{'variant':16} {'should':>7} {'n':>4}" + "".join(f"  {j['name'][:22]:>22}" for j in judges))
    for variant, should in SHOULD_PASS.items():
        group = [it for it in items if it["variant"] == variant]
        if group:
            say(f"{variant:16} {'pass' if should else 'fail':>7} {len(group):4}" + "".join(
                f"  {pct(it[j['name'] + ' passed'] == should for it in group):>22}" for j in judges))
    say("(each cell: share of items the judge got right)\n")
    for j in judges:
        k = j["name"] + " passed"
        good = [it for it in items if it["should_pass"]]
        planted = [it for it in items if not it["should_pass"]]
        unparsed = sum(it[k] is None for it in items)
        say(f"{j['name']}: caught {pct(it[k] is False for it in planted)} of planted errors, "
            f"wrongly marked down {pct(it[k] is False for it in good)} of good answers, "
            f"overall {pct(it[k] == it['should_pass'] for it in items)} right"
            + (f", {unparsed} not graded (errors/unparsable)" if unparsed else ""))
    if len(judges) >= 2:
        ks = [j["name"] + " passed" for j in judges]
        both = [it for it in items if all(it[k] is not None for k in ks)]
        diff = [it for it in both if len({it[k] for k in ks}) > 1]
        say(f"\nThe {len(judges)} judges agree on {pct(len({it[k] for k in ks}) == 1 for it in both)} "
            f"of items. Disagreements ({len(diff)}):")
        for it in diff[:20]:
            say(f"  [{it['lang']}/{it['variant']}, should {'pass' if it['should_pass'] else 'fail'}] {it['question']}")
            for j in judges:
                v = it[j["name"]]
                say(f"      {j['name']}: correct {v['correct']}, grounded {v['grounded']}: {v['reason']}")
    outdir = HERE / "eval_results"
    outdir.mkdir(exist_ok=True)
    stamp = f"{datetime.now():%Y%m%d_%H%M}"
    with open(outdir / f"judge_test_{stamp}.jsonl", "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    (outdir / f"judge_test_{stamp}_summary.txt").write_text("\n".join(report), encoding="utf-8")
    print(f"\nPer-item results: {outdir / f'judge_test_{stamp}.jsonl'}")


async def main():
    entries = kb.load_kb()
    faq_text = {kb.faq_id(e): kb.note(e) for e in entries if e.get("lang", "en") == "en"}
    if a.judge_test:
        await judge_test(a.judge_test, faq_text)
        return
    j = None if a.no_judge or a.judge == "none" else make_judge(a.judge)
    qs = [json.loads(l) for l in open(a.questions, encoding="utf-8") if l.strip()]
    if a.limit and a.limit < len(qs):   # spread over the set, so every kind and language is in
        qs = [qs[round(i * len(qs) / a.limit)] for i in range(a.limit)]

    r = httpx.get(f"http://{a.host}:{kb.QDRANT_PORT}/collections/{kb.COLLECTION}", timeout=10,
                  headers={"api-key": kb.QDRANT_KEY, "Connection": "close"})
    if r.status_code == 401:
        raise SystemExit("Qdrant answered 401: set $env:SKYY_QDRANT_KEY to the Qdrant API key.")
    if r.status_code == 404:
        raise SystemExit(f"No Qdrant collection '{kb.COLLECTION}': run 08_kb_index.py --host {a.host} first.")
    r.raise_for_status()

    base = kb.KnowledgeBase(a.host, timeout=10)
    say(f"Knowledge-base evaluation, {datetime.now():%Y-%m-%d %H:%M}, {len(qs)} questions "
        f"from {Path(a.questions).name}, {kb.KB_PATH.name}\n")
    best = await retrieval(base, qs)

    if not a.retrieval_only:
        threshold = a.min_score if a.min_score is not None else best
        say(f"\n=== Part B: answers (rag threshold {threshold:.2f}, top {a.top_k}"
            f", judge: {j['name'] if j else 'none'}) ===")
        outdir = HERE / "eval_results"
        outdir.mkdir(exist_ok=True)
        stamp = f"{datetime.now():%Y%m%d_%H%M}"
        path = outdir / f"kb_eval_{stamp}.jsonl"
        results = {}
        with open(path, "w", encoding="utf-8") as out:
            for mode in a.modes:
                results[mode] = summarize(mode, await answers(mode, base, qs, threshold, faq_text, out, j))
        if len(results) > 1:
            say("\nSummary: " + ", ".join(
                f"{m} facts {100 * s['facts']:.1f}%"
                + (f" / fully correct {100 * s['correct']:.1f}%" if s["correct"] is not None else "")
                for m, s in results.items()))
        (outdir / f"kb_eval_{stamp}_summary.txt").write_text("\n".join(report), encoding="utf-8")
        print(f"\nPer-question results: {path}")
    await base.aclose()


asyncio.run(main())
