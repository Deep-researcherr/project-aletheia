# Deep Researcher

An agentic research system. It takes a question, plans sub-questions, searches several sources in parallel, collects and deduplicates evidence, checks whether the evidence is sufficient and searches again as needed, and then synthesises a final cited report.

## How it works (the simple version)

Think of it as a small research team working down an assembly line:

```text
        ┌──────────────────────────────────────────────────┐
        │ 1. YOU ASK — one question, e.g. "which framework  │
        │    should I build my app with?"                   │
        └────────────────────────┬─────────────────────────┘
                                 ▼
        ┌──────────────────────────────────────────────────┐
        │ 2. THE PLANNER splits it into 3–5 smaller        │
        │    questions, one per angle                      │
        └────────────────────────┬─────────────────────────┘
                                 ▼
        ┌──────────────────────────────────────────────────┐
        │ 3. THE SEARCHERS look everything up at once —    │
        │    web, news, papers, code, patents              │
        └────────────────────────┬─────────────────────────┘
                                 ▼
        ┌──────────────────────────────────────────────────┐
        │ 4. THE READER opens the actual pages (not just   │
        │    snippets) and takes clean, traceable notes    │
        └────────────────────────┬─────────────────────────┘
                                 ▼
        ┌──────────────────────────────────────────────────┐
        │ 5. THE FACT-CHECKER asks "are we missing         │
        │    anything?" If yes → back to step 3. Genuine    │
        │    disagreements between sources get flagged.    │
        └────────────────────────┬─────────────────────────┘
                                 ▼
        ┌──────────────────────────────────────────────────┐
        │ 6. THE WRITER assembles the final report, every  │
        │    claim footnoted to a source that was read     │
        └──────────────────────────────────────────────────┘
```

Two rules hold the whole line together: nothing the team writes may come from thin air (every claim traces back to a document it actually read), and when the evidence is too thin the report says so on the cover instead of bluffing.

## Overview

```text
                    ┌──────────────┐
   research  ──────▶│   Planner    │  question → 3-5 sub-questions + searches
   question          └──────┬───────┘
                           │ pending searches
                           ▼
                 ┌─────────────────────┐
                 │   route (LangGraph) │  one parallel worker per search
                 └──┬───┬───┬───┬──────┘
                    ▼   ▼   ▼   ▼
              ┌──────────────┐
              │ search_worker│  web / news / scholar / github
              └──────┬───────┘
                     ▼
              ┌──────────────┐
              │   collect    │  dedupe by URL, flag stale
              └──────┬───────┘
                     ▼
              ┌──────────────┐   not sufficient → follow-up searches, loop
              │  gap_check   │ ─────────────────────────────┐
              └──────┬───────┘                              │
                     │ sufficient / out of rounds           │
                     ▼                                      │
          ┌────────────────────┐                             │
          │ contradiction_check│  sources that disagree      │
          └─────────┬──────────┘                             │
                    ▼                                        │
             ┌────────────┐                                  │
             │ synthesizer│  final cited report              │
             └────────────┘                                  │
                    └────────────────────────────────────────┘
```

Orchestration is a real [LangGraph](https://github.com/langchain-ai/langgraph) `StateGraph` in `backend/research/graph.py`. Fan-out uses `Send`, and the gap-check loop re-enters the same search graph until the evidence is sufficient or the round limit is reached.

The backend is FastAPI; the frontend is plain HTML/CSS/JS with no build step. FastAPI also serves the frontend, so there is no second server to run.

## Features

- Breaks a question into 3–5 sub-questions, each with its own searches
- Runs every search in a source round concurrently
- Sources: Google web, Google News, Google Scholar, GitHub, patents via SerpApi, plus arXiv's free Atom API (no credits, full-text preprints)
- Prefetches a Wikipedia baseline before any model sees retrieved content
- Reads the real documents (PDFs, papers, ebooks, repos, patents) instead of summarising snippets
- Falls back to open-access copies, then Tavily, then Firecrawl for unreadable pages
- Deduplicates evidence by normalised URL and flags sources older than three years as stale
- Filters social/video noise from results and downranks wrong-entity matches
- Loops back for more searching when the evidence is thin, aimed at named evidence gaps, up to a configurable round limit
- Detects genuine disagreements between sources
- Caps off-target evidence (wrong model variant, wrong architecture) instead of spending strong findings on it
- Marks the report partial when a dimension lacks adequate evidence, and says so in the report itself
- Streams every stage to the browser over SSE — no polling, no waiting for the whole run
- Retries and falls back across several models when one is rate-limited or overloaded, with one bounded second pass when everything throttles at once
- Skips models that just failed instead of re-trying them at the next stage
- Server-renders the report to HTML so the browser just displays it
- Tabbed results (Report / Sources / How it was researched) with a live step-by-step progress list, light and dark themes
- Optional mock mode runs the entire pipeline with zero API calls
- Dev-only SerpApi response cache

## Requirements

- **Python 3.10 or newer.** This is a hard floor, not a preference: `langgraph`,
  `litellm` and `fastapi` all declare `Requires-Python >=3.10`. On 3.9 or older
  pip cannot install them and the test suite dies at import time.
- A SerpApi key (required for all search)
- At least one LLM provider key — Gemini, OpenRouter, Cohere or Groq
- `GITHUB_TOKEN` optional; raises GitHub API rate limits only

## Setup

```bash
git clone <repository-url>
cd project-aletheia

python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements-dev.txt      # includes requirements.txt + pytest
cp .env.example .env                     # then fill in your keys
```

`requirements-dev.txt` is what you want for development — it pulls in
`requirements.txt` plus `pytest` and `dukpy` (used to execute
`frontend/markdown.js` under test).

### Verifying the install

```bash
python -c "import sys; print(sys.version)"    # must be 3.10+
pytest -q -m "not live"                       # 225 tests, no network needed
```

### If you see many errors at once

An `ERROR collecting ...` block per test file, all naming the same missing
module, means **dependencies were not installed** — not broken code. It looks
like a wall of failures because pytest reports one collection error per test
module:

```text
ERROR collecting tests/test_deep_research.py
  backend/research/fetch.py:33: in <module>
    from bs4 import BeautifulSoup
E   ImportError: No module named 'bs4'
ERROR collecting tests/test_frontend_render.py
...
```

Fix it from the repository root with the virtualenv active:

```bash
pip install -r requirements-dev.txt
```

Common causes:

| Symptom | Cause |
| --- | --- |
| `No module named 'bs4'` / `lxml` / `pypdf` | dependencies not installed, or `pip install` run without the venv active |
| `SyntaxError` or `TypeError: unsupported operand type(s) for \|` on import | Python older than 3.10 |
| `No module named 'backend'` | pytest invoked from the wrong directory — run it from the repository root, or use `python -m pytest` |
| Tests pass but every answer is canned | `LLM_MOCK=1` or `SEARCH_MOCK=1` is set in the environment; unset it |

## Configuration

Copy the template and fill in the keys:

```bash
cp .env.example .env
```

`.env` is gitignored. Never commit it.

### Credentials

| Variable | Required | Purpose |
| --- | --- | --- |
| `GEMINI_API_KEY` | yes | used by `litellm` for the `gemini/...` model prefix |
| `SERPAPI_KEY` | yes | all five search sources |
| `GITHUB_TOKEN` | no | raises GitHub API rate limits |
| `TAVILY_API_KEY` | no | paid extraction fallback; empty disables it |
| `FIRECRAWL_API_KEY` | no | paid fallback after Tavily; empty disables it |
| `UNPAYWALL_EMAIL` | no | any address; enables Unpaywall open-access lookup |

### Model chain

Calls try models in order and move on when one is unavailable. Switching
models is a `.env`-only change: set the two lines, restart uvicorn (it loads
`.env` once at boot), and the new chain takes effect. Restarting also clears
all model cooldowns.

| Variable | Default | Purpose |
| --- | --- | --- |
| `LLM_MODEL` | `gemini/gemini-3.5-flash-lite` | primary model |
| `LLM_FALLBACKS` | — | comma-separated fallbacks, tried in order |
| `LLM_MODEL_DEFAULT` | falls back to `LLM_MODEL` | planner, gap check |
| `LLM_MODEL_JUDGE` | falls back to `LLM_MODEL` | contradiction check |
| `LLM_MODEL_SYNTH` | falls back to `LLM_MODEL` | report synthesis |

Every model string must carry a provider prefix, and every model in the chain
must support JSON mode (the planner, extraction and gap check all send
`response_format=json_object`). A per-role override is prepended, not
isolated: `LLM_MODEL` stays second in that role's chain.

### Resilience

| Variable | Default | Purpose |
| --- | --- | --- |
| `LLM_MAX_ATTEMPTS` | `2` | attempts per model for transient 5xx / connection errors |
| `LLM_DAILY_COOLDOWN_S` | `3600` | how long to park a model after a daily-quota 429 |
| `LLM_PER_MINUTE_COOLDOWN_S` | `60` | fallback cooldown when a 429 reports no retry delay |
| `LLM_OVERLOAD_COOLDOWN_S` | `30` | cooldown after a model exhausts every attempt on 503 |
| `LLM_CHAIN_RETRY_WAIT_S` | `30` | longest to wait before one second pass over the whole chain; only when every failure was a temporary rate limit |

A per-model health tracker, guarded by a lock and timed with `time.monotonic()`, remembers which models just failed. Cooling models are skipped when the chain is built, so a model that returned 429s or 503s does not get retried immediately in the next stage. Cooldowns and learned prompt-size ceilings persist in `.cache/model_health.json`, so a restart no longer wipes the system's memory of failing models.

### Caching

| Variable | Default | Purpose |
| --- | --- | --- |
| `SEARCH_CACHE_DIR` | — | dev-only SerpApi response cache; unset disables it |

Entries are keyed by a sha1 of the source, query and result count. A relative path resolves against the repository root, not the process working directory. Mock mode bypasses the cache.

### Mock mode

Everything above is optional. With these set, the pipeline runs end to end with no API calls at all — useful for frontend work, offline development and tests.

| Variable | Purpose |
| --- | --- |
| `LLM_MOCK=1` | return canned replies shaped like each caller's parser |
| `LLM_MOCK_FAIL` | `planner`, `gap`, `judge`, `synth` or `all` — force a failure |
| `LLM_MOCK_DELAY_MS` | artificial latency so streaming is visible (default 400) |
| `SEARCH_MOCK=1` | return deterministic synthetic results |
| `SEARCH_MOCK_FAIL` | `web`, `news`, `scholar`, `github` or `all` |
| `SEARCH_MOCK_DELAY_MS` | artificial latency (default 300) |

## Running

```bash
source venv/bin/activate
uvicorn backend.main:app --reload
```

Then open:

```text
http://127.0.0.1:8000
```

The frontend is served by the same process. To develop the frontend separately, run it on port 5500, which is already allowed by the CORS configuration:

```bash
python3 -m http.server 5500 --directory frontend
```

The frontend requests `/api/research` with a relative path, so it works under either setup.

## API

### `GET /api/research`

| Parameter | Default | Notes |
| --- | --- | --- |
| `q` | required | the research question; missing returns 422 |
| `rounds` | `2` | clamped to the configured ceiling (`RESEARCH_MAX_ROUNDS_LIMIT`) |
| `throttle` | configured default | `0`/`1` overrides request pacing for this run only |

Responds with `text/event-stream`. Every frame is `data: {json}\n\n`, and the stream always ends with a `done` frame.

```bash
curl -N "http://127.0.0.1:8000/api/research?q=What%20is%20FastAPI%3F&rounds=1"
```

### Events

| Type | Payload |
| --- | --- |
| `plan` | `data` = `{"subquestions":[{"question","searches":[{"source","query"}]}]}` |
| `results` | `source`, `query`, `count`, `error` — one per search |
| `evidence` | `count` of unique evidence collected so far |
| `gap` | `data` = `{"sufficient","missing","follow_ups"}` |
| `retrieval` | `data` = `{attempted,retrieved,full_text,partial,metadata_only,failed,words,chunks,references_followed,sources[]}` |
| `analysis` | `data` = `{records,dimensions[],sources_summarised,notes[]}` |
| `contradictions` | `data` = list of `{topic,side_a,sources_a,side_b,sources_b,likely_cause}` |
| `status` | `data` = completion state, so clients can render it before the body arrives |
| `report_section` | one finished section at a time (`heading`, `data`, `html`) for progressive rendering |
| `report` | `data` = markdown, `html` = server-rendered HTML, `sources`, `stats`, `retrieval`, `export` |
| `error` | `message` |
| `done` | always last |

`retrieval` and `analysis` are additions. `evidence` still reports the count of
collected search results, so existing clients keep working unchanged.

A failing search does not stop the run: it is reported in that search's `results` event with `error` set. A pipeline-level failure emits `error` followed by `done`.

A typical `rounds=1` run:

```text
plan → results ×N → evidence → retrieval → analysis → gap → contradictions → report → done
```

With more rounds, a `gap` reporting `sufficient: false` is followed by another batch of `results`, then a second `evidence`/`retrieval`/`analysis`. Evidence accumulates: findings from earlier rounds are never discarded.

### CORS

The frontend calls the API with a **relative** URL, so it is same-origin and
CORS does not apply. It only becomes relevant when the page is served from a
different origin than the API — a separate dev server, or a teammate reaching
the app over the LAN.

By default any **loopback** origin is allowed on any port
(`localhost`, `127.0.0.1`, `[::1]`), so a dev server on 5173 or 3000 works
without configuration. Remote hosts are refused. To allow one:

```bash
CORS_ORIGINS=http://192.168.1.50:8000,https://research.internal
```

Set `CORS_ORIGIN_REGEX` to replace the loopback pattern, or to an empty string
to disable CORS matching entirely.

If you hit a CORS error, check the **browser console**, not the terminal: the
server logs the request as a normal 200 while the browser blocks the response.

### `GET /api/report/download?q=…`

Serves the markdown export for the most recent run of that question as a file
download (`Content-Disposition: attachment`). 404 if no export exists for the
question. `/api/report/markdown?q=…` returns the same text as JSON for clients
that would rather render it themselves. Generated files are also browsable at
`/exports/<filename>`.

A browser cannot write to the user's disk, so the server writes the file and
serves it back. Nothing here tries to work around browser restrictions.

### `GET /api/report/pdf?q=…`

Compiles the most recent completed run for the question into a styled PDF
(navy cover band, run-overview tiles, report sections, evidence table,
regrouped sources, reference cards with tappable citations) and serves it as
a download. Built deterministically from the recorded run bundle — the same
payloads the frontend received — so the PDF mirrors what was displayed. 404
when no completed run exists. The frontend's **Save PDF** button calls this
endpoint with the current question, depth and pacing for the overview table.

### Health and model inventory

- `GET /health` and `GET /api/health` return `{"status": "ok"}` for uptime monitors and load balancers.
- `GET /v1/models` lists the models in the configured chain in OpenAI list format, so external tooling probing for an LLM endpoint gets an answer instead of a 404.

## Frontend

No build step, no framework — `index.html`, `app.js`, `style.css` plus a
bundled serif display font. Results render as three tabs (Report / Sources /
How it was researched) above a live step-by-step progress list, with flowing
retrieval counters while the run is in motion. The report arrives as
server-rendered HTML with hover reference popups; a floating settings menu
holds the light/dark theme switch. Selects carry `ⓘ` tooltips fed by the
backend's own limit data.

## How it works

The pipeline has two halves. Everything that can be done by parsing is done by
parsing; the model is only asked for things that need judgement.

```
question
  → planner            LLM   sub-questions and searches
  → search ×N          API   SerpApi: web / news / scholar / github
  → collect            —     dedupe by URL, flag stale
  → retrieve           —     download + parse the real documents
  → analyse            LLM   batched evidence extraction, quote-verified
  → gap_check          LLM   is there a specific gap worth another search?
      ├─ gaps → search again (evidence accumulates, nothing is discarded)
      └─ done ↓
  → contradiction_check LLM  disagreements between findings
  → synthesizer        LLM   section-wise deep report
```

### Planner — `backend/research/planner.py`

Sends the question to the model and validates the reply: it must be a JSON object with a `subquestions` list, and searches are filtered to known sources with a non-empty query. Invalid shapes are rejected before the search stage begins.

### Search — `backend/research/searcher.py`

`search(source, query, n)` maps each source to a SerpApi engine — `google`, `google_news`, `google_scholar`, `google` with a `site:github.com` prefix, and `google_patents` — and normalises every row to `{title, link, snippet, source}`. Social/video hosts are filtered from web/news results (`SEARCH_BLOCKED_HOSTS`), and Scholar rows carry cited-by counts plus open-access PDF links. `search_full()` additionally returns what the response carried for free: related questions/searches (fed to gap recovery), the knowledge-graph entity (used to downrank wrong-entity matches), and answer boxes (added as bonus evidence). The planner can sharpen queries with autocomplete suggestions (`SEARCH_EXPAND_QUERIES`), and forward citation chasing (`cited_by_search`) finds newer papers citing heavily-cited ones. The results are then sent to the collector.

### Retrieval — `backend/research/fetch.py`

The search snippet is treated as a pointer, not as the source. Each result URL
is downloaded and parsed, by strategy:

| source | method | what you get |
| --- | --- | --- |
| `.pdf` | `pypdf` | full text, per page, up to `FETCH_MAX_PAGES` |
| `arxiv.org` | arXiv API | authoritative title, authors, date, DOI, abstract |
| `github.com/owner/repo` | GitHub API + raw README | stars, licence, topics, last push, and the README |
| Project Gutenberg | predictable plain-text URLs | full ebook text, boilerplate stripped |
| PubMed Central | BioC JSON API | section-headed full text, no HTML parsing |
| `link.springer.com` | article HTML | full text when open access, honest partial otherwise |
| `patents.google.com` | patent HTML | the filing, labelled as a patent |
| `text/html` | `lxml` parse | article container only, headings and tables preserved |
| plain / markdown / json | as-is | text kept verbatim, code indentation intact |

When the direct fetch fails, recovery runs cheapest-first: an open-access copy via OpenAlex / Semantic Scholar / Unpaywall (`backend/research/oa.py`), then Tavily extraction, then Firecrawl scraping (`backend/research/webextract.py`, capped per run with `TAVILY_MAX_PER_RUN` / `FIRECRAWL_MAX_PER_RUN`). Robots-blocked URLs are never escalated.

Every document is graded, and the grade travels with it into the report:

- `full` — the whole usable document was extracted
- `partial` — paywalled, truncated by a byte/page cap, or JavaScript-rendered
- `metadata_only` — only the search snippet exists; the document was never read
- `failed` — robots.txt, HTTP 403/429, network error, or unreachable

Nothing tries to defeat a login wall, a paywall or a `robots.txt` rule. When one
of those blocks retrieval the source is kept, explicitly downgraded, and the
reason is recorded rather than the source being silently dropped. Rate-limited
responses are retried once with backoff. Outbound links are recorded for
reference discovery.

Deduplicates by URL with `www.` and trailing slashes normalised away, drops rows without a URL, and marks anything three years or older as stale. `context(limit=40)` renders the numbered block that the model sees when it judges sufficiency and contradictions.

Documents are split on their own structure, never truncated wholesale. HTML
headings are recovered as `#`-prefixed markers during extraction, so the chunker
can rebuild a section path (`3.2 Attention`) and PDF page breaks are preserved.
Paragraphs, list items and table rows are the atomic units; a single oversized
block is split on sentence boundaries. Every chunk keeps `source_url`,
`source_title`, `source_type`, `publication_date`, `section`, `page` and
`chunk_id`.

### Relevance — `backend/research/relevance.py`

No model call. Passages are scored by IDF-weighted term overlap against the
question, with boosts for chunks that cover a named dimension, name a compared
entity in full, contain quantitative claims or were recently published (on
recency-signalled questions), and penalties for boilerplate, single-generic-term
matches (how song lyrics lose to benchmarks), snippet-only sources and
wrong-knowledge-graph-entity passages. Results are then diversified
round-robin across sources so one verbose document cannot fill the batch.
The same module extracts the comparison targets (`React Native, Flutter, native
Android`) and the analysis dimensions (`performance`, `app size`, …); the
dimensions decide which sections the report needs. It also detects
related-but-different-variant evidence (another release, another architecture
class) so extraction can cap it instead of spending strong findings on it.

### Evidence — `backend/research/evidence.py`, `extract.py`

Passages are packed to a character budget so many chunks go into one request, and
one request yields every finding those chunks support — evidence throughput per
model call rather than a call per fragment. Each record keeps `claim`, `detail`,
a verbatim `quote`, its dimension, its targets, full source metadata, the section
and page it came from, the retrieval grade, a quality tier
(`strong`/`moderate`/`weak`/`anecdotal`/`speculative`) and its limitations.

Two deterministic guards then run, because a model will happily produce a
plausible citation for a passage that does not contain it:

- a record naming a chunk that was not in the batch is discarded
- a quote that cannot be found in its passage is downgraded, then discarded

Records accumulate across gap-check rounds; a later round that finds less never
discards what an earlier round already established.

Findings about a related-but-different variant carry an explicit applicability
note and rank below directly applicable evidence for the same dimension.
Per-source synthesis skips weak-only sources instead of spending quota
assessing them. Scholar cited-by counts travel with records into prompts and
break reference-numbering ties toward authoritative works.

### Reference discovery — `backend/research/references.py`

Outbound links and DOIs from documents that were actually read are mined for
extra leads, filtered deterministically against the question, and capped by
`REFERENCES_MAX`, `REFERENCES_PER_SOURCE` and a default depth of 1. Social and
account-wall hosts are skipped. Highly-cited papers are additionally chased
*forward* to the documents citing them (`SCHOLAR_FORWARD_MAX` per run), since
citers are usually newer than the cited work. A paper's own citations are worth chasing once;
chasing them recursively is not.

### Contradiction check

Runs over the extracted findings rather than raw snippets, in batches, and is
instructed to distinguish a genuine contradiction from a difference explained by
date, configuration, workload or scope. Skipped when there are fewer than two
findings.

### Synthesis — `backend/research/report.py`

Section-wise. Reference numbers are assigned deterministically from the evidence
records *before any model runs*, so a citation in the prose always resolves to a
real document URL and a fabricated one is structurally impossible. Each section
is written by its own call given only the evidence relevant to it, which is what
keeps citations attached to the claim they actually support.

Sections: Executive Summary, Scope and Methodology, one per requested dimension,
Where the Evidence Conflicts, Scenario Analysis, Decision Framework, Evidence
Quality and Limitations, Conclusion, then a deterministic Evidence-by-Source
table and References list. `REPORT_DEPTH=brief` produces a short summary instead.
Each section is handed only the top-ranked evidence (`REPORT_EVIDENCE_PER_SECTION`).

A finished section is stored verbatim in a per-run output pool; if its model
call failed, one pool-grounded retry lets a fallback continue the report
instead of restarting it. Dimensions without enough adequate findings become
named evidence holes: the gap loop targets them, the limitations section must
state them, and the run reports `PARTIAL` until they are covered — completion
means adequate evidence, not just assembled sections. The server compiles the
finished markdown to HTML once, so browsers (and any other client) display it
without parsing anything.

If a section cannot be written the report says so in a visible note, rather than
reading as complete when it is not. With no retrieved evidence at all, the
original single-call report is used unchanged.

### LLM layer — `backend/research/llm.py`

Every call goes through one `ask()`:

- up to `LLM_MAX_ATTEMPTS` attempts per model with exponential backoff and jitter
- `RateLimitError` moves to the next model without retrying the same one
- `NotFoundError`, `AuthenticationError` and `BadRequestError` skip immediately
- programming errors such as `TypeError` and `KeyError` propagate untouched
- 60s per-call timeout, with `litellm`'s own retries disabled so they do not multiply
- failures log a `WARNING` naming the model, the exception class and a short summary
- exhausting the chain raises `LLMChainError`, which the SSE layer turns into an `error` event
- malformed model JSON is salvaged when possible (prose wrapping, trailing commas); otherwise the next model is tried, never a raw decoder traceback
- when every model failed only with temporary rate limits, the chain waits once (bounded by `LLM_CHAIN_RETRY_WAIT_S`) for the soonest recovery and tries a single second pass; auth failures and oversized prompts fail fast instead

A per-process health table parks a model that just returned 429/503 so the next
pipeline stage does not walk straight back into it, classifying Google's
`RESOURCE_EXHAUSTED` bodies into per-day and per-minute quotas, and learning
per-model prompt-size ceilings from request-too-large rejections so oversized
prompts skip models that cannot serve them. If every model is
cooling down, the soonest is tried anyway rather than failing outright.

Every prompt that processes collected material opens with the same hard rule:
use only what the prompt contains, never add facts from model knowledge. The
planner and gap check are exempt — their job is inventing questions, not
processing evidence.

`CallBudget` bounds total model calls per run; later stages reserve part of it so
report writing cannot be starved by extraction.

Model strings must carry a litellm provider prefix (`groq/…`, `openrouter/…`,
`mistral/…`, `cohere/…`, `nvidia_nim/…`). A bare name fails with "LLM Provider
NOT provided". The prefix is the provider, not always the vendor.

## Project structure

```text
project-aletheia/
├── backend/
│   ├── main.py                  FastAPI app, CORS, SSE, export download
│   ├── research/
│   │   ├── graph.py             LangGraph pipeline
│   │   ├── config.py            every tunable, env-driven with defaults
│   │   ├── fetch.py             document retrieval: pdf / arxiv / github / html,
│   │   │                        gutenberg / pmc / springer / patent
│   │   ├── oa.py                open-access resolution (OpenAlex, Scholar, Unpaywall)
│   │   ├── webextract.py        paid extraction fallback: Tavily, then Firecrawl
│   │   ├── chunker.py           structure-aware chunking with provenance
│   │   ├── relevance.py         deterministic scoring, targets & dimensions
│   │   ├── extract.py           batched evidence extraction + quote verification
│   │   ├── evidence.py          Evidence records, quality tiers, statistics
│   │   ├── references.py        bounded reference discovery, forward citations
│   │   ├── report.py            deep section-wise synthesis, reference numbering
│   │   ├── report_prompts.py    the original single-call prompt (fallback)
│   │   ├── output_pool.py       per-run finished-section memory for fallbacks
│   │   ├── markdown_html.py     server-side markdown → HTML compiler
│   │   ├── wiki.py              Wikipedia baseline, keyword-driven per sub-question
│   │   ├── export.py            markdown export with front matter
│   │   ├── llm.py               retries, fallback chain, health, budgets, mock
│   │   ├── planner.py           question → sub-questions (autocomplete-sharpened)
│   │   ├── searcher.py          SerpApi wrapper, extras, patents, cache, mock mode
│   │   ├── collector.py         dedupe, freshness, contradiction detection
│   │   └── researcher.py        graph → SSE events
├── frontend/
│   ├── index.html
│   ├── app.js                   SSE consumer, tabs, progress, tooltips, export
│   ├── markdown.js              escaping markdown renderer, citation anchors
│   ├── style.css
│   └── Anthropic_font/          bundled display font
├── exports/                     generated markdown reports (gitignored)
├── tests/
│   ├── test_sse.py              SSE contract, ordering, clamping, 422
│   ├── test_llm.py              retry, fallback, cooldowns, 429 classification, JSON salvage
│   ├── test_mock_pipeline.py    whole graph on mocks
│   ├── test_deep_research.py    fetch, chunking, evidence, relevance, export
│   ├── test_evidence_adequacy.py  rank penalties, fidelity caps, holes, recovery
│   ├── test_resilience_audit.py batch math, budgets, host filter, headings
│   ├── test_serpapi_integration.py  extras, patents, expansion, citations
│   ├── test_output_pool.py      pool immutability, resume, budget discipline
│   ├── test_search_cache.py     search cache
│   ├── test_pipeline.py         graph wiring, collector, live SerpApi
│   ├── test_planner_dedup.py    dedup, regeneration, planner fallback
│   ├── test_wiki.py             baseline pooling, keyword queries
│   ├── test_fetch_sources.py    gutenberg, BioC, OA resolver, paid fallbacks
│   └── test_frontend_render.py  markdown renderer via dukpy
├── conftest.py                  sys.path setup, `live` marker, env cleanup
├── .env.example
└── requirements.txt
```

## Request pacing

Free-tier providers do not fail politely: one 429 can cost the rest of the day,
and a tokens-per-minute ceiling is not something you discover cheaply. Spacing
calls out bounds the total sent in any window by construction.

Two independent knobs, because they solve different problems:

| Setting | Default | Protects against |
| --- | --- | --- |
| `THROTTLE_LLM_PER_MINUTE` | 12 | per-minute token/request budgets |
| `THROTTLE_LLM_MAX_CONCURRENT` | 1 | two runs jointly exceeding one quota |

A single run is already sequential for model calls, so the concurrency ceiling
matters when two browser tabs run pipelines at once — each run's pacing looks
correct in isolation while together they blow through a shared limit. Search can
be paced too (`THROTTLE_SEARCH`), off by default since SerpApi is a metered paid
plan.

The throttle is a process-wide singleton per kind. The provider's constraint
belongs to the provider, not to one request, so every run shares it.

**Cost.** For a run needing ~53 model calls:

| Rate | Spacing | Added to a ~20 min run |
| --- | --- | --- |
| off | — | 0 |
| 30/min | 2.0s | ~1.8 min |
| 20/min | 3.0s | ~2.7 min |
| 12/min | 5.0s | ~4.4 min |

**Turning it off.** The UI has a *Request pacing* selector — *Throttled* (the
default, while providers are free-tier) or *Unthrottled*. `GET /api/research`
also takes `?throttle=0|1`, which applies to that run only and then restores the
configured default, so one unthrottled request does not silently disable pacing
for everything after it. `GET /api/throttle` reports the live state, and the
UI shows how many calls were held back and for how long.

Once you are on paid subscriptions with headroom, either flip the selector or
set `THROTTLE_LLM=0` and raise `THROTTLE_LLM_PER_MINUTE` to taste.

## Sub-question planning

A planner asked to split one question often returns the same sub-question twice.
Duplicates waste a parallel search slot and skew the evidence base -- the same
material is retrieved and cited twice while another dimension goes unresearched.

`planner.make_plan()` therefore plans, deduplicates, and replaces:

1. **Detect.** Two independent signals, each with its own threshold. Rare-term
   overlap catches a hard restatement; character trigrams catch the same
   question reworded, since they are blind to morphology. Weighting the term
   overlap by rarity is what stops the shared entity names in a comparison
   question from making genuinely different dimensions look like duplicates.
2. **Replace.** The planner is asked again for sub-questions covering ground
   none of the survivors reach, told explicitly what was rejected.
3. **Re-filter.** Replacements can collide too, so the whole set is filtered
   again rather than trusting the new text.

If regeneration fails or returns nothing usable, the surviving plan is used
unchanged -- a smaller plan beats no plan. If *everything* looks duplicated, one
is kept rather than planning nothing.

Tune with `PLAN_DEDUP_THRESHOLD`, `PLAN_DEDUP_GRAM_THRESHOLD` (set either to
`1.0` to disable) and `PLAN_DEDUP_REGENERATE`.

Before searching, each planned query can be sharpened with autocomplete
phrasing (`SEARCH_EXPAND_QUERIES`): a suggestion replaces the draft only when
it adds a new technical term, and any failure keeps the original. A Wikipedia
baseline is prefetched per question and sub-question (keyword queries only)
so the pipeline starts from stable definitions at zero API cost.

## Research depth

The UI's depth selector and the `rounds` parameter both feed the gap-check loop.
The ceiling is `RESEARCH_MAX_ROUNDS_LIMIT` (default 10), read by the API and by
`/api/config`, so raising it needs no code change.

**Depth and budget are coupled.** Measured on a representative run, each extra
round costs about 18 model calls (~16 small extraction batches plus ~5 per-source
synthesis calls) on top of a fixed ~17:

| rounds | model calls needed |
| --- | --- |
| 1 | ~35 |
| 2 | ~53 |
| 3 | ~71 |
| 4 | ~89 |
| 6 | ~125 |
| 10 | ~197 |

If `RESEARCH_LLM_BUDGET` is unset the budget auto-sizes to the requested depth,
so deeper runs actually go deeper. If you pin it explicitly and it is too small,
the planner logs a warning and the shortfall becomes a pipeline note, and the
report comes back marked `PARTIAL` — a starved budget produces a shallower
report, never a silently truncated one.

Raising the round limit without raising the budget is the one way to make
things worse, so it is worth knowing which of the two is binding: look for
`model-call budget auto-sized to N` in the server log at the start of a run.

### Incremental analysis

Later rounds re-chunk the whole corpus, so without tracking what has been read
an extra round re-pays for documents already processed. `analyse` now skips
sources it has already extracted from, and re-reads a source only when its
retrieval grade *improves* -- a snippet that later became a real document is new
material. Measured on a corpus of 10 documents with 2 new ones per round:

| rounds | passages read before | after | saved |
| --- | --- | --- | --- |
| 1 | 10 | 10 | 0% |
| 2 | 22 | 12 | 45% |
| 3 | 36 | 14 | 61% |
| 4 | 52 | 16 | 69% |

Per-source synthesis is filtered the same way, so an extra round costs only its
genuinely new material. Set `ANALYSE_REUSE_SOURCES=0` to restore the old
behaviour.

## Testing

```bash
pytest -q
```

225 tests. All but one run offline. The retrieval and chunking tests stub the HTTP
session, so they need no network either. `test_serpapi_live` is marked `live` and spends one SerpApi credit; skip it with:

```bash
pytest -q -m "not live"
```

A `live` marker on a test tells the autouse `clean_env` fixture to leave `SERPAPI_KEY` alone, so the test can reach the real API.

## Tech stack

- **Backend** Python, FastAPI, Uvicorn, Pydantic
- **Orchestration** LangGraph
- **Models** `litellm`, Google Gemini
- **Search** SerpApi (`google-search-results`)
- **Frontend** HTML, CSS, vanilla JavaScript — no build step, no framework
- **Tests** pytest, FastAPI `TestClient`

## Development workflow

```bash
git switch -c feature/your-feature
venv/bin/python -m pytest -q
git diff --check
git add .
git commit -m "Describe your change"
git push -u origin feature/your-feature
```

Then open a pull request.

Mock mode makes it possible to work on the frontend with no API spend:

```bash
LLM_MOCK=1 SEARCH_MOCK=1 uvicorn backend.main:app --reload
```

## Known gaps

- `README` aside, there is no CI workflow, so nothing runs the suite on a pull request.
- The 429 parser matches several spellings of Google's quota id and retry delay because no live 429 was available to test against. A misclassification degrades to the shorter per-minute cooldown instead of the more precise daily backoff.
- Multiple uvicorn workers would each keep their own throttle state, though model health itself is shared through the on-disk registry.
- `graph.py` catches broad `Exception` around search calls, which will also mask genuine bugs in that block.
- The SerpApi cache rarely hits across runs, because the planner is nondeterministic and generates different queries each time. It reliably prevents duplicate calls within a run.

## Contributing

1. Create a feature branch.
2. Make focused changes.
3. Add or update tests.
4. Run the full suite.
5. Push and open a pull request.

Never commit API keys or other secrets.
