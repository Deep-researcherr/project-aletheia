# Deep Researcher

An agentic research system. It takes a question, plans sub-questions, searches several sources in parallel, collects and deduplicates evidence, checks whether the evidence is sufficient and searches again as needed, and then synthesises a final cited report.

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
- Sources: Google web, Google News, Google Scholar, and GitHub, all through SerpApi
- Deduplicates evidence by normalised URL and flags sources older than three years as stale
- Loops back for more searching when the evidence is thin, up to a configurable round limit
- Detects genuine disagreements between sources
- Streams every stage to the browser over SSE — no polling, no waiting for the whole run
- Retries and falls back across several models when one is rate-limited or overloaded
- Skips models that just failed instead of re-trying them at the next stage
- Optional mock mode runs the entire pipeline with zero API calls
- Dev-only SerpApi response cache

## Requirements

- Python 3.10+ (developed and tested on 3.14.7)
- A Gemini API key
- A SerpApi key
- A GitHub token is optional

## Setup

```bash
git clone <repository-url>
cd deep-researcher
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

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
| `SERPAPI_KEY` | yes | all four search sources |
| `GITHUB_TOKEN` | no | raises GitHub API rate limits |

### Model chain

Calls try models in order and move on when one is unavailable.

| Variable | Default | Purpose |
| --- | --- | --- |
| `LLM_MODEL` | `gemini/gemini-3.6-flash` | primary model |
| `LLM_FALLBACKS` | — | comma-separated fallbacks, tried in order |
| `LLM_MODEL_DEFAULT` | falls back to `LLM_MODEL` | planner, gap check |
| `LLM_MODEL_JUDGE` | falls back to `LLM_MODEL` | contradiction check |
| `LLM_MODEL_SYNTH` | falls back to `LLM_MODEL` | report synthesis |

### Resilience

| Variable | Default | Purpose |
| --- | --- | --- |
| `LLM_MAX_ATTEMPTS` | `2` | attempts per model for transient 5xx / connection errors |
| `LLM_DAILY_COOLDOWN_S` | `3600` | how long to park a model after a daily-quota 429 |
| `LLM_PER_MINUTE_COOLDOWN_S` | `60` | fallback cooldown when a 429 reports no retry delay |
| `LLM_OVERLOAD_COOLDOWN_S` | `30` | cooldown after a model exhausts every attempt on 503 |

A per-model health tracker, guarded by a lock and timed with `time.monotonic()`, remembers which models just failed. Cooling models are skipped when the chain is built, so a model that returned 429s or 503s does not get retried immediately in the next stage.

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

The frontend is served by the same process. To develop the frontend separately, run it on port 5500. The CORS configuration accepts localhost and 127.0.0.1 origins on any port:

```bash
python3 -m http.server 5500 --directory frontend
```

The frontend requests `/api/research` with a relative path, so it works under either setup.

## API

### `GET /api/research`

| Parameter | Default | Notes |
| --- | --- | --- |
| `q` | required | the research question; missing returns 422 |
| `rounds` | `2` | clamped to `1`–`4` |

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
| `report` | `data` = markdown, `sources`, `stats`, `retrieval`, `export` |
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

### `GET /api/report/download?q=…`

Serves the markdown export for the most recent run of that question as a file
download (`Content-Disposition: attachment`). 404 if no export exists for the
question. `/api/report/markdown?q=…` returns the same text as JSON for clients
that would rather render it themselves. Generated files are also browsable at
`/exports/<filename>`.

A browser cannot write to the user's disk, so the server writes the file and
serves it back. Nothing here tries to work around browser restrictions.

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

`search(source, query, n)` maps each source to a SerpApi engine — `google`, `google_news`, `google_scholar`, and `google` with a `site:github.com` prefix — and normalises every row to `{title, link, snippet, source}`. The results are then sent to the collector.

### Retrieval — `backend/research/fetch.py`

The search snippet is treated as a pointer, not as the source. Each result URL
is downloaded and parsed, by strategy:

| source | method | what you get |
| --- | --- | --- |
| `.pdf` | `pypdf` | full text, per page, up to `FETCH_MAX_PAGES` |
| `arxiv.org` | arXiv API | authoritative title, authors, date, DOI, abstract |
| `github.com/owner/repo` | GitHub API + raw README | stars, licence, topics, last push, and the README |
| `text/html` | `lxml` parse | article container only, headings and tables preserved |
| plain / markdown / json | as-is | text kept verbatim, code indentation intact |

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
question, with boosts for chunks that cover a named dimension or contain
quantitative claims, and a penalty for boilerplate. Results are then diversified
round-robin across sources so one verbose document cannot fill the batch.
The same module extracts the comparison targets (`React Native, Flutter, native
Android`) and the analysis dimensions (`performance`, `app size`, …); the
dimensions decide which sections the report needs.

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

### Reference discovery — `backend/research/references.py`

Outbound links and DOIs from documents that were actually read are mined for
extra leads, filtered deterministically against the question, and capped by
`REFERENCES_MAX`, `REFERENCES_PER_SOURCE` and a default depth of 1. Social and
account-wall hosts are skipped. A paper's own citations are worth chasing once;
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

A per-process health table parks a model that just returned 429/503 so the next
pipeline stage does not walk straight back into it, classifying Google's
`RESOURCE_EXHAUSTED` bodies into per-day and per-minute quotas. If every model is
cooling down, the soonest is tried anyway rather than failing outright.

`CallBudget` bounds total model calls per run; later stages reserve part of it so
report writing cannot be starved by extraction.

Model strings must carry a litellm provider prefix (`groq/…`, `openrouter/…`,
`mistral/…`, `cohere/…`, `nvidia_nim/…`). A bare name fails with "LLM Provider
NOT provided". The prefix is the provider, not always the vendor.

## Project structure

```text
deep-researcher/
├── backend/
│   ├── main.py                  FastAPI app, CORS, SSE, export download
│   ├── research/
│   │   ├── graph.py             LangGraph pipeline
│   │   ├── config.py            every tunable, env-driven with defaults
│   │   ├── fetch.py             document retrieval: pdf / arxiv / github / html
│   │   ├── chunker.py           structure-aware chunking with provenance
│   │   ├── relevance.py         deterministic scoring, targets & dimensions
│   │   ├── extract.py           batched evidence extraction + quote verification
│   │   ├── evidence.py          Evidence records, quality tiers, statistics
│   │   ├── references.py        bounded reference discovery
│   │   ├── report.py            deep section-wise synthesis, reference numbering
│   │   ├── report_prompts.py    the original single-call prompt (fallback)
│   │   ├── export.py            markdown export with front matter
│   │   ├── llm.py               retries, fallback chain, health, budgets, mock
│   │   ├── planner.py           question → sub-questions
│   │   ├── searcher.py          SerpApi wrapper, response cache, mock mode
│   │   ├── collector.py         dedupe, freshness, contradiction detection
│   │   └── researcher.py        graph → SSE events
│   ├── tools/                   legacy single-purpose search tools (off-path)
│   └── evidence/                legacy evidence collector (off-path)
├── frontend/
│   ├── index.html
│   ├── app.js                   SSE consumer, progress panels, export
│   ├── markdown.js              escaping markdown renderer, citation anchors
│   └── style.css
├── exports/                     generated markdown reports (gitignored)
├── tests/
│   ├── test_sse.py              SSE contract, ordering, clamping, 422
│   ├── test_llm.py              retry, fallback, cooldowns, 429 classification
│   ├── test_mock_pipeline.py    whole graph on mocks
│   ├── test_deep_research.py    fetch, chunking, evidence, relevance, export
│   ├── test_search_cache.py     search cache
│   ├── test_pipeline.py         graph wiring, collector, live SerpApi
│   └── test_{collector,github,news,serpapi}.py
├── conftest.py                  sys.path setup, `live` marker, env cleanup
├── .env.example
└── requirements.txt
```

`backend/tools/` and `backend/evidence/` are the pre-LangGraph layer. They are still covered by tests but are no longer on the live path.

## Testing

```bash
pytest -q
```

141 tests. All but one run offline. The retrieval and chunking tests stub the HTTP
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
- The cooldown tracker is per-process: it resets on restart, and multiple uvicorn workers would each rediscover the same failing models independently.
- There is no blocked-domain list and no relevance filtering, so low-quality sources reach the model.
- `graph.py` catches broad `Exception` around search calls, which will also mask genuine bugs in that block.
- The report renders as plain text in the browser, so markdown headings appear literally.
- The SerpApi cache rarely hits across runs, because the planner is nondeterministic and generates different queries each time. It reliably prevents duplicate calls within a run.

## Contributing

1. Create a feature branch.
2. Make focused changes.
3. Add or update tests.
4. Run the full suite.
5. Push and open a pull request.

Never commit API keys or other secrets.
