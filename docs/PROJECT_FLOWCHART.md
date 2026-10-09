# Deep Researcher — Project Flowchart

The whole pipeline, stage by stage. Each box explains **why** the stage exists,
not just what it does.

```text
╔══════════════════════════════════════════════════════════════════════╗
║  YOU: ask one question in the browser                                ║
║  Motive: one seed in, a cited research report out.                   ║
╚══════════════════════════════════╤═══════════════════════════════════╝
                                   ▼
┌──────────────────────────────────────────────────────────────────────┐
│  1. PLANNER                                     [one model call]     │
│  Split your question into 10–14 clearly different sub-questions,     │
│  each with 2–3 concrete search queries and the best source per       │
│  query (web / news / scholar / github / patent / arxiv).             │
│  Motive: a big question is unsearchable as-is. Many narrow, distinct │
│  angles = broad coverage with no wasted duplicate work.              │
│  Guard: duplicate filter + LLM regeneration if two angles overlap.   │
└──────────────────────────────────╤───────────────────────────────────┘
                                   ▼
┌──────────────────────────────────────────────────────────────────────┐
│  2. SEARCHERS (all at once)                    [SerpApi + arXiv]     │
│  Every query fans out in parallel. Each response also yields free    │
│  extras: related questions (later used to find gaps), knowledge-     │
│  graph entities (used to spot wrong-topic results), and AI-Overview  │
│  reference links (discovery only — never its prose).                 │
│  Motive: breadth. Many independent sources beat one deep crawl.      │
│  Guard: hard timeout per call, so one hung socket can't freeze the   │
│  whole run; social/video hosts filtered out.                         │
└──────────────────────────────────╤───────────────────────────────────┘
                                   ▼
┌──────────────────────────────────────────────────────────────────────┐
│  3. COLLECT                                                          │
│  Deduplicate results by URL, flag stale sources, keep the metadata   │
│  tidy (title, date, snippet, type).                                  │
│  Motive: the same page often arrives from three queries. Pay for it  │
│  once, cite it once.                                                 │
└──────────────────────────────────╤───────────────────────────────────┘
                                   ▼
┌──────────────────────────────────────────────────────────────────────┐
│  4. RETRIEVE                                       [no model]        │
│  Actually download the real documents behind each result — PDFs,     │
│  papers, ebooks, repos, patents — and parse them into clean,         │
│  structure-aware chunks. If a page is blocked or paywalled, try,     │
│  cheapest first: free open-access copy → Tavily → Firecrawl.         │
│  Wikipedia gives a free baseline before anything else.               │
│  Motive: snippets lie by omission. The report must rest on documents │
│  that were really read — and every source keeps a grade saying       │
│  exactly how much of it was actually retrieved.                      │
└──────────────────────────────────╤───────────────────────────────────┘
                                   ▼
┌──────────────────────────────────────────────────────────────────────┐
│  5. ANALYSE                                    [model, batched]      │
│  Rank all chunks against your question (purely by math — term        │
│  overlap, entities, freshness), then extract findings in batched     │
│  calls. Each finding must quote the passage verbatim or it is        │
│  thrown away. Each keeps its source, section, page and quality tier. │
│  Motive: turn a pile of webpages into a small set of checkable       │
│  claims. The quote-verification is what makes citations real         │
│  instead of decorative.                                              │
└──────────────────────────────────╤───────────────────────────────────┘
                                   ▼
┌──────────────────────────────────────────────────────────────────────┐
│  6. GAP CHECK                                  [one model call]      │
│  "Given what we found, what's still missing?" It names concrete      │
│  holes (a thin dimension, no primary source, nothing recent) and     │
│  asks for targeted follow-up searches — including real questions     │
│  Google's related-questions suggested that nobody answered yet.      │
│       ┌─────── gaps found and rounds remain ───────┐                 │
│       ▼                                            │                 │
│   go back to step 2 for one more round              │                 │
│   (new evidence only — old findings are never re-bought)             │
│       └────────────────────────────────────────────┘                 │
│  Motive: a one-shot search is a guess. This loop is the difference   │
│  between "we searched" and "we checked, then searched again".        │
└──────────────────────────────────╤───────────────────────────────────┘
                                   │  no more gaps (or round limit)
                                   ▼
┌──────────────────────────────────────────────────────────────────────┐
│  7. CONTRADICTION CHECK                        [model, batched]      │
│  Compare findings and surface genuine disagreements — not "different  │
│  emphasis" but incompatible claims, with the likely cause (different │
│  method, date, conditions).                                          │
│  Motive: honest reports show fights in the evidence; they don't      │
│  average them away.                                                  │
└──────────────────────────────────╤───────────────────────────────────┘
                                   ▼
┌──────────────────────────────────────────────────────────────────────┐
│  8. SYNTHESIS                                  [one call/section]    │
│  Write the report section by section: summary, scope, one section    │
│  per dimension, conflicts, scenarios, decision guide, limitations,   │
│  conclusion. Reference numbers are assigned from real sources        │
│  BEFORE writing — so a citation cannot be invented. If a section's   │
│  model call fails, a fallback resumes it using the sections already  │
│  written (the output pool), so the report continues instead of       │
│  restarting or breaking.                                             │
│  Motive: citations that structurally cannot point at nothing.        │
│  Completion means the evidence was adequate — thin dimensions make   │
│  the report PARTIAL and say so on the cover.                         │
└──────────────────────────────────╤───────────────────────────────────┘
                                   ▼
┌──────────────────────────────────────────────────────────────────────┐
│  9. DELIVER (streaming the whole way)                                │
│  Every stage above streams to your browser live over SSE: results as │
│  they arrive, retrieval stats, gap verdicts, sections as they're     │
│  written. The final report renders as HTML with tappable [n]         │
│  citations. Save PDF compiles the finished run into the styled       │
│  PDF; Download .md gives the raw markdown.                           │
│  Motive: a 40-minute black box is unusable; you watch it think.      │
└──────────────────────────────────────────────────────────────────────┘
```

## The four ideas holding it all together

1. **Everything is traceable.** A claim in the report → an evidence record →
   a verbatim quote → a chunk → a real document → a URL. Nothing can cite a
   source that wasn't read.

2. **Models never supply facts.** They plan, extract, and write — but every
   sentence must trace back to retrieved material. The grounding rule is
   stamped on all 11 evidence-processing prompts.

3. **Cheapest wins first.** Free open-access copies before paid extraction;
   math before model calls; one document bought once, not three times.

4. **Failure is visible, not silent.** Hung search → a failed search entry,
   not a frozen run. Thin evidence → `PARTIAL` on the cover, not a
   confident-sounding lie. Every source carries its retrieval grade into the
   report.

## Stage to code map

| Stage | Main files |
| --- | --- |
| 1. Planner | `backend/research/planner.py` |
| 2. Searchers | `backend/research/searcher.py`, `backend/research/wiki.py` |
| 3. Collect | `backend/research/collector.py` |
| 4. Retrieve | `backend/research/fetch.py`, `oa.py`, `webextract.py`, `chunker.py` |
| 5. Analyse | `backend/research/relevance.py`, `extract.py`, `evidence.py` |
| 6. Gap check | `backend/research/graph.py` (gap_check node) |
| 7. Contradiction check | `backend/research/extract.py` (find_conflicts), `graph.py` |
| 8. Synthesis | `backend/research/report.py`, `output_pool.py`, `markdown_html.py` |
| 9. Deliver | `backend/research/researcher.py`, `runlog.py`, `pdf_report.py`, `backend/main.py`, `frontend/` |

## What runs concurrently, and what must not

The pipeline is not uniformly sequential. Three loops issue independent model
calls, and one loop deliberately does not:

| Loop | Concurrency | Why it is safe |
| --- | --- | --- |
| Extraction batches (`extract.py`) | `LLM_PARALLELISM` at a time | Each batch sends its own passages and validates the reply against exactly those |
| Per-source synthesis (`extract.py`) | `LLM_PARALLELISM` at a time | One source per call, each writing to its own key |
| Report sections (`report.py`) | **serial, on purpose** | Later sections are grounded in the sections already written, via `OutputMemoryPool`. Parallelising this would remove the feature, not just the wait |
| Planner, gap check, contradiction check | serial | Single calls whose output feeds the next decision |

Two invariants make the parallel path safe, and both are load-bearing:

- **Results are reassembled in submission order, never completion order.**
  `evidence_id` is assigned by `enumerate()` after `dedupe()`, so collecting
  calls as they finish would renumber every citation between two runs of the
  same question. `[3]` in one run would be `[7]` in the next.
- **`CallBudget.take()` is atomic.** The check and the increment happen under
  one lock. Without it, threads that both read `remaining = 1` both proceed,
  and the run overspends its own ceiling. A measured replica of the unlocked
  version handed out 27 calls against a budget of 10.

The throttle (`THROTTLE_LLM_MAX_CONCURRENT`, `THROTTLE_LLM_PER_MINUTE`) is the
real governor on what a provider sees; `LLM_PARALLELISM` is how many of those
calls a stage has in flight. Measured on a 40-batch extraction stage: 12/min at
concurrency 1 took 195s, 30/min at concurrency 4 took 78s.

## Progress reporting

`retrieve` and `analyse` are the two longest stages and used to emit nothing
until they finished, so the UI showed a frozen step for minutes. Both now write
progress from inside the stage through LangGraph's `get_stream_writer()`,
delivered as a `custom` stream alongside the usual node updates: a real line per
document as each fetch completes, and per-phase lines for ranking, extraction
and source assessment.

Progress is cosmetic by construction. `_progress()` is a no-op when no stream is
attached, swallows writer failures, and the per-document fetch callback is
optional — a fetcher that predates it still retrieves documents rather than
failing the stage.

## Budgets and ceilings

| Knob | Default | Enforced |
| --- | --- | --- |
| `RESEARCH_LLM_BUDGET` | `25 + 60/rounds` | Per call, atomically. A wave clipped mid-flight still reports the truncation rather than passing off a partial pass as complete |
| `RESEARCH_TIME_BUDGET_S` | 600 | Between stages. The run stops and says why, keeping every document and chunk already retrieved |
| `RESEARCH_FETCH_BUDGET` | 100 | Per run, race-free across fetch worker threads |
| `LOG_LEVEL` | `INFO` | At startup; without it the pipeline's progress lines were emitted into a void |
