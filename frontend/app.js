const form = document.getElementById("research-form");

const questionInput = document.getElementById("question");

const button = document.getElementById("research-button");

const roundsInput = document.getElementById("rounds");

const roundsNote = document.getElementById("rounds-tip");

const throttleToggle = document.getElementById("throttle-toggle");

const throttleNote = document.getElementById("throttle-tip");

const throttleStats = document.getElementById("throttle-stats");

const errorBox = document.getElementById("error");

const results = document.getElementById("results");

const resultQuestion = document.getElementById("result-question");

const evidenceCount = document.getElementById("evidence-count");

const planList = document.getElementById("plan");

const searchList = document.getElementById("searches");

const gapCard = document.getElementById("gap-card");

const gapBox = document.getElementById("gap");

const contradictionCard = document.getElementById("contradiction-card");

const contradictionList = document.getElementById("contradictions");

const report = document.getElementById("report");

const retrievalStats = document.getElementById("retrieval-stats");

const retrievalNote = document.getElementById("retrieval-note");

const retrievalDetails = document.getElementById("retrieval-details");

const retrievalList = document.getElementById("retrieval-list");

const analysisStats = document.getElementById("analysis-stats");

const analysisNotes = document.getElementById("analysis-notes");

const downloadButton = document.getElementById("download-button");

const pdfButton = document.getElementById("pdf-button");

const copyButton = document.getElementById("copy-button");

const reportWords = document.getElementById("report-words");

const statusBanner = document.getElementById("status-banner");

const statusMissing = document.getElementById("status-missing");

const reportLabel = document.getElementById("report-label");

const reportCard = document.querySelector(".report-card");

const tabButtons = [...document.querySelectorAll(".tab")];

const tabPanels = [...document.querySelectorAll(".tab-panel")];

const sourcesCount = document.getElementById("sources-count");

const progressSteps = {};
for (const step of document.querySelectorAll(".progress-step")) {
    progressSteps[step.dataset.step] = step;
}

const progPlanDetail = document.getElementById("prog-plan-detail");
const progSearchTitle = document.getElementById("prog-search-title");
const progSearchDetail = document.getElementById("prog-search-detail");
const progSearchSources = document.getElementById("prog-search-sources");
const progReadTitle = document.getElementById("prog-read-title");
const progReadDetail = document.getElementById("prog-read-detail");
const progAnalyseTitle = document.getElementById("prog-analyse-title");
const progAnalyseDetail = document.getElementById("prog-analyse-detail");
const progGapDetail = document.getElementById("prog-gap-detail");
const progWriteTitle = document.getElementById("prog-write-title");
const progWriteDetail = document.getElementById("prog-write-detail");

const ROUND_LABELS = {
    1: "Quick",
    2: "Standard",
    3: "Deep",
    4: "Exhaustive",
};

let currentQuestion = "";

let currentMarkdown = "";

let progressiveHtml = "";

let progressSearchSeen = new Set();
let progressSearchResults = 0;
let progressSections = 0;


// --- run timer -------------------------------------------------------------
//
// Elapsed is exact. Remaining is an estimate, and it is built from what this
// run has actually done rather than from a fixed guess: each stage's first
// event starts its clock, and when a stage finishes its measured duration
// replaces the prior for any later round. A prior with nothing behind it is a
// guess, so the label says "estimating…" until the first stage completes.

const TIMER_STAGES = ["plan", "search", "retrieve", "analyse", "write"];

// Seconds, taken from the stage boundaries of a real logged run (plan 43s,
// search 136s, retrieve 59s, analyse 235s across two rounds, write 93s), so a
// cold start lands within ~10% before anything has been measured here. Each
// stage's own measurement replaces its prior as soon as it finishes.
const TIMER_PRIORS = {
    plan: 45, search: 130, retrieve: 95, analyse: 210, write: 140,
};
const TIMER_EXPECTED_SECTIONS = 18;

let timerState = null;
let timerHandle = null;

const runTimer = document.getElementById("run-timer");
const runTimerElapsed = document.getElementById("run-timer-elapsed");
const runTimerRemaining = document.getElementById("run-timer-remaining");
const runTimerFill = document.getElementById("run-timer-fill");


function timerReset() {
    timerState = {
        startedAt: Date.now(),
        active: null,
        activeSince: null,
        measured: {},
        completed: {},
        fraction: {},
        rounds: 1,
        extraRounds: 0,
        finished: false,
    };
    if (timerHandle) {
        clearInterval(timerHandle);
    }
    timerHandle = setInterval(timerTick, 1000);
    timerTick();
}


function timerFormat(seconds) {
    const total = Math.max(0, Math.round(seconds));
    if (total < 60) {
        return `${total}s`;
    }
    const mins = Math.floor(total / 60);
    const secs = total % 60;
    if (mins < 60) {
        return `${mins}m ${String(secs).padStart(2, "0")}s`;
    }
    return `${Math.floor(mins / 60)}h ${String(mins % 60).padStart(2, "0")}m`;
}


/** Mark a stage as the active one, timing the transition. */
function timerStage(name) {
    if (!timerState || timerState.finished) {
        return;
    }
    if (timerState.active === name) {
        return;
    }
    timerFinishStage();
    timerState.active = name;
    timerState.activeSince = Date.now();
}


function timerFinishStage() {
    // `activeSince == null` rather than a falsy check: it is a timestamp, and 0
    // is a legitimate one. A falsy guard silently refuses to close a stage that
    // began at the epoch, which leaves its prior in the estimate forever.
    if (!timerState || !timerState.active || timerState.activeSince == null) {
        return;
    }
    const spent = (Date.now() - timerState.activeSince) / 1000;
    const name = timerState.active;
    // Longest observed wins: a stage interrupted by a round gap should not
    // overwrite a longer, cleaner measurement with a short one.
    if (spent > (timerState.measured[name] || 0)) {
        timerState.measured[name] = spent;
    }
    timerState.completed[name] = true;
    timerState.active = null;
    timerState.activeSince = null;
}


/** How far through the active stage we are, 0..1. */
function timerFraction(name) {
    const f = timerState && timerState.fraction[name];
    if (typeof f === "number") {
        return Math.max(0, Math.min(1, f));
    }
    return 0;
}


function timerRemainingSeconds() {
    if (!timerState) {
        return null;
    }
    const extraRounds = timerState.extraRounds;
    let remaining = 0;
    for (const stage of TIMER_STAGES) {
        if (timerState.completed[stage]) {
            continue;
        }
        const prior = timerState.measured[stage] || TIMER_PRIORS[stage];
        // retrieve and analyse run once per round; extra gap rounds repeat them.
        const repeats = (stage === "retrieve" || stage === "analyse")
            ? 1 + extraRounds : 1;
        if (timerState.active === stage) {
            remaining += prior * (1 - timerFraction(stage)) * repeats;
        } else {
            remaining += prior * repeats;
        }
    }
    return Math.max(0, remaining);
}


function timerTotalSeconds() {
    if (!timerState) {
        return null;
    }
    const elapsed = (Date.now() - timerState.startedAt) / 1000;
    const remaining = timerRemainingSeconds();
    return remaining === null ? null : elapsed + remaining;
}


function timerTick() {
    if (!timerState || !runTimer) {
        return;
    }
    runTimer.hidden = false;

    const elapsed = (Date.now() - timerState.startedAt) / 1000;
    if (runTimerElapsed) {
        runTimerElapsed.textContent = timerFormat(elapsed);
    }

    // Only claim an estimate once something has actually been timed; before
    // that the priors are guesses and should read as guesses.
    const timed = Object.keys(timerState.measured).length > 0
        || timerState.completed.plan;
    if (timerState.finished) {
        if (runTimerRemaining) {
            runTimerRemaining.textContent = "done";
        }
    } else if (timed) {
        const remaining = timerRemainingSeconds();
        if (runTimerRemaining) {
            runTimerRemaining.textContent = `~${timerFormat(remaining)}`;
        }
        const total = elapsed + remaining;
        if (runTimerFill && total > 0) {
            runTimerFill.style.width =
                `${Math.min(100, Math.max(0, (elapsed / total) * 100))}%`;
        }
    } else if (runTimerRemaining) {
        runTimerRemaining.textContent = "estimating…";
    }
}


function timerFinish(totalSeconds) {
    if (!timerState) {
        return;
    }
    timerFinishStage();
    timerState.finished = true;
    if (timerHandle) {
        clearInterval(timerHandle);
        timerHandle = null;
    }
    if (!runTimer) {
        return;
    }
    runTimer.classList.add("is-done");
    if (runTimerElapsed) {
        // Prefer the server's own measurement: it covers the whole run,
        // whereas the browser timer starts when the page did.
        runTimerElapsed.textContent = timerFormat(
            typeof totalSeconds === "number" && totalSeconds > 0
                ? totalSeconds : (Date.now() - timerState.startedAt) / 1000);
    }
    if (runTimerRemaining) {
        runTimerRemaining.textContent = "done";
    }
    if (runTimerFill) {
        runTimerFill.style.width = "100%";
    }
}


function selectTab(name, focusTab) {
    for (const tab of tabButtons) {
        const active = tab.dataset.tab === name;
        tab.classList.toggle("is-active", active);
        tab.setAttribute("aria-selected", active ? "true" : "false");
        tab.tabIndex = active ? 0 : -1;
        if (active && focusTab) {
            tab.focus();
        }
    }
    for (const panel of tabPanels) {
        const active = panel.id === `panel-${name}`;
        panel.classList.toggle("is-active", active);
        if (active) {
            panel.removeAttribute("hidden");
        } else {
            panel.setAttribute("hidden", "");
        }
    }
}


for (const tab of tabButtons) {
    tab.addEventListener("click", () => selectTab(tab.dataset.tab, false));
}

const tabList = document.querySelector(".tabs");
if (tabList) {
    tabList.addEventListener("keydown", (event) => {
        if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") {
            return;
        }
        event.preventDefault();
        const current = tabButtons.indexOf(document.activeElement);
        const next = event.key === "ArrowRight"
            ? (current + 1) % tabButtons.length
            : (current - 1 + tabButtons.length) % tabButtons.length;
        selectTab(tabButtons[next].dataset.tab, true);
    });
}


function setStepState(step, state) {
    const row = progressSteps[step];
    if (!row) {
        return;
    }
    row.classList.remove("is-todo", "is-active", "is-done");
    row.classList.add(`is-${state}`);
}


function resetProgress() {
    progressSearchSeen = new Set();
    progressSearchResults = 0;
    progressSections = 0;
    // A new run must not inherit the previous one's timings or its finished
    // styling, or the timer opens already reading "done".
    if (timerHandle) {
        clearInterval(timerHandle);
        timerHandle = null;
    }
    timerState = null;
    if (runTimer) {
        runTimer.hidden = true;
        runTimer.classList.remove("is-done", "is-stalled");
        if (runTimerFill) {
            runTimerFill.style.width = "0%";
        }
    }
    for (const step of Object.keys(progressSteps)) {
        setStepState(step, "todo");
    }
    if (progressSteps.gap) {
        progressSteps.gap.classList.add("hidden");
    }
    if (progPlanDetail) {
        progPlanDetail.textContent = "Waiting…";
    }
    if (progSearchTitle) {
        progSearchTitle.textContent = "Searched 0 queries";
    }
    if (progSearchDetail) {
        progSearchDetail.textContent = "Waiting for the first results…";
    }
    if (progSearchSources) {
        progSearchSources.textContent = "";
    }
    if (progReadTitle) {
        progReadTitle.textContent = "Read 0 pages";
    }
    if (progReadDetail) {
        progReadDetail.textContent = "Waiting for downloads…";
    }
    if (progAnalyseTitle) {
        progAnalyseTitle.textContent = "Extracting evidence";
    }
    if (progAnalyseDetail) {
        progAnalyseDetail.textContent = "Waiting for passages…";
    }
    if (progWriteTitle) {
        progWriteTitle.textContent = "Writing the report";
    }
    if (progWriteDetail) {
        progWriteDetail.textContent = "Waiting for sections…";
    }
    selectTab("report", false);
}


function progressPlan(plan) {
    const subs = (plan && plan.subquestions) || [];
    setStepState("plan", "done");
    if (progPlanDetail) {
        progPlanDetail.textContent = subs.length
            ? `${subs.length} sub-question${subs.length === 1 ? "" : "s"} planned`
            : "No sub-questions planned";
    }
    setStepState("search", "active");
}


function progressSearch(data) {
    setStepState("search", "active");
    progressSearchResults += data.error ? 0 : (data.count || 0);
    const queries = searchList.children.length;
    if (progSearchTitle) {
        progSearchTitle.textContent =
            `Searched ${queries} quer${queries === 1 ? "y" : "ies"}, `
            + `${progressSearchResults} results`;
    }
    if (progSearchDetail) {
        progSearchDetail.textContent = data.error
            ? `${data.source || "search"} failed: ${data.error}`
            : `Latest: ${data.source || "search"} returned ${data.count || 0}`;
    }
    if (progSearchSources && data.source
        && !progressSearchSeen.has(data.source)) {
        progressSearchSeen.add(data.source);
        const tag = document.createElement("span");
        tag.className = "plan-source";
        tag.textContent = data.source;
        progSearchSources.appendChild(tag);
    }
}


function progressLive(data) {
    // Mid-stage progress from retrieve/analyse. These two stages take minutes
    // and used to emit nothing until they finished, so the browser showed a
    // frozen step for the whole of it with no sign anything was happening.
    if (!data) {
        return;
    }
    const stage = data.stage;
    const note = data.note || "";
    const done = Number(data.done || 0);
    const total = Number(data.total || 0);

    if (stage === "retrieve") {
        setStepState("search", "done");
        setStepState("read", "active");
        if (progReadTitle && total) {
            progReadTitle.textContent = `Reading ${done} of ${total}`;
        }
        if (progReadDetail && note) {
            progReadDetail.textContent = note;
        }
    }

    if (stage === "analyse") {
        setStepState("read", "done");
        setStepState("analyse", "active");
        if (progAnalyseTitle && total) {
            progAnalyseTitle.textContent = done === 0
                ? `Analysing ${total} passages`
                : `Analysed ${done} of ${total}`;
        }
        if (progAnalyseDetail && note) {
            progAnalyseDetail.textContent = note;
        }
    }
}


function progressRead(data) {
    if (!data || data.enabled === false) {
        return;
    }
    setStepState("search", "done");
    setStepState("read", "active");
    if (progReadTitle) {
        progReadTitle.textContent = `Read ${data.retrieved || 0} pages`;
    }
    if (progReadDetail) {
        progReadDetail.textContent =
            `${data.full_text || 0} full · ${data.partial || 0} partial · `
            + `${data.metadata_only || 0} snippet-only`;
    }
}


function progressGap(gap) {
    if (!gap) {
        return;
    }
    const row = progressSteps.gap;
    const missing = (gap.missing || "").toLowerCase();
    const hasFollowUps = (gap.follow_ups || []).length > 0;
    if (gap.sufficient || missing.includes("round limit")) {
        if (row) {
            row.classList.add("hidden");
        }
        setStepState("read", "done");
        return;
    }
    if (row) {
        row.classList.remove("hidden");
    }
    setStepState("read", "done");
    if (!hasFollowUps) {
        // A gap with nothing to chase is still being assessed.
        setStepState("gap", "active");
        if (progGapDetail) {
            progGapDetail.textContent = gap.missing || "Checking coverage…";
        }
        return;
    }
    // Once the follow-ups are dispatched this step has finished its job. It
    // used to stay "active" for the whole of the next round -- several minutes
    // of spinner with nothing behind it, and then never ticked even after the
    // report landed. The chasing is now shown by the search step going active
    // again, which is what is actually happening.
    setStepState("gap", "done");
    if (progGapDetail) {
        progGapDetail.textContent =
            `Chased ${gap.follow_ups.length} follow-up${gap.follow_ups.length === 1 ? "" : "s"}: ${gap.missing || ""}`;
    }
    setStepState("search", "active");
}


function progressWrite() {
    progressSections += 1;
    // Analysis is finished by the time sections stream, so mark it done --
    // otherwise the evidence step sits "active" next to a finished report.
    setStepState("analyse", "done");
    setStepState("write", "active");
    if (progWriteTitle) {
        progWriteTitle.textContent = "Writing the report";
    }
    if (progWriteDetail) {
        progWriteDetail.textContent =
            `${progressSections} section${progressSections === 1 ? "" : "s"} streamed`;
    }
}


function progressDone() {
    for (const step of Object.keys(progressSteps)) {
        const row = progressSteps[step];
        if (row && !row.classList.contains("hidden")) {
            setStepState(step, "done");
        }
    }
    if (progWriteDetail && progressSections === 0) {
        progWriteDetail.textContent = "Report ready";
    }
}


// The round ceiling is a server setting, so the options are fetched rather than
// hardcoded here. Without this the UI silently caps at whatever it shipped with.
async function loadRoundOptions() {
    if (!roundsInput) {
        return;
    }
    const fallback = [1, 2, 3, 4];
    let cfg = { rounds_ceiling: 4, rounds_default: 2, budget_estimate: {} };
    try {
        const response = await fetch("/api/config");
        if (response.ok) {
            cfg = await response.json();
        }
    } catch (error) {
        // Backend unreachable: fall back to the conservative set.
    }

    const ceiling = Math.max(1, Math.min(cfg.rounds_ceiling || 4, 10));
    roundsInput.textContent = "";
    for (let n = 1; n <= ceiling; n += 1) {
        const option = document.createElement("option");
        option.value = String(n);
        option.textContent =
            (ROUND_LABELS[n] || `Round ${n}`) + ` (${n} round${n > 1 ? "s" : ""})`;
        roundsInput.appendChild(option);
    }
    const preferred = Math.min(cfg.rounds_default || 2, ceiling);
    roundsInput.value = String(preferred);

    if (roundsNote) {
        const estimates = cfg.budget_estimate || {};
        const parts = [];
        for (let n = 2; n <= Math.min(ceiling, 4); n += 1) {
            if (estimates[String(n)]) {
                parts.push(`${n} rounds ≈ ${estimates[String(n)]} model calls`);
            }
        }
        if (parts.length) {
            roundsNote.textContent =
                "Deeper runs cost more model calls per round: " +
                parts.join(", ") +
                ". The budget scales automatically unless RESEARCH_LLM_BUDGET is set.";
            const info = roundsNote.closest(".info");
            if (info) {
                info.classList.add("has-tip");
            }
        }
    }
}

function renderThrottleStats(payload) {
    if (!throttleStats || !payload) {
        return;
    }
    const live = payload.llm || payload;
    throttleStats.textContent = "";

    const tiles = [
        [live.enabled ? "on" : "off", "pacing", "model calls"],
        [live.per_minute + "/min", "rate", "configured ceiling"],
        [live.calls || 0, "calls sent", "this process"],
        [Math.round(live.waited_seconds || 0) + "s", "deliberate waiting",
         "held back to stay under quota"],
    ];
    for (const [value, label, hint] of tiles) {
        throttleStats.appendChild(statTile(value, label, hint));
    }
}


function loadThrottleState() {
    if (!throttleToggle || !throttleNote) {
        return Promise.resolve();
    }
    return fetch("/api/throttle")
        .then((response) => (response.ok ? response.json() : null))
        .then((body) => {
            if (!body || !body.configured) {
                return;
            }
            const cfg = body.configured;
            throttleToggle.value = cfg.enabled ? "1" : "0";
            if (cfg.enabled) {
                const spacing = cfg.per_minute > 0
                    ? (60 / cfg.per_minute).toFixed(1)
                    : "0";
                throttleNote.textContent =
                    `Pacing model calls to ${cfg.per_minute}/min `
                    + `(~${spacing}s apart, ${cfg.max_concurrent} at a time). `
                    + "Slower, but far less likely to exhaust a free-tier quota. "
                    + "Choose Unthrottled once you are on paid plans with headroom.";
            } else {
                throttleNote.textContent =
                    "Pacing is off. Calls go out as fast as the pipeline allows, "
                    + "which risks exhausting free-tier quotas.";
            }
            const info = throttleNote.closest(".info");
            if (info) {
                info.classList.add("has-tip");
            }
        })
        .catch(() => { /* backend unreachable: keep the default selection */ });
}


loadRoundOptions();
loadThrottleState();


// A fresh invitation on every boot: pure random pick, not tied to any
// counter, so repeats carry no pattern.
const PLACEHOLDERS = [
    "What complex problem can we untangle together?",
    "Got a tough technical challenge? Let's debug it.",
    "Let's brainstorm some fresh ideas...",
    "What's on your mind?",
    "Which decision needs real evidence behind it?",
    "Pick a debate — I'll gather the facts.",
    "What technology should I compare for you?",
    "Curious about something? Let's research it properly.",
    "Give me a rabbit hole worth going down.",
    "What should we investigate today?",
    "Two options, one choice — I'll do the homework.",
    "Ask the question Google can't answer in one link.",
];

if (questionInput) {
    questionInput.placeholder = PLACEHOLDERS[
        Math.floor(Math.random() * PLACEHOLDERS.length)
    ];
}


form.addEventListener("submit", async (event) => {
    event.preventDefault();

    const question = questionInput.value.trim();

    if (!question) {
        return;
    }

    currentQuestion = question;
    currentMarkdown = "";
    progressiveHtml = "";

    errorBox.classList.add("hidden");
    results.classList.remove("hidden");

    resetResults();
    report.classList.add("is-loading");

    startResearchButton();

    try {
        const rounds = roundsInput ? roundsInput.value : "2";
        const throttle = throttleToggle ? throttleToggle.value : "1";

        const response = await fetch(
            `/api/research?q=${encodeURIComponent(question)}`
            + `&rounds=${encodeURIComponent(rounds)}`
            + `&throttle=${encodeURIComponent(throttle)}`
        );

        if (!response.ok) {
            throw new Error("Research request failed.");
        }

        if (!response.body) {
            throw new Error("Research stream is unavailable.");
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();

        let buffer = "";
        let finished = false;

        while (!finished) {
            const { value, done } = await reader.read();

            if (done) {
                break;
            }

            buffer += decoder.decode(value, { stream: true });

            const events = buffer.split("\n\n");
            buffer = events.pop();

            for (const eventText of events) {
                const line = eventText
                    .split("\n")
                    .find((line) => line.startsWith("data: "));

                if (!line) {
                    continue;
                }

                let data;
                try {
                    data = JSON.parse(line.slice(6));
                } catch (parseError) {
                    continue;
                }

                if (data.type === "error") {
                    throw new Error(data.message || "Research failed.");
                }

                if (data.type === "done") {
                    finished = true;

                    break;
                }

                displayEvent(data, question);
            }
        }
    } catch (error) {
        errorBox.textContent = error.message;
        errorBox.classList.remove("hidden");
        report.classList.remove("is-loading");
        snapRetrieval();
    } finally {
        stopResearchButton();
    }
});


const RESEARCH_MESSAGES = [
    "Collecting Ideas From The Internet",
    "Reading Source Documents",
    "Extracting Evidence",
    "Checking For Gaps",
    "Resolving Contradictions",
    "Writing Your Report",
];

let researchTimer = null;


function startResearchButton() {
    button.disabled = true;
    button.classList.add("is-researching");
    let index = 0;
    const show = () => {
        button.innerHTML = "";
        const msg = document.createElement("span");
        msg.className = "research-msg";
        msg.textContent = RESEARCH_MESSAGES[index % RESEARCH_MESSAGES.length];
        const dots = document.createElement("span");
        dots.className = "dots";
        dots.setAttribute("aria-hidden", "true");
        for (let i = 0; i < 3; i += 1) {
            const dot = document.createElement("i");
            dot.textContent = ".";
            dots.appendChild(dot);
        }
        button.appendChild(msg);
        button.appendChild(dots);
        index += 1;
    };
    show();
    researchTimer = setInterval(show, 2600);
}


function stopResearchButton() {
    if (researchTimer) {
        clearInterval(researchTimer);
        researchTimer = null;
    }
    button.classList.remove("is-researching");
    button.disabled = false;
    button.textContent = "Start Research →";
}


function resetResults() {
    planList.textContent = "";
    searchList.textContent = "";
    contradictionList.textContent = "";
    evidenceCount.textContent = "";
    report.textContent = "";
    reportWords.textContent = "";
    retrievalStats.textContent = "";
    for (const key of Object.keys(retrievalTiles)) {
        if (retrievalTiles[key].timer) {
            clearInterval(retrievalTiles[key].timer);
        }
    }
    retrievalTiles = {};
    for (const key in retrievalDisplayed) {
        retrievalDisplayed[key] = 0;
    }
    for (const key in retrievalTarget) {
        retrievalTarget[key] = 0;
    }
    retrievalList.textContent = "";
    analysisStats.textContent = "";
    analysisNotes.textContent = "";
    throttleStats.textContent = "";

    retrievalDetails.classList.add("hidden");

    gapBox.textContent = "";
    gapCard.classList.add("hidden");
    contradictionCard.classList.add("hidden");

    downloadButton.classList.add("hidden");
    copyButton.classList.add("hidden");
    pdfButton.classList.add("hidden");
    const pdfLoading = document.getElementById("pdf-loading");
    if (pdfLoading) {
        pdfLoading.classList.add("hidden");
    }

    statusBanner.textContent = "";
    statusBanner.className = "status-banner hidden";
    statusMissing.textContent = "";
    statusMissing.classList.add("hidden");

    if (reportCard) {
        reportCard.classList.remove("is-partial", "is-failed");
    }
    if (reportLabel) {
        reportLabel.textContent = "FINAL REPORT";
    }

    resetProgress();
}


function displayEvent(data, question) {
    resultQuestion.textContent = question;

    if (data.type === "plan") {
        timerReset();
        timerStage("plan");
        renderPlan(data.data);
        progressPlan(data.data);
    }

    if (data.type === "results") {
        if (!timerState) {
            timerReset();
        }
        timerStage("search");
        renderSearch(data);
        progressSearch(data);
    }

    if (data.type === "evidence") {
        evidenceCount.textContent = `${data.count} sources collected`;
        if (sourcesCount) {
            sourcesCount.textContent = String(data.count);
        }
    }

    if (data.type === "retrieval") {
        renderRetrieval(data.data);
        progressRead(data.data);
    }

    if (data.type === "analysis") {
        renderAnalysis(data.data);
    }

    if (data.type === "progress") {
        if (!timerState) {
            timerReset();
        }
        if (data.stage === "retrieve") {
            timerStage("retrieve");
            if (data.total) {
                timerState.fraction.retrieve = (data.done || 0) / data.total;
            }
        }
        if (data.stage === "analyse") {
            timerStage("analyse");
            if (data.total) {
                timerState.fraction.analyse = (data.done || 0) / data.total;
            }
        }
        progressLive(data);
    }

    if (data.type === "gap") {
        if ((data.data.follow_ups || []).length && timerState) {
            // Another round means retrieve and analyse will run again, so the
            // estimate has to carry a second pass or it collapses to zero while
            // the pipeline is plainly still working.
            timerState.extraRounds = Math.min(3, (timerState.extraRounds || 0) + 1);
        }
        renderGap(data.data);
        progressGap(data.data);
    }

    if (data.type === "contradictions") {
        renderContradictions(data.data);
    }

    if (data.type === "status") {
        renderStatus(data.data);
    }

    if (data.throttle) {
        renderThrottleStats(data.throttle);
    }

    if (data.type === "report_section") {
        timerStage("write");
        if (timerState) {
            // progressWrite() owns the counter; read it after the call below
            // would be too late, so derive the fraction from its own total.
            timerState.fraction.write = Math.min(
                1, (progressSections + 1) / TIMER_EXPECTED_SECTIONS);
        }
        renderReportSection(data);
        progressWrite();
    }

    if (data.type === "report") {
        if (data.status) {
            renderStatus(data.status);
        }
        // The server stamps the true run duration when the report lands; fall
        // back to the browser's own count if it is somehow absent.
        timerFinish(data.elapsed_s);
        renderReport(data);
        progressDone();
    }

    if (data.type === "done") {
        // Covers a run that ended without a report body: stop the clock rather
        // than leaving it counting forever.
        timerFinish();
    }

    results.scrollIntoView({
        behavior: "smooth",
        block: "start"
    });
}


// Flowing retrieval counters: tiles persist across events and tick upward in
// small random steps toward each new cumulative total, so the dashboard reads
// as a pipeline in motion rather than jumping between fixed values.
const retrievalDisplayed = {
    retrieved: 0,
    attempted: 0,
    full: 0,
    partial: 0,
    metadata: 0,
    words: 0,
    chunks: 0,
    references: 0,
};

let retrievalTiles = {};

const retrievalTarget = {
    retrieved: 0,
    attempted: 0,
    full: 0,
    partial: 0,
    metadata: 0,
    words: 0,
    chunks: 0,
    references: 0,
};

const retrievalFormat = {};


function flowTile(key, label, hint) {
    let entry = retrievalTiles[key];
    if (!entry) {
        const tile = document.createElement("div");
        tile.className = "stat-tile";
        const valueEl = document.createElement("strong");
        valueEl.textContent = "0";
        const labelEl = document.createElement("span");
        labelEl.textContent = label;
        tile.appendChild(valueEl);
        tile.appendChild(labelEl);
        if (hint) {
            const hintEl = document.createElement("small");
            hintEl.textContent = hint;
            tile.appendChild(hintEl);
        }
        retrievalStats.appendChild(tile);
        entry = { tile, valueEl, labelEl, timer: null };
        retrievalTiles[key] = entry;
    } else {
        entry.labelEl.textContent = label;
    }
    return entry;
}


function flowNumber(key, target, format) {
    const entry = retrievalTiles[key];
    if (!entry) {
        return;
    }
    target = Number(target) || 0;
    retrievalTarget[key] = target;
    if (format) {
        retrievalFormat[key] = format;
    }
    const fmt = retrievalFormat[key];
    const render = (value) => {
        entry.valueEl.textContent = fmt ? fmt(value) : String(value);
    };
    const reduce = window.matchMedia
        && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    if (reduce || target <= retrievalDisplayed[key]) {
        retrievalDisplayed[key] = target;
        render(target);
        return;
    }
    if (!entry.timer) {
        entry.timer = setInterval(() => crawlNumber(key), 150);
    }
}


function crawlNumber(key) {
    const entry = retrievalTiles[key];
    if (!entry) {
        return;
    }
    const target = retrievalTarget[key] || 0;
    const fmt = retrievalFormat[key];
    const render = (value) => {
        entry.valueEl.textContent = fmt ? fmt(value) : String(value);
    };
    const remaining = target - retrievalDisplayed[key];
    // Hold just short of the target so counters keep moving until research
    // completes; the final snap lands the exact backend totals.
    const holdback = target < 25 ? 1 : Math.round(target * 0.04);
    if (remaining <= holdback) {
        clearInterval(entry.timer);
        entry.timer = null;
        return;
    }
    const hop = Math.max(1, Math.round(remaining * (0.03 + Math.random() * 0.05)));
    retrievalDisplayed[key] += Math.min(hop, remaining - holdback);
    render(retrievalDisplayed[key]);
}


function snapRetrieval() {
    for (const key of Object.keys(retrievalTiles)) {
        const entry = retrievalTiles[key];
        if (entry.timer) {
            clearInterval(entry.timer);
            entry.timer = null;
        }
        const target = retrievalTarget[key] || 0;
        const fmt = retrievalFormat[key];
        retrievalDisplayed[key] = target;
        entry.valueEl.textContent = fmt ? fmt(target) : String(target);
    }
}


function statTile(value, label, hint) {
    const tile = document.createElement("div");

    tile.className = "stat-tile";

    const valueEl = document.createElement("strong");

    valueEl.textContent = String(value);

    const labelEl = document.createElement("span");

    labelEl.textContent = label;

    tile.appendChild(valueEl);
    tile.appendChild(labelEl);

    if (hint) {
        const hintEl = document.createElement("small");

        hintEl.textContent = hint;
        tile.appendChild(hintEl);
    }

    return tile;
}


function renderRetrieval(data) {
    if (!data || data.enabled === false) {
        retrievalNote.textContent =
            (data && data.note) ||
            "Full-text retrieval is disabled; only search snippets were used. "
            + "No document was read, so no finding below rests on document text.";

        return;
    }

    retrievalNote.textContent =
        "Documents were downloaded and parsed, not summarised from snippets. "
        + "Figures are cumulative across the whole research run, so they cover "
        + "every round, not just the most recent one.";

    // Tiles persist across events: each new cumulative total flows upward
    // from the currently displayed value instead of restarting at zero.
    flowTile("retrieved", `of ${data.attempted || 0} attempted`, "documents fetched");
    flowNumber("retrieved", data.retrieved || 0);
    retrievalDisplayed.attempted = data.attempted || 0;
    flowTile("full", "full text", "readable end to end");
    flowNumber("full", data.full_text || 0);
    flowTile("partial", "partial", "paywall, truncation or PDF limits");
    flowNumber("partial", data.partial || 0);
    flowTile("metadata", "metadata only", "no document text");
    flowNumber("metadata", data.metadata_only || 0);
    flowTile("words", "words", "retrieved in total");
    flowNumber("words", data.words || 0, formatNumber);
    flowTile("chunks", "chunks", "section-aware passages");
    flowNumber("chunks", data.chunks || 0);

    if (data.references_followed) {
        flowTile("references", "references", "followed from sources");
        flowNumber("references", data.references_followed);
    }

    const sources = data.sources || [];

    if (sources.length) {
        retrievalList.textContent = "";

        for (const source of sources) {
            const row = document.createElement("div");

            row.className = `retrieval-row status-${source.status || "unknown"}`;

            const badge = document.createElement("span");

            badge.className = "status-badge";
            badge.textContent = (source.status || "unknown").replace("_", " ");

            const title = document.createElement("a");

            title.href = safeExternalUrl(source.url);
            title.target = "_blank";
            title.rel = "noopener noreferrer";
            title.textContent = source.title || "(untitled)";

            const meta = document.createElement("small");

            meta.textContent = [
                source.method,
                `${formatNumber(source.words || 0)} words`,
                source.limitation
            ].filter(Boolean).join(" · ");

            row.appendChild(badge);
            row.appendChild(title);
            row.appendChild(meta);

            retrievalList.appendChild(row);
        }

        retrievalDetails.classList.remove("hidden");
        retrievalDetails.open = true;
    }
}


function renderAnalysis(data) {
    analysisStats.textContent = "";

    if (!data) {
        return;
    }

    analysisStats.appendChild(
        statTile(data.records || 0, "findings", "extracted and verified")
    );

    const dimensions = data.dimensions || [];

    analysisStats.appendChild(
        statTile(dimensions.length, "dimensions", "covered by evidence")
    );

    if (data.sources_summarised) {
        analysisStats.appendChild(
            statTile(data.sources_summarised, "sources", "individually assessed")
        );
    }

    const notes = data.notes || [];

    analysisNotes.textContent = "";

    for (const note of notes) {
        const item = document.createElement("li");

        item.textContent = note;
        analysisNotes.appendChild(item);
    }
}


function renderPlan(plan) {
    planList.textContent = "";

    const subquestions = (plan && plan.subquestions) || [];

    if (!subquestions.length) {
        const li = document.createElement("li");

        li.textContent = "No sub-questions were planned.";
        planList.appendChild(li);

        return;
    }

    for (const subquestion of subquestions) {
        const li = document.createElement("li");

        li.textContent = subquestion.question || "";

        const searches = subquestion.searches || [];

        if (searches.length) {
            const tags = document.createElement("div");

            tags.className = "plan-sources";

            for (const search of searches) {
                const tag = document.createElement("span");

                tag.className = "plan-source";
                tag.textContent = search.source || "web";

                tags.appendChild(tag);
            }

            li.appendChild(tags);
        }

        planList.appendChild(li);
    }
}


function renderSearch(data) {
    const item = document.createElement("article");

    item.className = "finding";

    if (data.error) {
        item.classList.add("finding-error");
    }

    const number = document.createElement("span");

    number.className = "finding-number";
    number.textContent = String(
        searchList.children.length + 1
    ).padStart(2, "0");

    const body = document.createElement("div");

    const heading = document.createElement("h5");

    heading.textContent = data.query || "(no query)";

    const detail = document.createElement("p");

    detail.textContent = data.error
        ? `${data.source || "search"} failed: ${data.error}`
        : `${data.source || "search"} returned ${data.count} result${
              data.count === 1 ? "" : "s"
          }`;

    body.appendChild(heading);
    body.appendChild(detail);

    item.appendChild(number);
    item.appendChild(body);

    searchList.appendChild(item);
}


function renderGap(gap) {
    if (!gap) {
        return;
    }

    gapBox.textContent = "";

    const line = document.createElement("p");

    line.className = "gap-line";

    if (gap.sufficient) {
        line.classList.add("is-sufficient");
        line.textContent = "Evidence looks sufficient.";
    } else if ((gap.missing || "").toLowerCase().includes("round limit")) {
        line.classList.add("is-complete");
        line.textContent = "Research complete — round limit reached.";
    } else {
        line.textContent = "Still researching.";
    }

    gapBox.appendChild(line);

    if (gap.missing
        && !(gap.missing || "").toLowerCase().includes("round limit")) {
        const missing = document.createElement("p");

        missing.className = "gap-missing";
        missing.textContent = `Missing: ${gap.missing}`;

        gapBox.appendChild(missing);
    }

    const followUps = gap.follow_ups || [];

    if (followUps.length) {
        const list = document.createElement("ul");

        list.className = "follow-ups";

        for (const followUp of followUps) {
            const li = document.createElement("li");

            li.textContent = `${followUp.source} — ${followUp.query}`;

            list.appendChild(li);
        }

        gapBox.appendChild(list);
    }

    gapCard.classList.remove("hidden");
}


function renderContradictions(contradictions) {
    if (!Array.isArray(contradictions) || !contradictions.length) {
        return;
    }

    contradictionList.textContent = "";

    for (const item of contradictions) {
        const row = document.createElement("article");

        row.className = "finding";

        const body = document.createElement("div");

        const heading = document.createElement("h5");

        heading.textContent = item.topic || "Unclear topic";

        body.appendChild(heading);

        if (item.side_a) {
            const sideA = document.createElement("p");

            sideA.innerHTML =
                "<strong>A:</strong> " + window.escapeHtml(item.side_a);
            body.appendChild(sideA);
        }

        if (item.side_b) {
            const sideB = document.createElement("p");

            sideB.innerHTML =
                "<strong>B:</strong> " + window.escapeHtml(item.side_b);
            body.appendChild(sideB);
        }

        if (item.likely_cause) {
            const cause = document.createElement("p");

            cause.className = "gap-missing";
            cause.textContent = `Likely cause: ${item.likely_cause}`;
            body.appendChild(cause);
        }

        row.appendChild(body);

        contradictionList.appendChild(row);
    }

    contradictionCard.classList.remove("hidden");
}


const STATUS_TEXT = {
    completed: "COMPLETE REPORT",
    partial: "PARTIAL REPORT",
    failed: "REPORT FAILED"
};

const STATUS_HINT = {
    completed: "All planned sections were generated.",
    partial: "This research run did not finish. Treat the report as incomplete.",
    failed: "No usable report could be generated."
};


function renderStatus(status) {
    if (!status || !status.status) {
        return;
    }

    const state = status.status;

    statusBanner.className = "status-banner state-" + state;
    statusBanner.textContent = status.headline || STATUS_TEXT[state] || state;

    const hint = document.createElement("small");

    hint.textContent = STATUS_HINT[state] || "";
    statusBanner.appendChild(hint);

    const missing = status.missing_sections || [];

    if (missing.length) {
        statusMissing.textContent = "";
        for (const item of missing) {
            const li = document.createElement("li");
            li.textContent = item;
            statusMissing.appendChild(li);
        }
        statusMissing.classList.remove("hidden");
    } else {
        statusMissing.textContent = "";
        statusMissing.classList.add("hidden");
    }

    statusBanner.classList.remove("hidden");

    // A partial or failed run must not look like an ordinary finished report.
    if (reportCard) {
        reportCard.classList.toggle("is-partial", state === "partial");
        reportCard.classList.toggle("is-failed", state === "failed");
    }
    if (reportLabel && state !== "completed") {
        reportLabel.textContent = (STATUS_TEXT[state] || state) + " — INCOMPLETE";
    }
}


function renderReportSection(data) {
    // Progressive section from the output pool: appended in document order so
    // the stream keeps flowing even while a fallback resumes a failed section
    // server-side. The final "report" event remains authoritative and replaces
    // this progressive rendering wholesale.
    let html = data.html || "";
    if (!html && data.data && window.renderMarkdown) {
        html = window.renderMarkdown(data.data);
    }
    if (!html && data.data) {
        html = "";
    }
    if (html) {
        progressiveHtml += (progressiveHtml ? "\n" : "") + html;
        report.classList.remove("is-loading");
        report.innerHTML = progressiveHtml;
        wireCitationTooltips();
    }
}


function renderReport(data) {
    currentMarkdown = data.data || "";
    progressiveHtml = "";
    report.classList.remove("is-loading");
    snapRetrieval();

    // Server pre-compiles markdown to HTML to avoid freezing the UI on
    // 15k-word reports. The client renderer is only a fallback for old payloads.
    if (data.html) {
        report.innerHTML = data.html;
    } else if (window.renderMarkdown) {
        report.innerHTML = window.renderMarkdown(currentMarkdown);
    } else {
        report.textContent = currentMarkdown;
    }
    wireCitationTooltips();

    const words = currentMarkdown.split(/\s+/).filter(Boolean).length;

    reportWords.textContent = `${formatNumber(words)} words`;

    const citations = report.querySelectorAll("a.citation").length;

    if (citations) {
        reportWords.textContent += ` · ${citations} citations`;
    }

    const sources = data.sources || [];

    if (sources.length) {
        const urls = sources
            .map((source) => source.url)
            .filter(Boolean);

        const unique = new Set(urls);

        evidenceCount.textContent =
            `${unique.size} sources · ${formatNumber(words)} words`;
    }

    const exportMeta = data.export || {};

    if (exportMeta.written) {
        downloadButton.classList.remove("hidden");
    }

    copyButton.classList.remove("hidden");
    pdfButton.classList.remove("hidden");

    wireDownload();
}


function wireDownload() {
    downloadButton.onclick = () => {
        // The browser cannot write to disk, so the server's export file is
        // served back as a download rather than faked client-side.
        window.location.href =
            `/api/report/download?q=${encodeURIComponent(currentQuestion)}`;
    };

    const rounds = roundsInput ? roundsInput.value : "2";
    const throttle = throttleToggle ? throttleToggle.value : "1";
    const pdfLoading = document.getElementById("pdf-loading");
    pdfButton.onclick = async () => {
        // Fetch (not navigate) so completion is observable: the bar runs
        // until the bytes arrive, then the download starts.
        const url =
            `/api/report/pdf?q=${encodeURIComponent(currentQuestion)}`
            + `&rounds=${encodeURIComponent(rounds)}`
            + `&throttle=${encodeURIComponent(throttle)}`;
        if (pdfLoading) {
            pdfLoading.classList.remove("hidden");
        }
        pdfButton.disabled = true;
        try {
            const response = await fetch(url);
            if (!response.ok) {
                throw new Error("PDF failed, status " + response.status);
            }
            const blob = await response.blob();
            const anchor = document.createElement("a");
            anchor.href = URL.createObjectURL(blob);
            anchor.download = "research-report.pdf";
            document.body.appendChild(anchor);
            anchor.click();
            anchor.remove();
            setTimeout(() => URL.revokeObjectURL(anchor.href), 5000);
        } catch (error) {
            pdfButton.textContent = "PDF failed";
            setTimeout(() => {
                pdfButton.textContent = "Save PDF";
            }, 2000);
        } finally {
            if (pdfLoading) {
                pdfLoading.classList.add("hidden");
            }
            pdfButton.disabled = false;
        }
    };

    copyButton.onclick = async () => {
        try {
            await navigator.clipboard.writeText(currentMarkdown);
            copyButton.textContent = "Copied";

            setTimeout(() => {
                copyButton.textContent = "Copy markdown";
            }, 1500);
        } catch (error) {
            copyButton.textContent = "Copy failed";
        }
    };
}


function formatNumber(value) {
    return Number(value || 0).toLocaleString("en-US");
}


function safeExternalUrl(url) {
    // Only schemes that cannot execute script are allowed out of scraped
    // search results; anything else falls back to an inert "#".
    const value = String(url || "").trim();
    if (/^https?:\/\//i.test(value)) {
        return value;
    }
    return "#";
}


function wireCitationTooltips() {
    const cites = report.querySelectorAll("a.citation");
    for (const cite of cites) {
        if (cite.dataset.wired) {
            continue;
        }
        cite.dataset.wired = "1";
        cite.addEventListener("mouseenter", () => {
            const number = cite.dataset.ref || cite.textContent.replace(/[^0-9]/g, "");
            const target = number && document.getElementById(`ref-${number}`);
            if (!target) {
                return;
            }
            const popup = document.createElement("span");
            popup.className = "cite-popup";
            const title = document.createElement("strong");
            title.textContent = `Reference [${number}]`;
            popup.appendChild(title);
            const body = document.createElement("span");
            body.textContent = target.textContent.trim().slice(0, 400);
            popup.appendChild(body);
            cite.appendChild(popup);
            const rect = cite.getBoundingClientRect();
            popup.style.left = "0";
            popup.style.bottom = "1.4em";
            if (rect.left < 340) {
                popup.style.left = "0";
            }
        });
        cite.addEventListener("mouseleave", () => {
            const popup = cite.querySelector(".cite-popup");
            if (popup) {
                popup.remove();
            }
        });
        cite.addEventListener("focus", () => {
            cite.dispatchEvent(new Event("mouseenter"));
        });
        cite.addEventListener("blur", () => {
            cite.dispatchEvent(new Event("mouseleave"));
        });
    }
}


(function initSettingsMenu() {
    const button = document.getElementById("settings-button");
    const menu = document.getElementById("settings-menu");
    if (!button || !menu) {
        return;
    }
    function setOpen(open) {
        menu.classList.toggle("hidden", !open);
        button.setAttribute("aria-expanded", open ? "true" : "false");
    }
    button.addEventListener("click", (event) => {
        event.stopPropagation();
        setOpen(menu.classList.contains("hidden"));
    });
    menu.addEventListener("click", (event) => {
        event.stopPropagation();
    });
    document.addEventListener("click", () => setOpen(false));
    document.addEventListener("keydown", (event) => {
        if (event.key === "Escape") {
            setOpen(false);
        }
    });
})();


(function initTheme() {
    const root = document.documentElement;
    const toggle = document.getElementById("theme-toggle");
    const label = toggle ? toggle.querySelector(".theme-label") : null;
    let saved = null;
    try {
        saved = localStorage.getItem("dr-theme");
    } catch (error) {
        saved = null;
    }
    if (saved === "light") {
        root.dataset.theme = "light";
    }
    function syncLabel() {
        const light = root.dataset.theme === "light";
        if (label) {
            label.textContent = light ? "Dark mode" : "Light mode";
        } else if (toggle) {
            toggle.textContent = light ? "Dark mode" : "Light mode";
        }
        if (toggle) {
            toggle.title = light ? "Switch to dark mode" : "Switch to light mode";
        }
    }
    syncLabel();
    if (toggle) {
        toggle.addEventListener("click", () => {
            const apply = () => {
                const light = root.dataset.theme !== "light";
                if (light) {
                    root.dataset.theme = "light";
                } else {
                    delete root.dataset.theme;
                }
                try {
                    localStorage.setItem("dr-theme", light ? "light" : "dark");
                } catch (error) {
                    /* private mode: theme just does not persist */
                }
                syncLabel();
                toggle.classList.remove("theme-spin");
                void toggle.offsetWidth;
                toggle.classList.add("theme-spin");
            };
            // Circular reveal from the top-left corner to the bottom-right
            // in 0.3s. Falls back to an instant swap where unsupported.
            if (document.startViewTransition) {
                document.startViewTransition(apply);
            } else {
                apply();
            }
        });
    }
})();