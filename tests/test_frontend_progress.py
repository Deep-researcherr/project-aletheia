"""Behavioural tests for the research-progress steps, executed for real.

The "Found a gap" step had two visible faults, both only reproducible by
running the code rather than reading it:

  * it stayed `active` for the whole of the following round -- several minutes
    of spinner with nothing behind it, because nothing updated it while the
    follow-up searches, retrieval and analysis ran;
  * it never appeared ticked once the report landed.

Both came from treating "a gap was found" as an ongoing state, when the gap
check has in fact *finished* the moment it hands follow-ups to the searcher.

dukpy executes the extracted functions against a minimal DOM stub, so these
assert on behaviour rather than on the presence of a string.
"""
import json
import re
from pathlib import Path

import pytest

dukpy = pytest.importorskip("dukpy", reason="dukpy not installed")

APP = (Path(__file__).resolve().parents[1] / "frontend" / "app.js").read_text()


def _extract(*names: str) -> str:
    """Pull the named top-level functions out of app.js, verbatim."""
    out = []
    for name in names:
        match = re.search(rf"^(?:async )?function {name}\(.*?^\}}", APP, re.M | re.S)
        assert match, f"{name} not found in app.js"
        out.append(match.group(0))
    return "\n\n".join(out)


def _extract_block(open_marker: str, close_marker: str = "\n};") -> str:
    """Grab a top-level `const X = { ... };` block verbatim."""
    start = APP.index(open_marker)
    end = APP.index(close_marker, start) + len(close_marker)
    return APP[start:end]


HARNESS = """
var states = {};
var text = {};
function mkRow(key) {
  return {
    dataset: {step: key},
    _cls: new Set(["progress-step"]),
    classList: {
      add: function (c) { this._owner._cls.add(c); },
      remove: function (c) { this._owner._cls.delete(c); },
      contains: function (c) { return this._owner._cls.has(c); },
      _owner: null,
    },
    _init: function () { this.classList._owner = this; return this; },
  };
}
var progressSteps = {
  plan: mkRow("plan")._init(),
  search: mkRow("search")._init(),
  read: mkRow("read")._init(),
  analyse: mkRow("analyse")._init(),
  gap: mkRow("gap")._init(),
  write: mkRow("write")._init(),
};
var progGapDetail = {textContent: ""};
var progReadTitle = {textContent: ""};
var progReadDetail = {textContent: ""};
var progAnalyseTitle = {textContent: ""};
var progAnalyseDetail = {textContent: ""};
var progWriteTitle = {textContent: ""};
var progWriteDetail = {textContent: ""};
var progSearchTitle = {textContent: ""};
var progSearchDetail = {textContent: ""};
var progPlanDetail = {textContent: ""};
var progSearchSources = {textContent: ""};
progressSteps.gap.classList.add("hidden");
var progressSearchSeen = new Set();
var progressSearchResults = 0;
var progressSections = 0;
function setStepState(step, state) { states[step] = state; }
"""


@pytest.fixture
def js():
    body = "\n\n".join([
        HARNESS,
        _extract("progressGap", "progressSearch", "progressDone", "progressLive",
                 "progressWrite", "progressRead"),
    ])
    interp = dukpy.JSInterpreter()
    interp.evaljs(body)
    return interp


def _run(js, calls):
    for call in calls:
        js.evaljs(call)
    return json.loads(js.evaljs("""
        JSON.stringify({
            states: states,
            gapHidden: progressSteps.gap.classList.contains('hidden'),
            gapDetail: progGapDetail.textContent
        })
    """))


def test_gap_with_followups_is_ticked_not_left_spinning(js):
    """The gap check is done once it has dispatched its follow-ups."""
    out = _run(js, [
        "progressGap({sufficient: false, missing: 'no primary source',"
        " follow_ups: [{source: 'web', query: 'a'},"
        " {source: 'scholar', query: 'b'}]});",
    ])
    assert out["states"]["gap"] == "done", (
        "gap step must not sit active through the next round")
    assert "Chased 2 follow-ups" in out["gapDetail"]
    # The chasing is real work, and search is what is doing it.
    assert out["states"]["search"] == "active"


def test_gap_with_a_single_followup_reads_naturally(js):
    out = _run(js, [
        "progressGap({sufficient: false, missing: 'thin coverage',"
        " follow_ups: [{source: 'web', query: 'a'}]});",
    ])
    assert "Chased 1 follow-up:" in out["gapDetail"]


def test_gap_is_hidden_when_the_run_is_satisfied(js):
    out = _run(js, ["progressGap({sufficient: true, missing: ''});"])
    assert out["gapHidden"] is True
    assert out["states"]["read"] == "done"


def test_gap_is_hidden_at_the_round_limit(js):
    """'round limit reached' is a normal ending, not an unresolved gap."""
    out = _run(js, ["progressGap({sufficient: false, missing: 'round limit reached'});"])
    assert out["gapHidden"] is True


def test_gap_without_followups_is_still_being_assessed(js):
    """A gap with nothing to chase has not resolved yet, so it may spin."""
    out = _run(js, ["progressGap({sufficient: false, missing: 'no primary source'});"])
    assert out["states"]["gap"] == "active"
    assert out["gapHidden"] is False
    assert out["gapDetail"] == "no primary source"


def test_gap_is_ticked_when_the_report_lands(js):
    """Whatever state it was in, a finished run ticks it."""
    out = _run(js, [
        "progressGap({sufficient: false, missing: 'thin'});",
        "progressDone();",
    ])
    assert out["states"]["gap"] == "done"


def test_every_visible_step_is_ticked_when_the_report_lands(js):
    out = _run(js, [
        "progressGap({sufficient: false, missing: 'thin',"
        " follow_ups: [{source: 'web', query: 'a'}]});",
        "progressDone();",
    ])
    for step in ("plan", "search", "read", "analyse", "gap", "write"):
        assert out["states"].get(step) == "done", f"{step} left as {out['states'].get(step)}"


# --- auto-scroll -----------------------------------------------------------

SCROLL_HARNESS = """
var scrollCalls = 0;
var PANEL_TOP = 400;
var LISTENERS = {};
var window = { scrollY: 0,
    addEventListener: function (t, f) {
        (LISTENERS[t] = LISTENERS[t] || []).push(f);
    } };
var results = {
    scrollIntoView: function () { scrollCalls++; },
    getBoundingClientRect: function () { return {top: PANEL_TOP}; },
};
function fire(type, event) {
    (LISTENERS[type] || []).forEach(function (fn) { fn(event || {}); });
}
"""


@pytest.fixture
def scroller():
    """The real registration block, not a hand-wired copy of it.

    `readerHasControl` is declared with `let` inside app.js's block, so the
    harness must not declare it -- doing so is an illegal redefinition. One
    combined eval keeps the whole thing in one global scope.
    """
    wiring = APP[APP.index("const SCROLL_KEYS"): APP.index("function followRun()")]
    interp = dukpy.JSInterpreter()
    interp.evaljs("\n".join([SCROLL_HARNESS, wiring, _extract("followRun")]))
    return interp


def _s(scroller, code):
    return scroller.evaljs(code)


def test_page_follows_the_run_while_the_reader_has_not_scrolled(scroller):
    _s(scroller, "PANEL_TOP = 400; followRun();")
    assert _s(scroller, "scrollCalls") == 1


def test_reading_input_hands_the_page_back_to_the_reader(scroller):
    """The bug: on the sources tab, every event yanked the view away."""
    _s(scroller, "fire('wheel'); PANEL_TOP = 400; followRun();")
    assert _s(scroller, "readerHasControl") is True
    assert _s(scroller, "scrollCalls") == 0, \
        "the page must not fight a reader who is scrolling"


def test_touch_and_keyboard_also_take_control(scroller):
    for kind, payload in (("touchmove", {}),
                          ("keydown", {"key": "ArrowDown"}),
                          ("keydown", {"key": "PageDown"})):
        _s(scroller, "readerHasControl = false;")
        _s(scroller, f"fire({kind!r}, {payload!r});")
        assert _s(scroller, "readerHasControl") is True, kind


def test_a_keypress_that_does_not_scroll_is_ignored(scroller):
    _s(scroller, "fire('keydown', {key: 'a'});")
    assert _s(scroller, "readerHasControl") is False


def test_returning_to_the_top_re_arms_following(scroller):
    _s(scroller, "fire('wheel');")
    assert _s(scroller, "readerHasControl") is True
    _s(scroller, "window.scrollY = 0; fire('scroll');")
    assert _s(scroller, "readerHasControl") is False
    _s(scroller, "PANEL_TOP = 400; followRun();")
    assert _s(scroller, "scrollCalls") == 1


def test_scrolling_within_the_page_does_not_disarm(scroller):
    """Only reaching the top re-arms; scrolling further down must not."""
    _s(scroller, "fire('wheel');")
    _s(scroller, "window.scrollY = 900; fire('scroll');")
    assert _s(scroller, "readerHasControl") is True


def test_no_scroll_when_the_panel_is_already_in_place(scroller):
    """Hundreds of events each re-scrolling causes visible jitter."""
    _s(scroller, "PANEL_TOP = 0; followRun();")
    assert _s(scroller, "scrollCalls") == 0


def test_repeated_events_do_not_stack_scroll_animations(scroller):
    for _ in range(50):
        _s(scroller, "PANEL_TOP = 400; followRun();")
    # One follow, then the panel is "in place" only if we model it moving;
    # with a static rect every call scrolls, so assert it is bounded per call.
    assert _s(scroller, "scrollCalls") == 50
    _s(scroller, "PANEL_TOP = 0;")
    before = _s(scroller, "scrollCalls")
    for _ in range(50):
        _s(scroller, "followRun();")
    assert _s(scroller, "scrollCalls") == before, \
        "a settled panel must stop being scrolled entirely"


# --- run timer -------------------------------------------------------------

TIMER_HARNESS = """
var runTimer = {hidden: true, classList: {
    add: function () {}, remove: function () {}}};
var runTimerElapsed = {textContent: ""};
var runTimerRemaining = {textContent: ""};
var runTimerFill = {style: {}};
var timerHandle = null;
var timerState = null;
var intervalCalls = 0;
function setInterval() { intervalCalls++; timerHandle = intervalCalls; return 1; }
function clearInterval() { timerHandle = null; }
"""


@pytest.fixture
def timer():
    """The real timer functions, with the clock and DOM stubbed.

    Elapsed/remaining are arithmetic over a controllable `now`, so these
    assert on the estimate rather than on wall-clock timing.
    """
    body = "\n\n".join([
        "var NOW = 1000000;",
        "var Date = {now: function () { return NOW; }};",
        TIMER_HARNESS,
        _extract_block("const TIMER_STAGES", "const TIMER_EXPECTED_SECTIONS = 18;"),
        _extract("timerReset", "timerFormat", "timerStage", "timerFinishStage",
                 "timerFraction", "timerRemainingSeconds", "timerTotalSeconds",
                 "timerTick", "timerFinish"),
    ])
    interp = dukpy.JSInterpreter()
    interp.evaljs(body)
    return interp


def _t(timer, setup, expr):
    timer.evaljs(setup)
    return timer.evaljs(expr)


def test_timer_formats_like_a_clock(timer):
    assert _t(timer, "", "timerFormat(0)") == "0s"
    assert _t(timer, "", "timerFormat(45)") == "45s"
    assert _t(timer, "", "timerFormat(252)") == "4m 12s"
    assert _t(timer, "", "timerFormat(3600)") == "1h 00m"
    assert _t(timer, "", "timerFormat(-5)") == "0s"


def test_timer_shows_an_estimate_only_once_something_is_timed(timer):
    """A prior with nothing measured behind it is a guess, and says so."""
    _t(timer, "timerReset();", "1")
    assert _t(timer, "", "runTimerRemaining.textContent") == "estimating…"

    # Once the plan stage has been timed, an estimate is legitimate.
    _t(timer, "timerStage('plan'); NOW += 30000; timerFinishStage();"
             " timerTick();", "1")
    assert _t(timer, "", "runTimerRemaining.textContent") != "estimating…"
    assert _t(timer, "", "runTimerElapsed.textContent") == "30s"


def test_measured_stage_replaces_the_prior(timer):
    """A stage timed in this run must inform the estimate for later ones."""
    prior = _t(timer, "", "TIMER_PRIORS.retrieve")
    _t(timer, "timerReset(); timerStage('retrieve'); NOW += 10000;"
             " timerFinishStage();", "1")
    measured = _t(timer, "", "timerState.measured.retrieve")
    assert abs(measured - 10.0) < 0.5
    assert measured < prior, "a 10s stage must beat a 110s prior"


def test_remaining_shrinks_as_a_stage_progresses(timer):
    _t(timer, "timerReset();", "1")
    _t(timer, "timerStage('retrieve');", "1")
    at_start = _t(timer, "", "timerRemainingSeconds()")
    _t(timer, "timerState.fraction.retrieve = 0.5;", "1")
    at_half = _t(timer, "", "timerRemainingSeconds()")
    assert at_half < at_start, "halfway through retrieval must reduce the estimate"
    assert at_half > 0


def test_completed_stages_cost_nothing(timer):
    _t(timer, "timerReset();", "1")
    before = _t(timer, "", "timerRemainingSeconds()")
    _t(timer, "timerStage('plan'); NOW += 1000; timerFinishStage();", "1")
    after = _t(timer, "", "timerRemainingSeconds()")
    assert after < before


def test_extra_gap_round_carries_another_pass(timer):
    """Without this the estimate collapses to zero while work is still running."""
    _t(timer, "timerReset(); timerState.extraRounds = 1;", "1")
    with_extra = _t(timer, "", "timerRemainingSeconds()")
    _t(timer, "timerState.extraRounds = 0;", "1")
    without = _t(timer, "", "timerRemainingSeconds()")
    assert with_extra > without, (
        "an extra round must not be estimated away")


def test_estimate_never_goes_negative(timer):
    _t(timer, "timerReset();", "1")
    for stage in ("plan", "search", "retrieve", "analyse", "write"):
        _t(timer, f"timerStage('{stage}'); timerState.fraction.{stage} = 1;"
                 " timerFinishStage();", "1")
    assert _t(timer, "", "timerRemainingSeconds()") == 0


def test_finished_timer_shows_the_servers_measurement(timer):
    """Prefer the server's number: it covers the whole run, the browser's does not."""
    _t(timer, "timerReset(); NOW += 5000;", "1")
    _t(timer, "timerFinish(423);", "1")
    assert _t(timer, "", "runTimerElapsed.textContent") == "7m 03s"
    assert _t(timer, "", "runTimerRemaining.textContent") == "done"
    assert _t(timer, "", "runTimerFill.style.width") == "100%"
    assert _t(timer, "", "timerHandle") is None, "interval must be cleared"


def test_finish_without_a_server_figure_falls_back_to_local_time(timer):
    _t(timer, "timerReset(); NOW += 90000;", "1")
    _t(timer, "timerFinish();", "1")
    assert _t(timer, "", "runTimerElapsed.textContent") == "1m 30s"


def test_reset_clears_previous_run_state(timer):
    _t(timer, "timerReset(); NOW += 60000; timerFinish(60);", "1")
    _t(timer, "timerReset();", "1")
    assert _t(timer, "", "timerState.finished") is False
    assert _t(timer, "", "timerState.measured.retrieve") is None


def test_a_stage_starting_at_time_zero_still_completes(timer):
    """`activeSince` is a timestamp and 0 is a legitimate one.

    A falsy guard silently refuses to close such a stage, which leaves its prior
    in the estimate for the rest of the run.
    """
    _t(timer, "NOW = 0; timerReset(); timerStage('plan');", "1")
    _t(timer, "NOW += 43000; timerFinishStage();", "1")
    assert _t(timer, "", "timerState.completed.plan") is True
    assert abs(_t(timer, "", "timerState.measured.plan") - 43.0) < 0.5


# Calibration, against the real stage boundaries of a logged 566s run. This is
# what stops the priors quietly drifting back to being wrong.
REAL_RUN = {"plan": 43, "search": 136, "retrieve": 59, "analyse": 235, "write": 93}
REAL_TOTAL = 566


def test_cold_start_estimate_is_within_15_percent(timer):
    """Priors alone, before anything in this run has been measured."""
    _t(timer, "NOW = 0; timerReset();", "1")
    projected = _t(timer, "", "timerTotalSeconds()")
    error = abs(projected - REAL_TOTAL) / REAL_TOTAL
    assert error < 0.15, f"cold start off by {error:.0%} ({projected:.0f}s vs {REAL_TOTAL}s)"


def test_estimate_converges_once_stages_are_measured(timer):
    _t(timer, "NOW = 0; timerReset();", "1")
    for stage, seconds in REAL_RUN.items():
        _t(timer, f"timerStage('{stage}'); NOW += {seconds * 1000};"
                  " timerFinishStage();", "1")
    projected = _t(timer, "", "timerTotalSeconds()")
    assert abs(projected - REAL_TOTAL) / REAL_TOTAL < 0.05


def test_mid_run_remaining_is_within_20_percent(timer):
    """The number a viewer actually watches while the run is working."""
    for fraction in (0.2, 0.4, 0.7, 0.9):
        _t(timer, "NOW = 0; timerReset();", "1")
        _t(timer, f"timerStage('plan'); NOW += {REAL_RUN['plan'] * 1000};"
                  " timerFinishStage();"
                  f" timerStage('search'); NOW += {REAL_RUN['search'] * 1000};"
                  " timerFinishStage();"
                  f" timerStage('retrieve');"
                  f" timerState.fraction.retrieve = {fraction};", "1")
        remaining = _t(timer, "", "timerRemainingSeconds()")
        spent = (REAL_RUN["plan"] + REAL_RUN["search"]
                 + REAL_RUN["retrieve"] * fraction)
        truth = REAL_TOTAL - spent
        error = abs(remaining - truth) / truth
        assert error < 0.20, (
            f"at {fraction:.0%} through retrieval, off by {error:.0%} "
            f"({remaining:.0f}s vs {truth:.0f}s)")