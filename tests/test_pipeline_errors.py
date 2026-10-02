import json

import pytest
from fastapi.testclient import TestClient
from google.genai.errors import APIError, ClientError

from backend import main
from backend.tools import github, news, serpapi

SUBQUESTIONS = [
    "What is FastAPI?",
    "Why use FastAPI?",
    "How to deploy FastAPI?",
]
PLAN = {"subquestions": SUBQUESTIONS}
QUESTION = "Tell me about FastAPI"


@pytest.fixture
def client():
    return TestClient(main.app)


@pytest.fixture
def pipeline(monkeypatch):
    """Replace the whole pipeline, so no Gemini or search API call is made."""
    state = {
        "search_errors": {},
        "research_errors": {},
        "report_error": None,
        "report_calls": [],
    }

    def fake_plan(question):
        return json.dumps(PLAN)

    def fake_search(subquestion):
        if subquestion in state["search_errors"]:
            raise state["search_errors"][subquestion]
        return [
            {
                "title": subquestion,
                "url": "https://example.com",
                "snippet": "Test evidence",
                "source": "google",
            }
        ]

    def fake_research(subquestion, search_results):
        if subquestion in state["research_errors"]:
            raise state["research_errors"][subquestion]
        return f"Finding for: {subquestion}"

    def fake_report(question, findings):
        state["report_calls"].append({"question": question, "findings": findings})
        if state["report_error"] is not None:
            raise state["report_error"]
        return "Final research report"

    monkeypatch.setattr(main, "create_research_plan", fake_plan)
    monkeypatch.setattr(main, "search_subquestion_tool", fake_search)
    monkeypatch.setattr(main, "run_researcher", fake_research)
    monkeypatch.setattr(main, "synthesize_report", fake_report)
    return state


def test_all_subquestions_succeed(client, pipeline):
    response = client.post("/research", json={"question": QUESTION})

    assert response.status_code == 200
    assert response.json()["findings"] == [
        {"subquestion": subquestion, "finding": f"Finding for: {subquestion}"}
        for subquestion in SUBQUESTIONS
    ]
    assert all("error" not in item for item in response.json()["findings"])


@pytest.mark.parametrize("failing", SUBQUESTIONS)
@pytest.mark.parametrize(
    "error",
    [
        serpapi.SerpApiError("google is down"),
        github.GitHubError("github is down"),
        news.MissingApiKeyError("no SERPAPI_KEY"),
    ],
    ids=["serpapi", "github", "news"],
)
def test_search_failure_does_not_stop_the_pipeline(client, pipeline, failing, error):
    pipeline["search_errors"][failing] = error

    findings = client.post("/research", json={"question": QUESTION}).json()["findings"]

    failed = [item for item in findings if item["subquestion"] == failing]
    assert len(failed) == 1
    assert failed[0]["finding"] is None
    assert "search failed" in failed[0]["error"]
    assert type(error).__name__ in failed[0]["error"]

    succeeded = [item for item in findings if item["subquestion"] != failing]
    assert len(succeeded) == len(SUBQUESTIONS) - 1
    assert all(item["finding"] == f"Finding for: {item['subquestion']}" for item in succeeded)


@pytest.mark.parametrize("failing", SUBQUESTIONS)
def test_researcher_failure_does_not_stop_the_pipeline(client, pipeline, failing):
    pipeline["research_errors"][failing] = APIError(503, "gemini is overloaded")

    findings = client.post("/research", json={"question": QUESTION}).json()["findings"]

    failed = [item for item in findings if item["subquestion"] == failing]
    assert failed[0]["finding"] is None
    assert "researcher failed" in failed[0]["error"]
    assert "APIError" in failed[0]["error"]

    succeeded = [item for item in findings if item["subquestion"] != failing]
    assert len(succeeded) == len(SUBQUESTIONS) - 1
    assert all(item["finding"] == f"Finding for: {item['subquestion']}" for item in succeeded)


def test_multiple_failures_keep_the_successful_findings(client, pipeline):
    pipeline["search_errors"]["What is FastAPI?"] = serpapi.SearchRequestError("timeout")
    pipeline["research_errors"]["Why use FastAPI?"] = APIError(429, "rate limited")

    findings = client.post("/research", json={"question": QUESTION}).json()["findings"]

    assert findings[0] == {
        "subquestion": "What is FastAPI?",
        "finding": None,
        "error": "search failed: SearchRequestError: timeout",
    }
    assert findings[1]["finding"] is None
    assert findings[1]["error"].startswith("researcher failed: APIError: ")
    assert "rate limited" in findings[1]["error"]
    assert findings[2] == {
        "subquestion": "How to deploy FastAPI?",
        "finding": "Finding for: How to deploy FastAPI?",
    }


def test_report_receives_only_successful_findings(client, pipeline):
    pipeline["search_errors"]["Why use FastAPI?"] = serpapi.SerpApiError("google is down")

    client.post("/research", json={"question": QUESTION})

    report_call = pipeline["report_calls"][0]
    assert report_call["question"] == QUESTION
    assert report_call["findings"] == [
        {"subquestion": "What is FastAPI?", "finding": "Finding for: What is FastAPI?"},
        {
            "subquestion": "How to deploy FastAPI?",
            "finding": "Finding for: How to deploy FastAPI?",
        },
    ]
    assert all(item["finding"] is not None for item in report_call["findings"])


def test_every_subquestion_failing_still_answers_with_an_empty_findings_list(
    client, pipeline
):
    for subquestion in SUBQUESTIONS:
        pipeline["search_errors"][subquestion] = serpapi.SerpApiError("google is down")

    response = client.post("/research", json={"question": QUESTION})

    assert response.status_code == 200
    assert all(item["finding"] is None for item in response.json()["findings"])
    assert pipeline["report_calls"][0]["findings"] == []


def test_response_keeps_the_expected_top_level_structure(client, pipeline):
    response = client.post("/research", json={"question": QUESTION})

    assert set(response.json()) == {"question", "plan", "report", "findings"}
    assert response.json()["question"] == QUESTION
    assert response.json()["plan"] == PLAN
    assert response.json()["report"] == "Final research report"


@pytest.mark.parametrize(
    "planner",
    [
        lambda question: "Sure, here are your subquestions: one and two",
        lambda question: (_ for _ in ()).throw(ValueError("plan is missing subquestions")),
    ],
    ids=["invalid-json-text", "planner-value-error"],
)
def test_planner_502_handling_is_preserved(client, monkeypatch, planner):
    monkeypatch.setattr(main, "create_research_plan", planner)

    response = client.post("/research", json={"question": QUESTION})

    assert response.status_code == 502
    assert response.json()["detail"] == "Research planner returned invalid JSON"


def test_unexpected_programming_error_is_not_swallowed(client, pipeline):
    pipeline["search_errors"]["What is FastAPI?"] = AttributeError("result shape is wrong")

    with pytest.raises(AttributeError):
        client.post("/research", json={"question": QUESTION})


@pytest.mark.parametrize(
    "error",
    [
        ClientError(
            429,
            "RESOURCE_EXHAUSTED: Quota exceeded for quota metric "
            "'Generate requests'.",
        ),
        APIError(503, "model overloaded"),
    ],
    ids=["quota-exceeded", "server-error"],
)
def test_report_gemini_error_returns_a_null_report(client, pipeline, error):
    pipeline["report_error"] = error

    response = client.post("/research", json={"question": QUESTION})

    assert response.status_code == 200
    body = response.json()
    assert body["report"] is None
    assert body["report_error"].startswith("Research report synthesis failed: ")
    assert type(error).__name__ in body["report_error"]
    assert body["findings"] == [
        {"subquestion": subquestion, "finding": f"Finding for: {subquestion}"}
        for subquestion in SUBQUESTIONS
    ]


def test_report_quota_error_names_the_quota_problem(client, pipeline):
    pipeline["report_error"] = ClientError(
        429,
        "RESOURCE_EXHAUSTED: Quota exceeded for quota metric "
        "'Generate requests'.",
    )

    body = client.post("/research", json={"question": QUESTION}).json()

    assert "RESOURCE_EXHAUSTED" in body["report_error"]
    assert "Quota exceeded" in body["report_error"]


def test_report_error_keeps_successful_and_failed_findings(client, pipeline):
    pipeline["search_errors"]["Why use FastAPI?"] = serpapi.SerpApiError("google is down")
    pipeline["report_error"] = ClientError(429, "RESOURCE_EXHAUSTED")

    body = client.post("/research", json={"question": QUESTION}).json()

    assert body["findings"] == [
        {"subquestion": "What is FastAPI?", "finding": "Finding for: What is FastAPI?"},
        {
            "subquestion": "Why use FastAPI?",
            "finding": None,
            "error": "search failed: SerpApiError: google is down",
        },
        {
            "subquestion": "How to deploy FastAPI?",
            "finding": "Finding for: How to deploy FastAPI?",
        },
    ]
    assert pipeline["report_calls"][0]["findings"] == [
        {"subquestion": "What is FastAPI?", "finding": "Finding for: What is FastAPI?"},
        {
            "subquestion": "How to deploy FastAPI?",
            "finding": "Finding for: How to deploy FastAPI?",
        },
    ]


def test_report_error_is_omitted_on_success(client, pipeline):
    body = client.post("/research", json={"question": QUESTION}).json()

    assert body["report"] == "Final research report"
    assert "report_error" not in body


def test_unexpected_report_error_is_not_swallowed(client, pipeline):
    pipeline["report_error"] = AttributeError("report module is broken")

    with pytest.raises(AttributeError):
        client.post("/research", json={"question": QUESTION})
