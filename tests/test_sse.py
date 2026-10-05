"""SSE contract for GET /api/research."""

import json

import pytest
from fastapi.testclient import TestClient

from backend import main


def parse_frames(text):
    """Turn raw SSE text back into the decoded event dicts."""
    events = []
    for frame in text.split("\n\n"):
        if not frame.strip():
            continue
        line = next(
            (ln for ln in frame.split("\n") if ln.startswith("data: ")),
            None,
        )
        assert line is not None, f"frame is not a data frame: {frame!r}"
        events.append(json.loads(line[6:]))
    return events


@pytest.fixture
def client():
    return TestClient(main.app)


def install_fake(monkeypatch, events, calls=None, raises=None):

    async def fake_deep_research(question, max_rounds=3):
        if calls is not None:
            calls.append({"question": question, "max_rounds": max_rounds})
        for event in events:
            yield event
        if raises is not None:
            raise raises

    monkeypatch.setattr(main, "deep_research", fake_deep_research)


def test_content_type_is_event_stream(client, monkeypatch):
    install_fake(monkeypatch, [{"type": "report", "data": "x", "sources": []}])

    response = client.get("/api/research?q=test")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")


def test_every_frame_is_data_json_blank_line(client, monkeypatch):
    install_fake(
        monkeypatch,
        [
            {"type": "plan", "data": {"subquestions": []}},
            {"type": "report", "data": "done", "sources": []},
        ],
    )

    raw = client.get("/api/research?q=test").text

    assert raw.startswith("data: ")
    for frame in raw.split("\n\n"):
        if frame.strip():
            assert frame.startswith("data: ")
            json.loads(frame[6:])


def test_done_is_always_last(client, monkeypatch):
    install_fake(
        monkeypatch,
        [
            {"type": "plan", "data": {"subquestions": []}},
            {"type": "evidence", "count": 3},
            {"type": "report", "data": "final", "sources": []},
        ],
    )

    events = parse_frames(client.get("/api/research?q=test").text)

    assert [e["type"] for e in events] == ["plan", "evidence", "report", "done"]
    assert events[-1]["type"] == "done"


def test_generator_error_produces_error_event_then_done(client, monkeypatch):
    install_fake(
        monkeypatch,
        [{"type": "plan", "data": {"subquestions": []}}],
        raises=RuntimeError("all models failed"),
    )

    response = client.get("/api/research?q=test")

    assert response.status_code == 200
    events = parse_frames(response.text)

    assert [e["type"] for e in events] == ["plan", "error", "done"]
    assert events[-1]["type"] == "done"
    assert "all models failed" in events[-2]["message"]


def test_done_is_last_even_when_the_first_event_errors(client, monkeypatch):
    install_fake(monkeypatch, [], raises=RuntimeError("boom"))

    events = parse_frames(client.get("/api/research?q=test").text)

    assert [e["type"] for e in events] == ["error", "done"]


@pytest.mark.parametrize(
    ("requested", "expected"),
    [(0, 1), (1, 1), (2, 2), (3, 3), (4, 4), (99, 4), (-5, 1)],
)
def test_rounds_are_clamped(client, monkeypatch, requested, expected):
    calls = []
    install_fake(monkeypatch, [{"type": "report", "data": "x", "sources": []}], calls=calls)

    client.get(f"/api/research?q=test&rounds={requested}")

    assert calls[0]["max_rounds"] == expected


def test_rounds_defaults_to_two_when_omitted(client, monkeypatch):
    calls = []
    install_fake(monkeypatch, [{"type": "report", "data": "x", "sources": []}], calls=calls)

    client.get("/api/research?q=test")

    assert calls[0]["max_rounds"] == 2


def test_question_reaches_the_pipeline_intact(client, monkeypatch):
    calls = []
    install_fake(monkeypatch, [{"type": "report", "data": "x", "sources": []}], calls=calls)

    client.get("/api/research?q=C%2B%2B+vs+Rust")

    assert calls[0]["question"] == "C++ vs Rust"


def test_missing_q_returns_422(client, monkeypatch):
    install_fake(monkeypatch, [{"type": "report", "data": "x", "sources": []}])

    assert client.get("/api/research").status_code == 422


def test_localhost_post_preflight_is_allowed(client):
    response = client.options(
        "/research",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type, authorization",
        },
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:5173"
    assert "POST" in response.headers["access-control-allow-methods"]
    assert "content-type" in response.headers["access-control-allow-headers"]
    assert "authorization" in response.headers["access-control-allow-headers"]


def test_non_local_origin_preflight_is_rejected(client):
    response = client.options(
        "/research",
        headers={
            "Origin": "https://example.com",
            "Access-Control-Request-Method": "POST",
        },
    )

    assert response.status_code == 400