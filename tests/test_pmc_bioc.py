"""Tests for the PMC BioC full-text path.

This path was silently dead. The endpoint answers with a BioCCollection, which
is a JSON list, and the code assumed a bare object -- so every PMC article
raised AttributeError and the run fell through to a scraped markdown page
instead. That is the worst kind of failure: the document still arrived, so
nothing looked broken, and the report simply got worse text with nothing in
it saying so.
"""
import json

import pytest

from backend.research.fetch import Fetcher


class _Resp:
    def __init__(self, payload, status_code=200):
        self.status_code = status_code
        self.text = payload if isinstance(payload, str) else json.dumps(payload)
        self.headers = {"content-type": "application/json"}

    def raise_for_status(self):
        pass

    def iter_content(self, n):
        return iter(())

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _Session:
    def __init__(self, payload, status_code=200):
        self.payload = payload
        self.status_code = status_code
        self.headers = {}

    def get(self, url, **kwargs):
        return _Resp(self.payload, self.status_code)


def _collection(*sections):
    """A BioCCollection in the shape the endpoint actually returns: a list."""
    return [{
        "bioctype": "BioCCollection",
        "source": "PMC",
        "documents": [{
            "bioctype": "BioCDocument",
            "id": "PMC1",
            "infons": {"license": "CC BY"},
            "passages": [
                {"infons": {"section_type": name},
                 "text": f"{name} passage. " * 40}
                for name in sections
            ],
        }],
    }]


def _doc(fetcher, payload, pmcid="PMC1", status_code=200):
    return fetcher._from_bioc(
        type("D", (), {"final_url": "", "url": f"https://x/{pmcid}",
                       "method": "", "publisher": "", "text": "",
                       "status": ""})(),
        pmcid,
    )


def test_a_list_response_is_parsed(monkeypatch):
    """The shape the endpoint actually returns. This is the regression."""
    session = _Session(_collection("Abstract", "Methods"))
    fetcher = Fetcher(session=session)

    doc = type("D", (), {"final_url": "", "url": "https://x/PMC1", "method": "",
                         "publisher": "", "text": "", "status": ""})()
    doc = fetcher._from_bioc(doc, "PMC1")

    assert doc is not None, "a BioCCollection list was rejected"
    assert doc.method == "pmc"
    assert "Abstract" in doc.text and "Methods" in doc.text


def test_the_old_bare_object_still_works(monkeypatch):
    """So a change back by the provider is harmless rather than another
    silent regression."""
    payload = _collection("Abstract")[0]          # unwrap the list
    fetcher = Fetcher(session=_Session(payload))
    doc = type("D", (), {"final_url": "", "url": "https://x/PMC1", "method": "",
                         "publisher": "", "text": "", "status": ""})()
    doc = fetcher._from_bioc(doc, "PMC1")
    assert doc is not None
    assert doc.method == "pmc"


def test_several_collections_are_all_read(monkeypatch):
    """A list can carry more than one collection; stopping at the first would
    silently truncate an article."""
    payload = _collection("First") + _collection("Second")
    fetcher = Fetcher(session=_Session(payload))
    doc = type("D", (), {"final_url": "", "url": "https://x/PMC1", "method": "",
                         "publisher": "", "text": "", "status": ""})()
    doc = fetcher._from_bioc(doc, "PMC1")
    assert "First" in doc.text and "Second" in doc.text


@pytest.mark.parametrize("payload", [
    {"documents": [{"passages": [{"text": "x " * 60}]}]},
    [],
    [{}],
    [{"documents": None}],
    "not json at all",
    None,
])
def test_rubbish_does_not_raise(payload):
    """Whatever comes back, this returns None or a document. It must never
    raise: it runs inside the fetch path, and a raise there loses the article."""
    fetcher = Fetcher(session=_Session(payload))
    doc = type("D", (), {"final_url": "", "url": "https://x/PMC1", "method": "",
                         "publisher": "", "text": "", "status": ""})()
    try:
        result = fetcher._from_bioc(doc, "PMC1")
    except Exception as exc:                      # noqa: BLE001 - the point
        pytest.fail(f"_from_bioc raised on {payload!r}: {exc}")
    assert result is None or hasattr(result, "text")


def test_a_non_200_declines_so_the_chain_continues(monkeypatch):
    fetcher = Fetcher(session=_Session({}, status_code=404))
    doc = type("D", (), {"final_url": "", "url": "https://x/PMC1", "method": "",
                         "publisher": "", "text": "", "status": ""})()
    assert fetcher._from_bioc(doc, "PMC1") is None


def test_passages_without_section_labels_still_count(monkeypatch):
    """A missing section_type must drop the heading, not the passage."""
    payload = [{"documents": [{"passages": [
        {"infons": {}, "text": "plain passage " * 40},
    ]}]}]
    fetcher = Fetcher(session=_Session(payload))
    doc = type("D", (), {"final_url": "", "url": "https://x/PMC1", "method": "",
                         "publisher": "", "text": "", "status": ""})()
    doc = fetcher._from_bioc(doc, "PMC1")
    assert doc is not None
    assert "plain passage" in doc.text