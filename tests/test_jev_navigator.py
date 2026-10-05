"""jev_navigator, tested offline: fake CDP clients and a scripted Jev agent, never a browser or TypeSafe."""

import ast
import asyncio
import inspect
import json
import logging
import sys
import textwrap
import threading
from types import SimpleNamespace

import httpx
import pytest

import jev_navigator
from fakes import FakeBrowser, FakeCDPClient, FakeChatModel, FakeHTTPClient, FakeHTTPResponse, FakeLangChainModel


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Open https://example.com/a?b=1 and read it.", "https://example.com/a?b=1"),
        ("Go to google.com and search for n8n", "https://google.com"),
        ("Visit en.wikipedia.org/wiki/Main_Page, then stop", "https://en.wikipedia.org/wiki/Main_Page"),
        ("Go to google.com, then open https://www.youtube.com/watch?v=x", "https://google.com"),
        ("See https://en.wikipedia.org/wiki/Mercury_(planet).", "https://en.wikipedia.org/wiki/Mercury_(planet)"),
        ("(see example.org)", "https://example.org"),
        ("Mail bob@example.com about report.pdf, then go to news.ycombinator.com", "https://news.ycombinator.com"),
        ("Email john.smith@example.com and open dashboard.acme.com", "https://dashboard.acme.com"),
        ("Call pandas.read_csv on the export", None),
        ("Write to first.middle.last@example.com", None),
        ("Call self.config.get_value() first", None),
        ("google.comで検索して", "https://google.com"),
        ("Summarize notes.txt and app.py", None),
        ("Find the cheapest flight from Zurich to London", None),
        ("", None),
        ("まずgoogle.comを開いて", "https://google.com"),
        ("https://example.comを開いて", "https://example.com"),
        ("https://www.mercari.com/jp/を開いて商品を検索", "https://www.mercari.com/jp/"),
        ("Open `https://example.com/x`", "https://example.com/x"),
        ("Open zürich.ch", None),
        ("Open www.café.fr", None),
    ],
)
def test_find_start_url(text, expected):
    assert jev_navigator.find_start_url(text) == expected


async def _in_worker(client, function, *args, **kwargs):
    """Run a sync function in a worker thread with a JevContext for `client`, as a Jev tick runs."""
    context = jev_navigator.JevContext(asyncio.get_running_loop(), client, "TARGET-1", "SESSION-1")
    token = jev_navigator._context.set(context)
    try:
        return await asyncio.to_thread(function, *args, **kwargs)
    finally:
        jev_navigator._context.reset(token)


def test_router_sends_page_calls_on_the_tab_session():
    client = FakeCDPClient(replies={"Runtime.evaluate": {"result": {"value": 2}}})
    result = asyncio.run(
        _in_worker(client, jev_navigator._cdp, "Runtime.evaluate", session_id="SESSION-1", expression="1+1", returnByValue=True)
    )
    assert result == {"result": {"value": 2}}
    assert client.calls == [("Runtime.evaluate", {"expression": "1+1", "returnByValue": True}, "SESSION-1")]


def test_router_sends_target_calls_at_browser_level():
    client = FakeCDPClient()
    asyncio.run(
        _in_worker(client, jev_navigator._cdp, "Target.attachToTarget", session_id="SESSION-1", targetId="FRAME-1", flatten=True)
    )
    assert client.calls == [("Target.attachToTarget", {"targetId": "FRAME-1", "flatten": True}, None)]


def test_router_turns_a_lost_connection_into_runtime_error():
    client = FakeCDPClient(error=ConnectionError("WebSocket connection closed"))
    with pytest.raises(RuntimeError, match="connection lost during Page.navigate"):
        asyncio.run(_in_worker(client, jev_navigator._cdp, "Page.navigate", session_id="S", url="https://example.com"))


def test_router_turns_a_closed_websocket_into_runtime_error():
    from websockets.exceptions import ConnectionClosedError

    # cdp-use fails requests in flight with ConnectionError, but a send after the drop raises websockets' own error.
    client = FakeCDPClient(error=ConnectionClosedError(None, None))
    with pytest.raises(RuntimeError, match="connection lost during Runtime.evaluate"):
        asyncio.run(_in_worker(client, jev_navigator._cdp, "Runtime.evaluate", session_id="S", expression="1"))


def test_router_keeps_cdp_errors_as_runtime_error():
    client = FakeCDPClient(error=RuntimeError({"code": -32000, "message": "No frame"}))
    with pytest.raises(RuntimeError, match="No frame"):
        asyncio.run(_in_worker(client, jev_navigator._cdp, "Page.createIsolatedWorld", session_id="S", frameId="F"))


def test_router_times_out_as_runtime_error(monkeypatch):
    monkeypatch.setattr(jev_navigator, "CDP_TIMEOUT_SECONDS", 0.05)
    client = FakeCDPClient(delay=1.0)
    with pytest.raises(RuntimeError, match="timed out"):
        asyncio.run(_in_worker(client, jev_navigator._cdp, "Runtime.evaluate", session_id="S", expression="1"))


def test_router_keeps_concurrent_runs_on_their_own_browser():
    first, second = FakeCDPClient(), FakeCDPClient()
    barrier = threading.Barrier(2)

    def two_calls(name):
        jev_navigator._cdp("Runtime.evaluate", session_id=name, expression=f"'{name}-1'")
        barrier.wait(timeout=5)  # both runs have set their context by now, so a shared one would cross over
        jev_navigator._cdp("Runtime.evaluate", session_id=name, expression=f"'{name}-2'")

    async def scenario():
        await asyncio.gather(_in_worker(first, two_calls, "A"), _in_worker(second, two_calls, "B"))

    asyncio.run(scenario())
    assert [call[1]["expression"] for call in first.calls] == ["'A-1'", "'A-2'"]
    assert [call[1]["expression"] for call in second.calls] == ["'B-1'", "'B-2'"]


def test_jev_patch_points_still_exist():
    """Fails loudly if a jev-ultrafast update renames what the adapter patches or mirrors."""
    jev_browser = pytest.importorskip("jev_ultrafast.browser")
    jev_agent = pytest.importorskip("jev_ultrafast.agent")
    assert hasattr(jev_browser, "cdp") and hasattr(jev_browser, "ensure_daemon")
    assert "Browser(url)" in inspect.getsource(jev_agent.Agent.__init__)
    init_source = inspect.getsource(jev_browser.Browser.__init__)
    for attribute in ("self.contexts", "self.frame_ids", "self.after_input", "self.target", "self.session"):
        assert attribute in init_source
    for method in ("call", "evaluate", "forget", "observe", "act", "fresh"):
        assert callable(getattr(jev_browser.Browser, method))
    agent_source = inspect.getsource(jev_agent.Agent)
    for key in ('["status"]', '["history"]', '["decisions"]', '["page"]'):
        assert key in agent_source
    assert callable(jev_agent.Agent.command)


def test_ensure_available_routes_jev_through_the_bridge(monkeypatch):
    pytest.importorskip("jev_ultrafast")
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key")
    jev_agent = jev_navigator.ensure_available()
    from jev_ultrafast import browser as jev_browser

    assert jev_browser.cdp is jev_navigator._cdp
    assert jev_agent.Browser.__mro__[1] is jev_browser.Browser
    assert jev_navigator.ensure_available() is jev_agent


def test_ensure_available_requires_a_typesafe_key(monkeypatch):
    pytest.importorskip("jev_ultrafast")
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    with pytest.raises(jev_navigator.JevUnavailable, match="TYPESAFE_API_KEY"):
        jev_navigator.ensure_available()


def test_ensure_available_explains_a_missing_install(monkeypatch):
    monkeypatch.setattr(jev_navigator, "_jev_agent_module", None)
    monkeypatch.setitem(sys.modules, "jev_ultrafast", None)  # makes `from jev_ultrafast import ...` fail
    with pytest.raises(jev_navigator.JevUnavailable, match="pip install -e ../jev-ultrafast"):
        jev_navigator.ensure_available()


def test_bridge_browser_attaches_to_the_bridge_tab(monkeypatch):
    pytest.importorskip("jev_ultrafast")
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key")
    jev_agent = jev_navigator.ensure_available()
    client = FakeCDPClient(replies={"Runtime.evaluate": {"result": {"value": "complete"}}})

    browser = asyncio.run(_in_worker(client, jev_agent.Browser, "https://ignored.example"))

    assert (browser.target, browser.session) == ("TARGET-1", "SESSION-1")
    assert client.calls == [
        ("Emulation.setFocusEmulationEnabled", {"enabled": True}, "SESSION-1"),
        ("Runtime.evaluate", {"expression": "document.readyState", "returnByValue": True}, "SESSION-1"),
    ]


def test_bridge_browser_close_detaches_frames_but_keeps_the_tab(monkeypatch):
    pytest.importorskip("jev_ultrafast")
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key")
    jev_agent = jev_navigator.ensure_available()
    client = FakeCDPClient(replies={"Runtime.evaluate": {"result": {"value": "complete"}}})
    browser = asyncio.run(_in_worker(client, jev_agent.Browser, "https://ignored.example"))
    browser.contexts = {"FRAME-1": {"session": "OOPIF-SESSION", "context": None}}
    client.calls.clear()

    asyncio.run(_in_worker(client, browser.close))

    assert client.calls == [("Target.detachFromTarget", {"sessionId": "OOPIF-SESSION"}, None)]
    assert browser.contexts == {} and browser.frame_ids == {}


def act(label, page_changed=True):
    """Script step: Jev executed one action."""

    def step(state):
        state["history"].append(
            {"step": len(state["history"]) + 1, "action": label, "operation": "CLICK",
             "page_changed": page_changed, "url": state["page"]["url"]}
        )
        state["decisions"].append({"choice": label})

    return step


def finish(status, choice=None):
    """Script step: Jev reached a terminal status, optionally by choosing DONE/BLOCKED."""

    def step(state):
        if choice:
            state["decisions"].append({"choice": choice})
        state["status"] = status

    return step


class FakeAgent:
    """Scripted jev_ultrafast.Agent: each tick runs the next script step against its state."""

    def __init__(self, script):
        self.script = list(script)
        self.ticks = 0
        self.closed = False
        self.state = {"status": "ready", "history": [], "decisions": [], "page": {"url": "https://example.com/start"}}

    def command(self, name):
        assert name == "tick"
        self.ticks += 1
        self.script.pop(0)(self.state)
        return self.state

    def close(self):
        self.closed = True


def run_navigate(agent, controller=None, timeout_seconds=None, session=None):
    """navigate() with a scripted agent; returns (outcome, step entries reported through on_step)."""
    reported = []

    async def on_step(entry):
        reported.append(entry)

    session = session or FakeBrowser()
    controller = controller or jev_navigator.JevController(session)
    outcome = asyncio.run(
        jev_navigator.navigate(
            session, "the goal", controller, timeout_seconds=timeout_seconds, on_step=on_step,
            agent_factory=lambda url, goal: agent,
        )
    )
    return outcome, reported


def test_navigate_reports_each_action_then_done():
    agent = FakeAgent([act("Search box"), act("Search button"), finish("done", "DONE")])
    outcome, reported = run_navigate(agent)
    assert (outcome.kind, outcome.target_id, outcome.url) == ("done", "TARGET-1", "https://example.com/start")
    assert [entry["action"] for entry in reported] == ["Search box", "Search button"]
    assert agent.closed


def test_navigate_attaches_without_moving_browser_use_focus():
    session = FakeBrowser()
    run_navigate(FakeAgent([finish("done", "DONE")]), session=session)
    assert session.session_requests == [("TARGET-1", False)]


def test_navigate_explains_a_blocked_choice():
    outcome, _ = run_navigate(FakeAgent([act("Menu"), finish("blocked", "BLOCKED")]))
    assert outcome.kind == "blocked"
    assert outcome.reason == "Jev chose BLOCKED; last actions: Menu; url: https://example.com/start"


def test_navigate_explains_no_progress():
    outcome, _ = run_navigate(FakeAgent([act("A", False), act("B", False), act("C", False), finish("blocked")]))
    assert outcome.reason == "3 actions without a page change; last actions: A, B, C; url: https://example.com/start"


def test_navigate_reports_a_challenge():
    outcome, _ = run_navigate(FakeAgent([finish("awaiting_human")]))
    assert (outcome.kind, outcome.url) == ("challenge", "https://example.com/start")


def test_navigate_keeps_actions_logged_before_an_error():
    def act_then_fail(state):
        act("Submit")(state)
        raise RuntimeError("Model provider returned HTTP 500; no action executed.")

    outcome, reported = run_navigate(FakeAgent([act_then_fail]))
    assert outcome.kind == "error" and "HTTP 500" in outcome.reason
    assert [entry["action"] for entry in reported] == ["Submit"]


def test_navigate_reports_an_agent_start_failure():
    session = FakeBrowser()

    def broken_factory(url, goal):
        raise ValueError("Supply a task")

    outcome = asyncio.run(
        jev_navigator.navigate(session, "", jev_navigator.JevController(session), timeout_seconds=None,
                               on_step=None, agent_factory=broken_factory)
    )
    assert (outcome.kind, outcome.reason) == ("error", "Supply a task")


def test_navigate_stops_between_ticks():
    session = FakeBrowser()
    controller = jev_navigator.JevController(session)
    controller.stop()
    agent = FakeAgent([act("never")])
    outcome, _ = run_navigate(agent, controller=controller, session=session)
    assert outcome.kind == "stopped" and agent.ticks == 0


def test_navigate_times_out_between_ticks():
    agent = FakeAgent([act("never")])
    outcome, _ = run_navigate(agent, timeout_seconds=0)
    assert (outcome.kind, outcome.reason, agent.ticks) == ("timeout", "Task timed out after 0 seconds", 0)


def test_navigate_needs_browser_use_cdp_api():
    session = SimpleNamespace(agent_focus_target_id=None)
    with pytest.raises(jev_navigator.JevUnavailable, match="0.13.10"):
        asyncio.run(
            jev_navigator.navigate(session, "goal", jev_navigator.JevController(session), timeout_seconds=None,
                                   on_step=None, agent_factory=lambda url, goal: None)
        )


@pytest.mark.parametrize(
    "outcome, expected",
    [
        (jev_navigator.JevOutcome("done"), ("finished", None)),
        (jev_navigator.JevOutcome("blocked", "Jev chose BLOCKED; last actions: A; url: u"),
         ("failed", "Jev blocked: Jev chose BLOCKED; last actions: A; url: u")),
        (jev_navigator.JevOutcome("challenge", url="https://x.test"), ("failed", "Challenge on https://x.test needs a person")),
        (jev_navigator.JevOutcome("error", "Reached the demo's model-call budget"), ("failed", "Reached the demo's model-call budget")),
        (jev_navigator.JevOutcome("stopped"), ("stopped", None)),
        (jev_navigator.JevOutcome("timeout", "Task timed out after 120 seconds"), ("stopped", "Task timed out after 120 seconds")),
    ],
)
def test_terminal_status(outcome, expected):
    assert jev_navigator.terminal_status(outcome) == expected


def test_step_record_maps_a_jev_history_entry():
    entry = {
        "step": 3, "action": "Where to?", "kind": "fill", "choice": "f7", "probability": 0.94, "confidence": 0.91,
        "latency_ms": 310, "text": "London", "text_helper": "mercury", "text_latency_ms": 280,
        "operation": "TYPE_TEXT", "target": "7", "page_changed": True,
        "url": "https://www.google.com/travel/flights", "usage": {}, "executed_ms": 2000, "elapsed_ms": 2140,
    }
    assert jev_navigator.step_record(entry, "2026-10-03T12:00:00Z") == {
        "step": 3,
        "timestamp": "2026-10-03T12:00:00Z",
        "navigator": "jev",
        "next_goal": "TYPE_TEXT: Where to?",
        "operation": "TYPE_TEXT",
        "action": "Where to?",
        "text": "London",
        "probability": 0.94,
        "confidence": 0.91,
        "url": "https://www.google.com/travel/flights",
        "page_changed": True,
        "typesafe_latency_ms": 310,
        "text_latency_ms": 280,
        "elapsed_ms": 2140,
    }


PAGE = {"url": "https://en.wikipedia.org/wiki/X", "title": "X - Wikipedia", "text": "X is a theorem.", "truncated": False}


def test_read_page_reads_jevs_tab_and_caps_the_text():
    session = FakeBrowser()
    session.cdp_client.replies["Runtime.evaluate"] = {
        "result": {"value": {"url": "https://a.test", "title": "A", "text": "x" * 50}}
    }
    page = asyncio.run(jev_navigator.read_page(session, "TARGET-9", 10))
    assert page == {"url": "https://a.test", "title": "A", "text": "x" * 10, "truncated": True}
    method, params, session_id = session.cdp_client.calls[0]
    assert (method, params["returnByValue"], session_id) == ("Runtime.evaluate", True, "SESSION-1")
    assert session.session_requests == [("TARGET-9", False)]


def test_read_page_times_out_like_other_cdp_calls(monkeypatch):
    monkeypatch.setattr(jev_navigator, "CDP_TIMEOUT_SECONDS", 0.05)
    session = FakeBrowser()
    session.cdp_client.delay = 1.0
    with pytest.raises(RuntimeError, match="CDP Runtime.evaluate timed out after 0.05s"):
        asyncio.run(jev_navigator.read_page(session, "TARGET-9", 10))


def test_extract_output_asks_once_and_pretty_prints_json():
    from browser_use.llm.messages import UserMessage

    model = FakeChatModel('```json\n{"summary": "found", "title": "X"}\n```')
    output = asyncio.run(jev_navigator.extract_output(model, "Report the title", PAGE, "CONTRACT-TEXT"))
    assert output == json.dumps({"summary": "found", "title": "X"}, indent=2)
    assert len(model.prompts) == 1
    (message,) = model.prompts[0]
    assert isinstance(message, UserMessage)
    for expected in ("Report the title", "https://en.wikipedia.org/wiki/X", "X - Wikipedia", "X is a theorem.",
                     "CONTRACT-TEXT", "untrusted"):
        assert expected in message.content


def test_extract_output_uses_langchain_messages_for_langchain_models():
    from langchain_core.messages import HumanMessage

    model = FakeLangChainModel('Sure: {"summary": "ok"} done')
    output = asyncio.run(jev_navigator.extract_output(model, "task", PAGE, "C"))
    assert isinstance(model.prompts[0][0], HumanMessage)
    assert output == json.dumps({"summary": "ok"}, indent=2)


def test_extract_output_marks_truncated_pages():
    model = FakeChatModel("plain answer")
    asyncio.run(jev_navigator.extract_output(model, "task", {**PAGE, "truncated": True}, "C"))
    assert "Final page text (truncated):" in model.prompts[0][0].content


@pytest.mark.parametrize(
    "raw, expected",
    [
        ('{"a": 1}', '{\n  "a": 1\n}'),
        ('```\n{"a": "日本"}\n```', '{\n  "a": "日本"\n}'),
        ("no json here", "no json here"),
        ("{broken", "{broken"),
    ],
)
def test_normalize_output(raw, expected):
    assert jev_navigator.normalize_output(raw) == expected


def test_state_output_reports_where_jev_ended():
    steps = [{"step": 1, "operation": "CLICK", "action": "Search", "text": None, "url": "https://a.test"}]
    output = jev_navigator.state_output({**PAGE, "truncated": True}, steps)
    assert json.loads(output) == {
        "url": "https://en.wikipedia.org/wiki/X",
        "title": "X - Wikipedia",
        "text": "X is a theorem.",
        "text_truncated": True,
        "steps": steps,
    }


ROUTER_SETTINGS = (
    "TYPESAFE_API", "TYPESAFE_BASE_URL", "TYPESAFE_CUSTOM_HEADERS", "TEXT_MODEL_API", "TEXT_MODEL_CUSTOM_HEADERS"
)


@pytest.fixture(autouse=True)
def no_router(monkeypatch):
    """Clear router settings a developer's .env may have put in the environment."""
    for name in ROUTER_SETTINGS:
        monkeypatch.delenv(name, raising=False)


def test_model_calls_go_direct_without_router_settings(no_router):
    typesafe = jev_navigator.TYPESAFE_DEFAULT_URL
    text = "https://api.deepseek.com/v1/chat/completions"
    assert jev_navigator._route(typesafe) == (typesafe, {})
    assert jev_navigator._route(text) == (text, {})


def test_decisions_follow_the_typesafe_router(monkeypatch, no_router):
    monkeypatch.setenv("TYPESAFE_BASE_URL", "https://gateway.example/v1/acct/gw/typesafe/v1/")
    monkeypatch.setenv("TYPESAFE_CUSTOM_HEADERS", '{"cf-aig-authorization": "Bearer gw-token"}')
    monkeypatch.setenv("TEXT_MODEL_CUSTOM_HEADERS", '{"x-text": "1"}')
    assert jev_navigator._route(jev_navigator.TYPESAFE_DEFAULT_URL) == (
        "https://gateway.example/v1/acct/gw/typesafe/v1/systemone",
        {"cf-aig-authorization": "Bearer gw-token"},
    )


def test_text_helper_calls_get_their_own_headers(monkeypatch, no_router):
    monkeypatch.setenv("TYPESAFE_CUSTOM_HEADERS", '{"cf-aig-authorization": "Bearer gw-token"}')
    monkeypatch.setenv("TEXT_MODEL_CUSTOM_HEADERS", '{"cf-aig-authorization": "Bearer text-token"}')
    url = "https://gateway.example/compat/chat/completions"
    assert jev_navigator._route(url) == (url, {"cf-aig-authorization": "Bearer text-token"})


@pytest.mark.parametrize(
    "raw, message",
    [("{not json", "not valid JSON"), ('["a"]', "JSON object"), ('{"x-count": 1}', "JSON object")],
)
def test_custom_headers_must_be_a_json_object_of_strings(monkeypatch, raw, message):
    monkeypatch.setenv("TYPESAFE_CUSTOM_HEADERS", raw)
    with pytest.raises(jev_navigator.JevUnavailable, match=f"TYPESAFE_CUSTOM_HEADERS .*{message}"):
        jev_navigator._custom_headers("TYPESAFE_CUSTOM_HEADERS")


def test_routed_client_puts_router_headers_over_jevs(monkeypatch, no_router):
    monkeypatch.setenv("TYPESAFE_BASE_URL", "https://gateway.example/typesafe/v1")
    monkeypatch.setenv(
        "TYPESAFE_CUSTOM_HEADERS", '{"authorization": "Bearer router-key", "cf-aig-authorization": "Bearer gw"}'
    )
    fake = FakeHTTPClient()

    jev_navigator._RoutedClient(fake).post(
        jev_navigator.TYPESAFE_DEFAULT_URL, json={"q": 1}, headers={"Authorization": "Bearer typesafe-key"}
    )

    (call,) = fake.calls
    assert call["url"] == "https://gateway.example/typesafe/v1/systemone"
    assert call["json"] == {"q": 1}
    assert call["headers"] == {"authorization": "Bearer router-key", "cf-aig-authorization": "Bearer gw"}


def test_ensure_available_rejects_a_non_http_base_url(monkeypatch, no_router):
    pytest.importorskip("jev_ultrafast")
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key")
    monkeypatch.setenv("TYPESAFE_BASE_URL", "api.typesafe.ai/v1")
    with pytest.raises(jev_navigator.JevUnavailable, match="TYPESAFE_BASE_URL"):
        jev_navigator.ensure_available()


def test_ensure_available_rejects_bad_router_headers(monkeypatch, no_router):
    pytest.importorskip("jev_ultrafast")
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key")
    monkeypatch.setenv("TEXT_MODEL_CUSTOM_HEADERS", "{oops")
    with pytest.raises(jev_navigator.JevUnavailable, match="TEXT_MODEL_CUSTOM_HEADERS"):
        jev_navigator.ensure_available()


def test_jev_decision_requests_reach_the_router(monkeypatch, no_router):
    """jev's own post_json, retries and errors included, running through the routed client."""
    jev_model = pytest.importorskip("jev_ultrafast.model")
    monkeypatch.setenv("TYPESAFE_API_KEY", "typesafe-key")
    monkeypatch.setenv("TYPESAFE_BASE_URL", "https://gateway.example/typesafe/v1")
    monkeypatch.setenv("TYPESAFE_CUSTOM_HEADERS", '{"cf-aig-authorization": "Bearer gw"}')
    jev_navigator.ensure_available()
    assert isinstance(jev_model.CLIENT, jev_navigator._RoutedClient)
    fake = FakeHTTPClient(FakeHTTPResponse({"answers": {}, "model": "jev-test"}))
    monkeypatch.setattr(jev_model.CLIENT, "_client", fake)

    result = jev_model.post_json(jev_navigator.TYPESAFE_DEFAULT_URL, "typesafe-key", {"state": {}})

    assert result == {"answers": {}, "model": "jev-test"}
    (call,) = fake.calls
    assert call["url"] == "https://gateway.example/typesafe/v1/systemone"
    assert call["headers"] == {"authorization": "Bearer typesafe-key", "cf-aig-authorization": "Bearer gw"}


def test_jev_model_call_sites_still_match_the_router():
    """Fails loudly if jev-ultrafast changes how it calls TypeSafe or the text helper."""
    jev_model = pytest.importorskip("jev_ultrafast.model")
    assert "CLIENT.post(url, json=body, headers=" in inspect.getsource(jev_model.post_json)
    assert f'post_json("{jev_navigator.TYPESAFE_DEFAULT_URL}"' in inspect.getsource(jev_model.choose)
    # The definition plus exactly two call sites: TypeSafe decisions and the text helper.
    assert inspect.getsource(jev_model).count("post_json(") == 3


WORKERS_AI_URL = "https://gateway.example/ai/run"


@pytest.fixture
def workers_ai(monkeypatch, no_router):
    """Decisions through a Workers AI /ai/run endpoint, with no developer settings leaking in."""
    monkeypatch.setenv("TYPESAFE_API", "workers-ai")
    monkeypatch.setenv("TYPESAFE_BASE_URL", WORKERS_AI_URL)
    monkeypatch.delenv("TYPESAFE_MODEL", raising=False)


def test_workers_ai_decisions_go_to_the_configured_endpoint_as_is(workers_ai, monkeypatch):
    monkeypatch.setenv("TYPESAFE_CUSTOM_HEADERS", '{"cf-aig-authorization": "Bearer gw"}')
    assert jev_navigator._route(jev_navigator.TYPESAFE_DEFAULT_URL) == (
        WORKERS_AI_URL,
        {"cf-aig-authorization": "Bearer gw"},
    )


def test_workers_ai_request_wraps_jevs_body(workers_ai, monkeypatch):
    body = {"model": "jev-latest", "state": {"page": {}}, "questions": {"operation": {"type": "choice"}}}
    assert jev_navigator._workers_ai_request(body) == {
        "model": "typesafe/jev",
        "input": {"state": {"page": {}}, "questions": {"operation": {"type": "choice"}}},
    }
    monkeypatch.setenv("TYPESAFE_MODEL", "typesafe/jev-preview")
    assert jev_navigator._workers_ai_request(body)["model"] == "typesafe/jev-preview"


def test_workers_ai_decision_round_trip_through_jevs_post_json(workers_ai, monkeypatch):
    """jev's own post_json, through the routed client, against a reply in Cloudflare's envelope."""
    jev_model = pytest.importorskip("jev_ultrafast.model")
    monkeypatch.setenv("TYPESAFE_API_KEY", "cf-token")
    monkeypatch.setenv("TYPESAFE_CUSTOM_HEADERS", '{"cf-aig-authorization": "Bearer gw"}')
    jev_navigator.ensure_available()
    output = {"model": "jev-1.13.0", "answers": {}, "usage": {"input_tokens": 1, "output_tokens": 1}}
    fake = FakeHTTPClient(FakeHTTPResponse({"result": output, "success": True, "errors": [], "messages": []}))
    monkeypatch.setattr(jev_model.CLIENT, "_client", fake)

    result = jev_model.post_json(
        jev_navigator.TYPESAFE_DEFAULT_URL, "cf-token", {"model": "jev-latest", "state": {}, "questions": {}}
    )

    assert result == output
    (call,) = fake.calls
    assert call["url"] == WORKERS_AI_URL
    assert call["json"] == {"model": "typesafe/jev", "input": {"state": {}, "questions": {}}}
    assert call["headers"] == {"authorization": "Bearer cf-token", "cf-aig-authorization": "Bearer gw"}


def test_workers_ai_reply_without_an_envelope_passes_through():
    output = {"model": "jev-1.13.0", "answers": {}, "usage": {}}
    response = jev_navigator._workers_ai_response(FakeHTTPResponse(output))
    assert (response.status_code, response.json()) == (200, output)


def test_unsuccessful_workers_ai_envelope_raises_cloudflares_message():
    reply = FakeHTTPResponse({"result": None, "success": False, "errors": [{"code": 5006, "message": "bad input"}]})
    with pytest.raises(RuntimeError, match="Workers AI call failed: bad input"):
        jev_navigator._workers_ai_response(reply)


def test_workers_ai_error_status_is_left_to_jev():
    reply = FakeHTTPResponse({"success": False, "errors": [{"message": "Unauthorized"}]}, status_code=401)
    assert jev_navigator._workers_ai_response(reply) is reply


def test_failed_model_calls_log_the_providers_message(no_router, caplog):
    reply = {"error": [{"code": 2009, "message": "Unauthorized"}], "name": "AiGatewayError"}
    fake = FakeHTTPClient(FakeHTTPResponse(reply, status_code=401))

    with caplog.at_level(logging.WARNING, logger="browser-use-bridge"):
        response = jev_navigator._RoutedClient(fake).post(jev_navigator.TYPESAFE_DEFAULT_URL, json={}, headers={})

    assert response.status_code == 401
    assert "api.typesafe.ai failed with HTTP 401" in caplog.text and "AiGatewayError" in caplog.text


def test_typesafe_api_keeps_jevs_request_and_reply(no_router):
    reply = FakeHTTPResponse({"answers": {}, "model": "jev"})
    fake = FakeHTTPClient(reply)

    response = jev_navigator._RoutedClient(fake).post(
        jev_navigator.TYPESAFE_DEFAULT_URL, json={"model": "jev-latest", "state": {}}, headers={}
    )

    assert response is reply
    assert fake.calls[0]["json"] == {"model": "jev-latest", "state": {}}


@pytest.mark.parametrize(
    "api, message",
    [("bedrock", "TYPESAFE_API must be one of"), ("workers-ai", "TYPESAFE_API=workers-ai needs TYPESAFE_BASE_URL")],
)
def test_ensure_available_checks_the_typesafe_api_setting(monkeypatch, no_router, api, message):
    pytest.importorskip("jev_ultrafast")
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key")
    monkeypatch.setenv("TYPESAFE_API", api)
    with pytest.raises(jev_navigator.JevUnavailable, match=message):
        jev_navigator.ensure_available()


def test_find_start_url_can_require_a_full_url():
    assert jev_navigator.find_start_url("Fill the Website field with acme.com") == "https://acme.com"
    assert jev_navigator.find_start_url("Fill the Website field with acme.com", bare_domains=False) is None
    assert jev_navigator.find_start_url("Now open https://example.com/next and read it", bare_domains=False) == (
        "https://example.com/next"
    )


def test_workers_ai_reply_that_is_not_json_names_the_status():
    reply = httpx.Response(302, text="<html>moved</html>")
    with pytest.raises(RuntimeError, match="Workers AI returned HTTP 302 without JSON: <html>moved</html>"):
        jev_navigator._workers_ai_response(reply)


@pytest.mark.parametrize(
    "payload, message",
    [
        ({"success": False, "errors": [{"message": "quota"}]}, "Workers AI call failed: quota"),
        ({"result": None, "success": True}, "Workers AI returned an unexpected reply"),
    ],
)
def test_workers_ai_replies_without_answers_raise(payload, message):
    with pytest.raises(RuntimeError, match=message):
        jev_navigator._workers_ai_response(FakeHTTPResponse(payload))


def test_failed_model_call_log_is_capped(no_router, caplog):
    fake = FakeHTTPClient(FakeHTTPResponse({"detail": "x" * 5000}, status_code=500))
    with caplog.at_level(logging.WARNING, logger="browser-use-bridge"):
        jev_navigator._RoutedClient(fake).post("https://api.deepseek.com/v1/chat/completions", json={}, headers={})
    (record,) = [r for r in caplog.records if "Jev model call" in r.getMessage()]
    assert "api.deepseek.com failed with HTTP 500" in record.getMessage()
    assert len(record.getMessage()) < 400


def test_model_call_transport_failures_are_logged_and_reraised(no_router, caplog):
    class Unreachable:
        def post(self, url, **kwargs):
            raise httpx.ConnectError("name not resolved")

    with caplog.at_level(logging.WARNING, logger="browser-use-bridge"), pytest.raises(httpx.ConnectError):
        jev_navigator._RoutedClient(Unreachable()).post(jev_navigator.TYPESAFE_DEFAULT_URL, json={}, headers={})
    assert "api.typesafe.ai failed before a reply: ConnectError: name not resolved" in caplog.text


def _self_attributes(function):
    """Names a function assigns as self.<name>."""
    names = set()
    for node in ast.walk(ast.parse(textwrap.dedent(inspect.getsource(function)))):
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
            targets = [node.target]
        else:
            continue
        for target in targets:
            for element in target.elts if isinstance(target, ast.Tuple) else [target]:
                if isinstance(element, ast.Attribute) and isinstance(element.value, ast.Name) and element.value.id == "self":
                    names.add(element.attr)
    return names


def test_bridge_browser_sets_everything_jevs_browser_init_sets():
    """BridgeBrowser replaces jev's Browser.__init__ instead of calling it, so a new attribute there must be mirrored."""
    jev_browser = pytest.importorskip("jev_ultrafast.browser")
    bridge = jev_navigator._bridge_browser_class(jev_browser.Browser)
    assert _self_attributes(jev_browser.Browser.__init__) <= _self_attributes(bridge.__init__)


class ScriptedPageCDP:
    """CDP replies for two pages, shaped like jev's own scripts expect: page 1 has a Go button, and
    clicking it shows page 2. A canary for jev-ultrafast drift: it matches jev's scripts by identity."""

    def __init__(self, jev_browser):
        self.jev_browser = jev_browser
        self.page = 1
        self.threads = set()

    async def send_raw(self, method, params=None, session_id=None):
        params = params or {}
        self.threads.add(threading.current_thread().name)
        expression = params.get("expression") if isinstance(params.get("expression"), str) else ""
        if method == "Runtime.evaluate":
            if expression == "document.readyState":
                return {"result": {"value": "complete"}}
            if expression == self.jev_browser.MARKER:
                return {"result": {"value": f"m{self.page}"}}
            if expression == self.jev_browser.READ_STATE:
                return {"result": {"value": self.state()}}
            if expression.startswith("(() => { const c=window.__jevFast"):
                return {"result": {"value": [f"k{self.page}", "g20"]}}
            if expression.startswith(self.jev_browser.RESOLVE):
                return {"result": {"value": {"x": 100, "y": 100}}}
            if "new Promise" in expression:
                return {"result": {}}
            return {"result": {"value": None}}
        if method == "Input.dispatchMouseEvent" and params.get("type") == "mouseReleased":
            self.page = 2
        return {}

    def state(self):
        actions = [{"id": "wait", "kind": "wait", "label": "Wait"}]
        if self.page == 1:
            actions.insert(0, {"id": "e1", "kind": "click", "label": "Go", "role": "button", "node": 20})
        return {
            "url": f"https://example.test/{self.page}", "title": f"P{self.page}", "text": f"Page {self.page}",
            "scroll": {"y": 0}, "actions": actions, "marker": f"m{self.page}", "page_key": f"k{self.page}",
            "guards": {"20": "g20"}, "w": 1280, "h": 720, "iframes": [], "tokens": {},
        }


def test_real_jev_agent_runs_end_to_end_on_the_bridge_adapter(monkeypatch, no_router):
    """jev's own Agent through BridgeBrowser, the CDP router, the routed client and navigate(), offline."""
    jev_browser = pytest.importorskip("jev_ultrafast.browser")
    jev_model = pytest.importorskip("jev_ultrafast.model")
    monkeypatch.setenv("TYPESAFE_API_KEY", "fake-key")
    jev_navigator.ensure_available()
    authorizations = []

    def typesafe(request):
        authorizations.append(request.headers.get_list("authorization"))
        body = json.loads(request.content)
        if body["state"]["page"]["url"].endswith("/1"):
            answers = {
                "operation": {"choice": "CLICK", "confidence": 0.9,
                              "probabilities": {"CLICK": 0.7, "WAIT": 0.1, "DONE": 0.1, "BLOCKED": 0.1}},
                "click_target": {"choice": "1", "confidence": 0.95, "probabilities": {"1": 1.0}},
            }
        else:
            answers = {"operation": {"choice": "DONE", "confidence": 0.9,
                                     "probabilities": {"WAIT": 0.1, "DONE": 0.8, "BLOCKED": 0.1}}}
        return httpx.Response(200, json={"answers": answers, "model": "jev-test"})

    monkeypatch.setattr(jev_model.CLIENT, "_client", httpx.Client(transport=httpx.MockTransport(typesafe)))
    session = FakeBrowser()
    session.cdp_client = ScriptedPageCDP(jev_browser)
    reported = []

    async def on_step(entry):
        reported.append(entry)

    async def run():
        controller = jev_navigator.JevController(session)
        outcome = await jev_navigator.navigate(
            session, "Click Go, then finish", controller, timeout_seconds=30, on_step=on_step
        )
        return outcome, jev_navigator._context.get(None)

    outcome, leftover_context = asyncio.run(run())

    assert (outcome.kind, outcome.url) == ("done", "https://example.test/2")
    assert [(e["operation"], e["action"], e["page_changed"], e["url"]) for e in reported] == [
        ("CLICK", "Go", True, "https://example.test/2")
    ]
    assert leftover_context is None
    assert session.cdp_client.threads == {threading.current_thread().name}  # every CDP call ran on the loop thread
    assert authorizations and all(values == ["Bearer fake-key"] for values in authorizations)


def test_workers_ai_rest_reply_unwraps_the_completed_job():
    """Cloudflare's REST /ai/run reply as seen live: the envelope, then a finished job, then the model output."""
    output = {"model": "jev-1.13.0", "answers": {"operation": {"type": "choice"}}, "usage": {}}
    reply = FakeHTTPResponse({"result": {"state": "Completed", "result": output}, "success": True, "errors": [], "messages": []})
    assert jev_navigator._workers_ai_response(reply).json() == output


def test_workers_ai_job_that_did_not_complete_raises_its_state():
    reply = FakeHTTPResponse({"result": {"state": "Running", "result": None}, "success": True, "errors": [], "messages": []})
    with pytest.raises(RuntimeError, match=r"Workers AI job did not complete \(state 'Running'\)"):
        jev_navigator._workers_ai_response(reply)


TEXT_URL = "https://gateway.example/compat/chat/completions"
JEV_TEXT_BODY = {"model": "openai/gpt-6-luna", "max_tokens": 1024, "response_format": {"type": "json_object"},
                 "reasoning": {"effort": "low"}, "messages": [{"role": "user", "content": "x"}]}


def test_openai_text_requests_drop_jevs_reasoning_fields_and_rename_max_tokens():
    assert jev_navigator._openai_chat_request({**JEV_TEXT_BODY, "thinking": {"type": "disabled"}}) == {
        "model": "openai/gpt-6-luna", "max_completion_tokens": 1024, "response_format": {"type": "json_object"},
        "messages": [{"role": "user", "content": "x"}],
    }


def test_text_helper_calls_are_reshaped_only_with_text_model_api_openai(monkeypatch):
    fake = FakeHTTPClient()
    jev_navigator._RoutedClient(fake).post(TEXT_URL, json=JEV_TEXT_BODY, headers={})
    monkeypatch.setenv("TEXT_MODEL_API", "openai")
    jev_navigator._RoutedClient(fake).post(TEXT_URL, json=JEV_TEXT_BODY, headers={})
    decision_body = {"model": "jev-latest", "max_tokens": 5, "reasoning": {"effort": "low"}}
    jev_navigator._RoutedClient(fake).post(jev_navigator.TYPESAFE_DEFAULT_URL, json=dict(decision_body), headers={})

    default, openai, decision = (call["json"] for call in fake.calls)
    assert default == JEV_TEXT_BODY
    assert "reasoning" not in openai and openai["max_completion_tokens"] == 1024 and "max_tokens" not in openai
    assert decision == decision_body  # decisions are never reshaped for OpenAI


def test_ensure_available_checks_the_text_model_api_setting(monkeypatch):
    pytest.importorskip("jev_ultrafast")
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key")
    monkeypatch.setenv("TEXT_MODEL_API", "anthropic")
    with pytest.raises(jev_navigator.JevUnavailable, match="TEXT_MODEL_API must be one of"):
        jev_navigator.ensure_available()


@pytest.mark.parametrize("state", ["Completed", "completed", "COMPLETED"])
def test_workers_ai_completed_job_unwraps_in_any_case(state):
    output = {"model": "jev", "answers": {}, "usage": {}}
    reply = FakeHTTPResponse({"result": {"state": state, "result": output}, "success": True})
    assert jev_navigator._workers_ai_response(reply).json() == output


def test_workers_ai_output_with_its_own_state_key_passes_through():
    output = {"state": "anything", "model": "jev", "answers": {}, "usage": {}}
    assert jev_navigator._workers_ai_response(FakeHTTPResponse({"result": output, "success": True})).json() == output


@pytest.mark.parametrize(
    "payload",
    [{"result": {"state": "Completed", "result": None}, "success": True}, {"result": {"foo": 1}, "success": True}, {"success": True}],
)
def test_workers_ai_replies_without_model_answers_are_unexpected(payload):
    with pytest.raises(RuntimeError, match="Workers AI returned an unexpected reply"):
        jev_navigator._workers_ai_response(FakeHTTPResponse(payload))


def test_workers_ai_decisions_and_openai_text_calls_share_one_client(monkeypatch, workers_ai):
    monkeypatch.setenv("TEXT_MODEL_API", "openai")
    output = {"model": "jev", "answers": {}, "usage": {}}
    fake = FakeHTTPClient(FakeHTTPResponse({"result": {"state": "Completed", "result": output}, "success": True}))
    client = jev_navigator._RoutedClient(fake)

    decision = client.post(
        jev_navigator.TYPESAFE_DEFAULT_URL, json={"model": "jev-latest", "state": {}, "questions": {}}, headers={}
    )
    client.post(TEXT_URL, json=JEV_TEXT_BODY, headers={})

    assert decision.json() == output
    decision_call, text_call = fake.calls
    assert decision_call["json"] == {"model": "typesafe/jev", "input": {"state": {}, "questions": {}}}
    assert "reasoning" not in text_call["json"] and text_call["json"]["max_completion_tokens"] == 1024


def test_openai_text_requests_keep_an_explicit_max_completion_tokens():
    assert jev_navigator._openai_chat_request({"max_tokens": 1, "max_completion_tokens": 2}) == {"max_completion_tokens": 2}
