"""Bridge-side Jev wiring, tested offline: no browser, no TypeSafe, no LLM calls."""

import asyncio
import json
import logging
import os
import subprocess
import sys
import uuid
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import app
import jev_navigator
from fakes import FakeAgent, FakeBrowser, FakeChatModel

USER = app.DEFAULT_USER_ID

REPO = Path(__file__).resolve().parent.parent

JEV_ENTRY = {
    "step": 1, "action": "Search Wikipedia", "operation": "TYPE_TEXT", "text": "Gödel", "probability": 0.9,
    "confidence": 0.8, "url": "https://en.wikipedia.org/wiki/Main_Page", "page_changed": True,
    "latency_ms": 300, "text_latency_ms": 250, "elapsed_ms": 900,
}
PAGE = {"url": "https://en.wikipedia.org/wiki/G%C3%B6del", "title": "Gödel - Wikipedia",
        "text": "Kurt Gödel was a logician.", "truncated": False}


class Recorder:
    """Async stand-in that records its call arguments and returns `result`."""

    def __init__(self, result=None):
        self.calls = []
        self.result = result

    async def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.result


def started_runs(monkeypatch):
    """Replace execute_task: record each start's keyword arguments, run nothing."""
    calls = []

    def fake_execute_task(*args, **kwargs):
        calls.append(kwargs)
        return asyncio.sleep(0)

    monkeypatch.setattr(app, "execute_task", fake_execute_task)
    return calls


def fake_navigate(outcome, entries=()):
    """jev_navigator.navigate stand-in: reports `entries` through on_step, then returns `outcome`."""

    async def navigate(browser_session, goal, controller, *, timeout_seconds, on_step, agent_factory=None):
        for entry in entries:
            await on_step(entry)
        return outcome

    return navigate


def run_jev(task_id, browser, start_url="https://en.wikipedia.org/wiki/Main_Page", ai_provider="ollama"):
    controller = jev_navigator.JevController(browser)
    asyncio.run(app._run_jev(task_id, USER, "Open the Gödel article", ai_provider, browser, controller, start_url))


# The browser-use prompt contract exactly as it was before the split; it must not change.
OLD_CONTRACT = (
    "Output contract:\n"
    "- Return the final answer as exactly one valid JSON object (no markdown, no code fences, no extra text).\n"
    "- Use dynamic keys that fit the task; do not rely on a fixed schema.\n"
    '- Include a short top-level "summary" string and put detailed values in other JSON fields.\n'
    '- If a requested value is unavailable, include the key with value null and explain briefly in "summary".\n'
    "\nCOMPLETION RULE (Critical):\n"
    "- Once the extract tool (or any tool) returns the requested data, IMMEDIATELY format as JSON and call done().\n"
    "- Do NOT attempt further browser navigation or tool calls after successful extraction.\n"
    "- Do NOT loop or retry; successful extraction = task complete."
)


@pytest.fixture(autouse=True)
def media_dir(monkeypatch, tmp_path):
    """Keep screenshots and run media out of the repo's media/ directory."""
    monkeypatch.setattr(app, "MEDIA_DIR", tmp_path)


def new_task(text="Go to example.com", status=app.TaskStatus.RUNNING, navigator="browser-use", extract=False):
    task_id = str(uuid.uuid4())
    app._new_task_record(task_id, text, "openai", USER, navigator=navigator, extract=extract)
    app.task_storage.update_task_status(task_id, status, USER)
    return task_id


def test_contract_split_keeps_the_browser_use_prompt_identical():
    assert app.JSON_OUTPUT_CONTRACT + app.BROWSER_USE_COMPLETION_RULE == OLD_CONTRACT
    assert app.build_agent_task("Do the thing").endswith(OLD_CONTRACT)


def test_record_auto_reward_scores_and_logs_a_finished_task():
    task_id = new_task()
    app.task_storage.set_task_output(task_id, '{"summary": "ok"}', USER)
    app.task_storage.mark_task_finished(task_id, USER, app.TaskStatus.FINISHED)

    app._record_auto_reward(task_id, USER)

    task = app.task_storage.get_task(task_id, USER)
    assert (task["reward"]["auto_score"], task["reward"]["effective_score"], task["reward"]["source"]) == (0.8, 0.8, "auto")
    assert task["trajectory"][-1]["event_type"] == "reward_auto"
    assert task["trajectory"][-1]["details"] == {"score": 0.8, "reason": "task finished with non-empty output"}


def test_navigator_setting_defaults_to_browser_use():
    assert app._navigator_setting(None) == "browser-use"
    assert app._navigator_setting("") == "browser-use"
    assert app._navigator_setting(" JEV ") == "jev"
    with pytest.raises(SystemExit, match="DEFAULT_NAVIGATOR must be one of"):
        app._navigator_setting("selenium")


def test_unknown_default_navigator_stops_startup():
    result = subprocess.run(
        [sys.executable, "-c", "import app"],
        cwd=REPO, env={**os.environ, "DEFAULT_NAVIGATOR": "selenium"},
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode != 0
    assert "DEFAULT_NAVIGATOR must be one of" in result.stderr


def test_task_request_accepts_only_known_navigators():
    assert app.TaskRequest(task="x", navigator="jev").navigator == "jev"
    assert app.TaskRequest(task="x").navigator is None
    assert app.TaskRequest(task="x", extract=True).extract is True
    with pytest.raises(ValueError):  # pydantic's ValidationError is a ValueError
        app.TaskRequest(task="x", navigator="selenium")


def test_create_run_uses_the_body_navigator(monkeypatch):
    calls = started_runs(monkeypatch)
    monkeypatch.setattr(app, "DEFAULT_NAVIGATOR", "jev")

    response = TestClient(app.app).post("/api/v4/runs", json={"task": "Go to example.com", "navigator": "browser-use"})

    assert response.status_code == 200
    assert app.task_storage.get_task(response.json()["id"], USER)["navigator"] == "browser-use"
    assert calls[0]["navigator"] == "browser-use"


def test_create_run_defaults_to_the_configured_navigator(monkeypatch):
    calls = started_runs(monkeypatch)
    monkeypatch.setattr(app, "DEFAULT_NAVIGATOR", "jev")

    response = TestClient(app.app).post("/api/v4/runs", json={"task": "Go to example.com"})

    assert app.task_storage.get_task(response.json()["id"], USER)["navigator"] == "jev"
    assert calls[0]["navigator"] == "jev"


def test_create_run_rejects_an_unknown_navigator(monkeypatch):
    calls = started_runs(monkeypatch)

    response = TestClient(app.app).post("/api/v4/runs", json={"task": "Go to example.com", "navigator": "selenium"})

    assert response.status_code == 400 and "navigator" in response.json()["detail"]
    assert calls == []


def test_create_run_takes_the_body_extract_flag(monkeypatch):
    started_runs(monkeypatch)
    monkeypatch.setattr(app, "JEV_EXTRACT", False)
    client = TestClient(app.app)

    opted_in = client.post("/api/v4/runs", json={"task": "Go to example.com", "extract": True}).json()
    default = client.post("/api/v4/runs", json={"task": "Go to example.com"}).json()

    assert app.task_storage.get_task(opted_in["id"], USER)["extract"] is True
    assert app.task_storage.get_task(default["id"], USER)["extract"] is False


def test_create_run_rejects_a_non_boolean_extract(monkeypatch):
    calls = started_runs(monkeypatch)

    response = TestClient(app.app).post("/api/v4/runs", json={"task": "Go to example.com", "extract": "yes"})

    assert response.status_code == 400 and "extract" in response.json()["detail"]
    assert calls == []


def test_jev_task_without_a_start_url_fails_before_any_browser(monkeypatch):
    monkeypatch.setattr(jev_navigator, "ensure_available", lambda: None)
    monkeypatch.setattr(app, "configure_browser_profile", lambda config, **kwargs: pytest.fail("a browser was configured"))
    text = "Find the cheapest flight from Zurich to London"
    task_id = new_task(text, navigator="jev")

    asyncio.run(app._execute_jev_task(task_id, text, "openai", USER))

    task = app.task_storage.get_task(task_id, USER)
    assert task["status"] == app.TaskStatus.FAILED
    assert "start URL" in task["error"]
    assert task["reward"]["source"] == "auto"


def test_jev_task_reports_a_missing_install(monkeypatch):
    def unavailable():
        raise jev_navigator.JevUnavailable("jev-ultrafast is not installed")

    monkeypatch.setattr(jev_navigator, "ensure_available", unavailable)
    task_id = new_task(navigator="jev")

    asyncio.run(app._execute_jev_task(task_id, "Go to example.com", "openai", USER))

    task = app.task_storage.get_task(task_id, USER)
    assert (task["status"], task["error"]) == (app.TaskStatus.FAILED, "jev-ultrafast is not installed")


def test_jev_task_runs_on_a_started_browser_and_closes_it(monkeypatch):
    browser = FakeBrowser()
    profile_calls = []
    run_jev_recorder = Recorder()

    def fake_profile(config, **kwargs):
        profile_calls.append(kwargs)
        return browser, {"headful": False}

    monkeypatch.setattr(jev_navigator, "ensure_available", lambda: None)
    monkeypatch.setattr(app, "configure_browser_profile", fake_profile)
    monkeypatch.setattr(app, "_run_jev", run_jev_recorder)
    task_id = new_task(navigator="jev")

    asyncio.run(app._execute_jev_task(task_id, "Go to example.com and read it", "ollama", USER))

    assert profile_calls == [{"downloads_dir": app.MEDIA_DIR / task_id}]  # no keep_alive: Jev has no Agent to reset it
    ((args, _kwargs),) = run_jev_recorder.calls
    assert args[:4] == (task_id, USER, "Go to example.com and read it", "ollama")
    assert args[4] is browser and isinstance(args[5], jev_navigator.JevController)
    assert args[6] == "https://example.com"
    assert browser.started and browser.closed
    assert app.task_storage.get_task_agent(task_id, USER) is args[5]


def test_cancel_during_browser_launch_stops_the_jev_run(monkeypatch):
    browser = FakeBrowser()
    task_id = new_task(navigator="jev")
    replies = []

    async def start_and_cancel():
        browser.started = True
        replies.append(await app.stop_task(task_id, USER))  # the user cancels while Chrome is still launching

    async def navigate(browser_session, goal, controller, *, timeout_seconds, on_step, agent_factory=None):
        return jev_navigator.JevOutcome("stopped" if controller.stopped else "done", target_id="TARGET-1")

    monkeypatch.setattr(browser, "start", start_and_cancel)
    monkeypatch.setattr(jev_navigator, "ensure_available", lambda: None)
    monkeypatch.setattr(app, "configure_browser_profile", lambda config, **kwargs: (browser, {}))
    monkeypatch.setattr(jev_navigator, "navigate", navigate)

    asyncio.run(app._execute_jev_task(task_id, "Go to example.com", "ollama", USER))

    assert replies == [{"message": "Task stopping"}]
    assert app.task_storage.get_task(task_id, USER)["status"] == app.TaskStatus.STOPPED


def test_headful_jev_task_without_chrome_path_builds_its_own_browser(monkeypatch):
    built = []

    def fake_browser(browser_profile):
        built.append(browser_profile)
        return FakeBrowser()

    monkeypatch.setattr(jev_navigator, "ensure_available", lambda: None)
    monkeypatch.setattr(app, "configure_browser_profile", lambda config, **kwargs: (None, {"headful": True}))
    monkeypatch.setattr(app, "Browser", fake_browser)
    monkeypatch.setattr(app, "_run_jev", Recorder())
    task_id = new_task(navigator="jev")

    asyncio.run(app._execute_jev_task(task_id, "Go to example.com", "ollama", USER))

    (profile,) = built
    assert profile.headless is False
    assert str(profile.downloads_path) == str(app.MEDIA_DIR / task_id)


def test_run_jev_finishes_with_the_extracted_output(monkeypatch, caplog):
    browser = FakeBrowser()
    model = FakeChatModel('{"summary": "found", "title": "Gödel"}')
    read_page = Recorder(PAGE)
    monkeypatch.setattr(
        jev_navigator, "navigate", fake_navigate(jev_navigator.JevOutcome("done", target_id="TARGET-1"), [JEV_ENTRY])
    )
    monkeypatch.setattr(jev_navigator, "read_page", read_page)
    monkeypatch.setattr(app, "get_llm", lambda provider: model)
    monkeypatch.setattr(app, "JEV_SCREENSHOTS", True)
    caplog.set_level(logging.INFO, logger="browser-use-bridge")
    task_id = new_task(navigator="jev", extract=True)

    run_jev(task_id, browser)

    task = app.task_storage.get_task(task_id, USER)
    assert task["status"] == app.TaskStatus.FINISHED and task["error"] is None
    assert json.loads(task["output"]) == {"summary": "found", "title": "Gödel"}
    assert browser.visited == ["https://en.wikipedia.org/wiki/Main_Page"]
    assert read_page.calls[0][0][1:] == ("TARGET-1", app.JEV_PAGE_TEXT_MAX_CHARS)
    (step,) = task["steps"]
    assert (step["navigator"], step["operation"], step["typesafe_latency_ms"]) == ("jev", "TYPE_TEXT", 300)
    assert [event["event_type"] for event in task["trajectory"]] == ["jev_action", "reward_auto"]
    assert [m["filename"].startswith("status-step-1-") for m in task["media"]] == [True]
    assert f"Task {task_id}: Jev step 1 TYPE_TEXT 'Search Wikipedia'" in caplog.text
    assert task["reward"]["auto_score"] == 0.8


def test_run_jev_returns_the_final_state_by_default(monkeypatch):
    monkeypatch.setattr(
        jev_navigator, "navigate", fake_navigate(jev_navigator.JevOutcome("done", target_id="TARGET-1"), [JEV_ENTRY])
    )
    monkeypatch.setattr(jev_navigator, "read_page", Recorder(PAGE))
    monkeypatch.setattr(app, "get_llm", lambda provider: pytest.fail("extraction LLM was called"))
    monkeypatch.setattr(app, "JEV_SCREENSHOTS", False)
    task_id = new_task(navigator="jev")

    run_jev(task_id, FakeBrowser())

    task = app.task_storage.get_task(task_id, USER)
    assert task["status"] == app.TaskStatus.FINISHED
    assert json.loads(task["output"]) == {
        "url": PAGE["url"], "title": PAGE["title"], "text": PAGE["text"], "text_truncated": False,
        "steps": [{"step": 1, "operation": "TYPE_TEXT", "action": "Search Wikipedia", "text": "Gödel",
                   "url": "https://en.wikipedia.org/wiki/Main_Page"}],
    }


def test_run_jev_without_a_start_url_stays_on_the_current_page(monkeypatch):
    browser = FakeBrowser()
    monkeypatch.setattr(jev_navigator, "navigate", fake_navigate(jev_navigator.JevOutcome("stopped")))
    task_id = new_task(navigator="jev")

    run_jev(task_id, browser, start_url=None)

    assert browser.visited == []
    assert app.task_storage.get_task(task_id, USER)["status"] == app.TaskStatus.STOPPED


def test_run_jev_blocked_fails_without_an_extraction_call(monkeypatch):
    reason = "Jev chose BLOCKED; last actions: none; url: https://example.com"
    monkeypatch.setattr(jev_navigator, "navigate", fake_navigate(jev_navigator.JevOutcome("blocked", reason)))
    monkeypatch.setattr(app, "get_llm", lambda provider: pytest.fail("extraction LLM was called"))
    task_id = new_task(navigator="jev")

    run_jev(task_id, FakeBrowser())

    task = app.task_storage.get_task(task_id, USER)
    assert (task["status"], task["error"], task["output"]) == (app.TaskStatus.FAILED, f"Jev blocked: {reason}", None)


def test_run_jev_fails_when_extraction_fails(monkeypatch):
    monkeypatch.setattr(jev_navigator, "navigate", fake_navigate(jev_navigator.JevOutcome("done", target_id="TARGET-1")))
    monkeypatch.setattr(jev_navigator, "read_page", Recorder(PAGE))
    monkeypatch.setattr(app, "get_llm", lambda provider: FakeChatModel(error=ConnectionError("Ollama is not running")))
    task_id = new_task(navigator="jev", extract=True)

    run_jev(task_id, FakeBrowser())

    task = app.task_storage.get_task(task_id, USER)
    assert (task["status"], task["error"]) == (app.TaskStatus.FAILED, "Extraction failed: Ollama is not running")


def test_run_jev_maps_a_timeout_to_stopped(monkeypatch):
    monkeypatch.setattr(
        jev_navigator, "navigate", fake_navigate(jev_navigator.JevOutcome("timeout", "Task timed out after 120 seconds"))
    )
    task_id = new_task(navigator="jev")

    run_jev(task_id, FakeBrowser())

    task = app.task_storage.get_task(task_id, USER)
    assert (task["status"], task["error"]) == (app.TaskStatus.STOPPED, "Task timed out after 120 seconds")


@pytest.fixture(autouse=True)
def isolated_sessions():
    """Remove any v4 sessions a test created."""
    before = set(app._sessions)
    yield
    for session_id in set(app._sessions) - before:
        app._sessions.pop(session_id, None)


def make_session(browser, navigator="jev"):
    session_id = f"sess-{uuid.uuid4()}"
    app._sessions[session_id] = {
        "id": session_id, "user_id": USER, "ai_provider": "ollama", "navigator": navigator,
        "status": "running", "current_run_id": None, "latest_run_id": None, "first_run_id": None,
        "agent": None, "browser_use_agent": None, "browser": browser, "queue": [], "next_message_id": 1,
        "created_at": "2026-10-03T00:00:00Z",
    }
    return session_id, app._sessions[session_id]


def test_new_session_takes_the_body_navigator(monkeypatch):
    calls = started_runs(monkeypatch)
    monkeypatch.setattr(app, "DEFAULT_NAVIGATOR", "jev")
    session_id = f"sess-{uuid.uuid4()}"

    response = TestClient(app.app).post(
        "/api/v4/runs", json={"task": "Go to example.com", "sessionId": session_id, "navigator": "browser-use"}
    )

    assert response.status_code == 200
    assert app._sessions[session_id]["navigator"] == "browser-use"
    assert calls[0]["navigator"] == "browser-use"
    assert app.task_storage.get_task(response.json()["id"], USER)["navigator"] == "browser-use"


def test_new_session_defaults_to_the_configured_navigator(monkeypatch):
    calls = started_runs(monkeypatch)
    monkeypatch.setattr(app, "DEFAULT_NAVIGATOR", "jev")
    session_id = f"sess-{uuid.uuid4()}"

    TestClient(app.app).post("/api/v4/runs", json={"task": "Go to example.com", "sessionId": session_id})

    assert app._sessions[session_id]["navigator"] == "jev"
    assert calls[0]["navigator"] == "jev"


@pytest.mark.parametrize("body_navigator, expected", [(None, "jev"), ("browser-use", "browser-use")])
def test_follow_ups_pick_their_navigator_or_use_the_sessions(monkeypatch, body_navigator, expected):
    continuations = []

    def fake_continuation(*args):
        continuations.append(args)
        return asyncio.sleep(0)

    monkeypatch.setattr(app, "_run_session_continuation", fake_continuation)
    monkeypatch.setattr(app, "DEFAULT_NAVIGATOR", "browser-use")  # so (None, "jev") proves the session's navigator wins
    session_id, session = make_session(FakeBrowser(), navigator="jev")
    session["status"] = "idle"
    body = {"task": "Now open the first result", "sessionId": session_id}
    if body_navigator:
        body["navigator"] = body_navigator

    response = TestClient(app.app).post("/api/v4/runs", json=body)

    assert response.status_code == 200
    assert app.task_storage.get_task(response.json()["id"], USER)["navigator"] == expected
    assert len(continuations) == 1


def test_follow_up_runs_take_their_own_extract_flag(monkeypatch):
    monkeypatch.setattr(app, "_run_session_continuation", lambda *args: asyncio.sleep(0))
    monkeypatch.setattr(app, "JEV_EXTRACT", False)
    session_id, session = make_session(FakeBrowser(), navigator="jev")
    session["status"] = "idle"

    response = TestClient(app.app).post(
        "/api/v4/runs", json={"task": "Now open the first result", "sessionId": session_id, "extract": True}
    )

    assert app.task_storage.get_task(response.json()["id"], USER)["extract"] is True


def test_jev_session_parks_the_browser_after_its_first_run(monkeypatch):
    browser = FakeBrowser()
    monkeypatch.setattr(jev_navigator, "ensure_available", lambda: None)
    monkeypatch.setattr(app, "configure_browser_profile", lambda config, **kwargs: (browser, {}))
    monkeypatch.setattr(app, "_run_jev", Recorder())
    session_id, session = make_session(None)
    task_id = new_task(navigator="jev")

    asyncio.run(app._execute_jev_task(task_id, "Go to example.com", "ollama", USER, session_id))

    assert session["browser"] is browser and not browser.closed
    assert isinstance(session["agent"], jev_navigator.JevController)
    assert session["status"] == "idle"
    assert session["browser_use_agent"] is None  # so a later browser-use follow-up fails fast with a clear message


@pytest.mark.parametrize(
    "message, start_url",
    [
        ("Now open the first result", None),
        ("Now open https://example.com/next and read it", "https://example.com/next"),
        ("Fill the Website field with acme.com and submit", None),
    ],
)
def test_jev_follow_up_continues_on_the_parked_tab(monkeypatch, message, start_url):
    browser = FakeBrowser()
    run_jev_recorder = Recorder()
    monkeypatch.setattr(jev_navigator, "ensure_available", lambda: None)
    monkeypatch.setattr(app, "_run_jev", run_jev_recorder)
    session_id, session = make_session(browser)
    task_id = new_task(message, navigator="jev")

    asyncio.run(app._run_session_continuation(session_id, task_id, message, USER))

    ((args, _kwargs),) = run_jev_recorder.calls
    assert args[:4] == (task_id, USER, message, "ollama")
    assert args[4] is browser and args[5] is session["agent"]
    assert isinstance(session["agent"], jev_navigator.JevController)
    assert args[6] == start_url
    assert (session["status"], session["current_run_id"]) == ("idle", None)
    assert app.task_storage.get_task_agent(task_id, USER) is session["agent"]


def test_jev_follow_up_without_a_browser_fails(monkeypatch):
    monkeypatch.setattr(jev_navigator, "ensure_available", lambda: None)
    session_id, session = make_session(None)
    task_id = new_task("Now open the first result", navigator="jev")

    asyncio.run(app._run_session_continuation(session_id, task_id, "Now open the first result", USER))

    task = app.task_storage.get_task(task_id, USER)
    assert task["status"] == app.TaskStatus.FAILED and "no active browser" in task["error"]
    assert session["status"] == "idle"


def test_queued_follow_ups_keep_the_session_navigator(monkeypatch):
    continuation = Recorder()
    monkeypatch.setattr(app, "_run_session_continuation", continuation)
    monkeypatch.setattr(app, "DEFAULT_NAVIGATOR", "browser-use")  # the session's own navigator (jev) must win
    session_id, session = make_session(FakeBrowser())
    session["status"] = "idle"
    session["queue"].append({"id": 1, "text": "Now open the first result", "run_id": None})

    asyncio.run(app._drain_session_queue(session_id, USER))

    ((args, _kwargs),) = continuation.calls
    task = app.task_storage.get_task(args[1], USER)
    assert (task["navigator"], task["extract"]) == ("jev", app.JEV_EXTRACT)


def test_parking_keeps_the_browser_use_agent_across_jev_runs():
    browser, agent = FakeBrowser(), FakeAgent()
    session_id, session = make_session(None, navigator="browser-use")

    async def park_both():
        await app._park_session_browser(session_id, agent, browser, USER)
        await app._park_session_browser(session_id, jev_navigator.JevController(browser), browser, USER)

    asyncio.run(park_both())

    assert session["browser_use_agent"] is agent
    assert isinstance(session["agent"], jev_navigator.JevController)


def test_jev_follow_up_on_a_browser_use_session_keeps_its_agent(monkeypatch):
    browser, agent = FakeBrowser(), FakeAgent()
    run_jev_recorder = Recorder()
    monkeypatch.setattr(jev_navigator, "ensure_available", lambda: None)
    monkeypatch.setattr(app, "_run_jev", run_jev_recorder)
    session_id, session = make_session(browser, navigator="browser-use")
    session["agent"] = session["browser_use_agent"] = agent
    message = "Now open the orders page"
    task_id = new_task(message, navigator="jev")

    asyncio.run(app._run_session_continuation(session_id, task_id, message, USER))

    ((args, _kwargs),) = run_jev_recorder.calls
    assert args[4] is browser and isinstance(args[5], jev_navigator.JevController)
    assert session["browser_use_agent"] is agent and agent.tasks == []
    assert session["agent"] is args[5]


def test_browser_use_follow_up_after_jev_continues_the_kept_agent(monkeypatch):
    browser, agent = FakeBrowser(), FakeAgent()
    monkeypatch.setattr(app, "process_task_result", Recorder())
    session_id, session = make_session(browser, navigator="browser-use")
    session["browser_use_agent"] = agent
    session["agent"] = jev_navigator.JevController(browser)  # a Jev follow-up drove the last run
    message = "Download the latest invoice"
    task_id = new_task(message, navigator="browser-use")

    asyncio.run(app._run_session_continuation(session_id, task_id, message, USER))

    assert (agent.tasks, agent.runs) == ([message], 1)
    assert session["agent"] is agent
    assert app.task_storage.get_task(task_id, USER)["status"] == app.TaskStatus.FINISHED
    assert (session["status"], session["current_run_id"]) == ("idle", None)


def test_browser_use_follow_up_on_a_jev_session_fails_fast(monkeypatch):
    # Before the fix this message takes the Jev path; keep that path offline and inert.
    monkeypatch.setattr(jev_navigator, "ensure_available", lambda: None)
    monkeypatch.setattr(app, "_run_jev", Recorder())
    browser = FakeBrowser()
    session_id, session = make_session(browser, navigator="jev")
    session["agent"] = jev_navigator.JevController(browser)
    message = "Download the latest invoice"
    task_id = new_task(message, navigator="browser-use")

    asyncio.run(app._run_session_continuation(session_id, task_id, message, USER))

    task = app.task_storage.get_task(task_id, USER)
    assert task["status"] == app.TaskStatus.FAILED
    assert 'start the session with "navigator": "browser-use"' in task["error"]
    assert session["status"] == "idle"


def test_jev_follow_up_can_use_the_browser_use_agents_own_browser(monkeypatch):
    """A headful browser-use first run without CHROME_PATH lets its Agent build the browser, so none is parked."""
    browser = FakeBrowser()
    run_jev_recorder = Recorder()
    monkeypatch.setattr(jev_navigator, "ensure_available", lambda: None)
    monkeypatch.setattr(app, "_run_jev", run_jev_recorder)
    session_id, session = make_session(None, navigator="browser-use")
    session["agent"] = session["browser_use_agent"] = FakeAgent(browser)
    message = "Now open the orders page"
    task_id = new_task(message, navigator="jev")

    asyncio.run(app._run_session_continuation(session_id, task_id, message, USER))

    ((args, _kwargs),) = run_jev_recorder.calls
    assert args[4] is browser
