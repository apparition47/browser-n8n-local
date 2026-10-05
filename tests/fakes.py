"""Offline stand-ins for browser-use, cdp-use and chat models. No network, no browser."""

import asyncio
import json
from types import SimpleNamespace

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64


class FakeCDPClient:
    """Records send_raw calls; answers from a {method: result} table, or raises `error`."""

    def __init__(self, replies=None, error=None, delay=0.0):
        self.replies = dict(replies or {})
        self.error = error
        self.delay = delay
        self.calls = []

    async def send_raw(self, method, params=None, session_id=None):
        self.calls.append((method, params, session_id))
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.error:
            raise self.error
        return self.replies.get(method, {})


class FakeBrowser:
    """The parts of a browser-use BrowserSession the bridge and jev_navigator touch."""

    def __init__(self, target_id="TARGET-1"):
        self.agent_focus_target_id = target_id
        self.cdp_client = FakeCDPClient()
        self.session_requests = []
        self.visited = []
        self.started = False
        self.closed = False

    async def get_or_create_cdp_session(self, target_id=None, focus=True):
        self.session_requests.append((target_id, focus))
        return SimpleNamespace(session_id="SESSION-1", target_id=target_id, cdp_client=self.cdp_client)

    async def start(self):
        self.started = True

    async def navigate_to(self, url, new_tab=False):
        self.visited.append(url)

    async def take_screenshot(self, full_page=False):
        return PNG

    async def close(self):
        self.closed = True


class FakeChatModel:
    """browser-use style chat model: ainvoke(messages) returns an object with .completion."""

    def __init__(self, reply=None, error=None):
        self.reply = reply
        self.error = error
        self.prompts = []

    async def ainvoke(self, messages):
        self.prompts.append(messages)
        if self.error:
            raise self.error
        return SimpleNamespace(completion=self.reply)


class FakeLangChainModel(FakeChatModel):
    """LangChain style chat model (the bridge's deepseek provider): replies through .content."""

    async def ainvoke(self, messages):
        self.prompts.append(messages)
        return SimpleNamespace(content=self.reply)


FakeLangChainModel.__module__ = "langchain_fake.chat_models"


class FakeHTTPResponse:
    """The parts of an httpx.Response that jev's post_json reads."""

    def __init__(self, payload=None, status_code=200):
        self.payload = payload if payload is not None else {}
        self.status_code = status_code
        self.is_error = status_code >= 400

    def json(self):
        return self.payload

    @property
    def text(self):
        return json.dumps(self.payload)


class FakeHTTPClient:
    """Records post() calls as httpx.Client receives them (header names lowercased); replies with `response`."""

    def __init__(self, response=None):
        self.response = response or FakeHTTPResponse()
        self.calls = []

    def post(self, url, json=None, headers=None, **kwargs):
        self.calls.append({"url": url, "json": json, "headers": {k.lower(): v for k, v in dict(headers or {}).items()}})
        return self.response


class FakeAgent:
    """The parts of a browser-use Agent a session follow-up touches."""

    def __init__(self, browser_session=None):
        self.browser_session = browser_session
        self.tasks = []
        self.runs = 0
        self.stopped = False

    def add_new_task(self, text):
        self.tasks.append(text)

    async def run(self, **kwargs):
        self.runs += 1
        return "history"

    def stop(self):
        self.stopped = True
