from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
import base64
import mimetypes
import signal
import sys
import re

from typing import Optional
from datetime import datetime, UTC
from enum import Enum
from typing import cast
from types import TracebackType


import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, Response, Query, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager

from langchain_anthropic import ChatAnthropic
from langchain_mistralai import ChatMistralAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_aws import ChatBedrock
from langchain_openai import AzureChatOpenAI, ChatOpenAI as LCChatOpenAI
from pydantic import BaseModel, Field

# This import will work once browser-use is installed
# For development, you may need to add the browser-use repo to your PYTHONPATH
# from browser_use import Agent
# from browser_use.agent.views import AgentHistoryList
# from browser_use import BrowserConfig, Browser
# from browser_use.browser.context import BrowserContext

# from browser_use.llm import LLMProvider

from browser_use import Agent
from browser_use.agent.views import AgentHistoryList
from browser_use import BrowserProfile, Browser


from browser_use.llm import (
    ChatAnthropic,
    ChatOpenAI,
    ChatGoogle,
    ChatOllama,
    ChatAzureOpenAI,
    ChatAWSBedrock,
)

from pathlib import Path

# Import our task storage abstraction
from task_storage import get_task_storage
from task_storage.base import DEFAULT_USER_ID


# Define task status enum
class TaskStatus(str, Enum):
    CREATED = "created"  # Task is initialized but not yet started
    RUNNING = "running"  # Task is currently executing
    FINISHED = "finished"  # Task has completed successfully
    STOPPED = "stopped"  # Task was manually stopped
    PAUSED = "paused"  # Task execution is temporarily paused
    FAILED = "failed"  # Task encountered an error and could not complete
    STOPPING = "stopping"  # Task is in the process of stopping (transitional state)


# Load environment variables from .env file
load_dotenv()

# Create media directory if it doesn't exist
MEDIA_DIR = Path("media")
MEDIA_DIR.mkdir(exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("browser-use-bridge")

MAX_DOM_SNAPSHOT_CHARS = int(os.environ.get("MAX_DOM_SNAPSHOT_CHARS", "120000"))
TASK_RUN_TIMEOUT_SECONDS = int(os.environ.get("TASK_RUN_TIMEOUT_SECONDS", "120"))
AGENT_MAX_STEPS = int(os.environ.get("AGENT_MAX_STEPS", "8"))
STATUS_TRACK_STEPS_ON_POLL = (
    os.environ.get("STATUS_TRACK_STEPS_ON_POLL", "false").lower() == "true"
)
STATUS_CAPTURE_SCREENSHOT = (
    os.environ.get("STATUS_CAPTURE_SCREENSHOT", "false").lower() == "true"
)
STATUS_SCREENSHOT_MIN_INTERVAL_SECONDS = int(
    os.environ.get("STATUS_SCREENSHOT_MIN_INTERVAL_SECONDS", "10")
)
AGENT_ENFORCE_CONCISE_EXECUTION = (
    os.environ.get("AGENT_ENFORCE_CONCISE_EXECUTION", "true").lower() == "true"
)
ENABLE_SIMPLE_TITLE_SHORTCUT = (
    os.environ.get("ENABLE_SIMPLE_TITLE_SHORTCUT", "true").lower() == "true"
)
PASS_SENSITIVE_DATA_TO_AGENT = (
    os.environ.get("PASS_SENSITIVE_DATA_TO_AGENT", "false").lower() == "true"
)
LOOP_GUARD_MAX_CONSECUTIVE_DUPLICATE_SCREENSHOTS = int(
    os.environ.get("LOOP_GUARD_MAX_CONSECUTIVE_DUPLICATE_SCREENSHOTS", "3")
)
LOOP_GUARD_MAX_SCREENSHOT_ERRORS = int(
    os.environ.get("LOOP_GUARD_MAX_SCREENSHOT_ERRORS", "3")
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle application startup and shutdown"""
    # Startup
    logger.info("Browser Use Bridge API starting up...")
    yield
    # Shutdown
    logger.info("Browser Use Bridge API shutting down...")
    await cleanup_all_tasks()


app = FastAPI(title="Browser Use Bridge API", lifespan=lifespan)

# Mount static files
app.mount("/media", StaticFiles(directory=str(MEDIA_DIR)), name="media")


# Custom JSON encoder for Enum serialization
class EnumJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Enum):
            return obj.value
        return super().default(obj)


# Configure FastAPI to use custom JSON serialization for responses
@app.middleware("http")
async def add_json_serialization(request: Request, call_next):
    response = await call_next(request)

    # Only attempt to modify JSON responses and check if body() method exists
    if response.headers.get("content-type") == "application/json" and hasattr(
        response, "body"
    ):
        try:
            content = await response.body()
            content_str = content.decode("utf-8")
            content_dict = json.loads(content_str)
            # Convert any Enum values to their string representation
            content_str = json.dumps(content_dict, cls=EnumJSONEncoder)
            response = Response(
                content=content_str,
                status_code=response.status_code,
                headers=dict(response.headers),
                media_type="application/json",
            )
        except Exception as e:
            logger.error(f"Error serializing JSON response: {str(e)}")

    return response


# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize task storage
task_storage = get_task_storage()


# Models
class TaskRequest(BaseModel):
    task: str
    ai_provider: Optional[str] = os.environ.get(
        "DEFAULT_AI_PROVIDER", "openai"
    )  # Default to OpenAI or env var
    save_browser_data: Optional[bool] = False  # Whether to save browser cookies
    headful: Optional[bool] = None  # Override BROWSER_USE_HEADFUL setting
    use_custom_chrome: Optional[bool] = (
        None  # Whether to use custom Chrome from env vars
    )


class TaskResponse(BaseModel):
    id: str
    status: str
    live_url: str


class TaskStatusResponse(BaseModel):
    status: str
    result: Optional[str] = None
    error: Optional[str] = None


class RewardRequest(BaseModel):
    manual_score: float = Field(..., ge=-1.0, le=1.0)
    reason: Optional[str] = None


# Dependency to get user_id from headers
async def get_user_id(x_user_id: Optional[str] = Header(None)) -> str:
    """Extract user ID from header or use default"""
    return x_user_id or DEFAULT_USER_ID


# Utility functions
def get_llm(ai_provider: str):
    """Get LLM based on provider"""
    if ai_provider == "anthropic":
        return ChatAnthropic(
            model=os.environ.get("ANTHROPIC_MODEL_ID", "claude-3-opus-20240229")
        )
    # elif ai_provider == "mistral":
    #     return LLMProvider.MISTRAL(
    #         model=os.environ.get("MISTRAL_MODEL_ID", "mistral-large-latest")
    #     )
    elif ai_provider == "google":
        return ChatGoogle(model=os.environ.get("GOOGLE_MODEL_ID", "gemini-1.5-pro"))
    elif ai_provider == "ollama":
        return ChatOllama(model=os.environ.get("OLLAMA_MODEL_ID", "llama3"))
    elif ai_provider == "deepseek":
        return LCChatOpenAI(
            model=os.environ.get("DEEPSEEK_MODEL_ID", "deepseek-chat"),
            base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
            api_key=os.environ.get("DEEPSEEK_API_KEY"),
        )
    elif ai_provider == "azure":
        return ChatAzureOpenAI(
            model=os.environ.get("AZURE_MODEL_ID", "gpt-4o"),
            azure_deployment=os.environ.get("AZURE_DEPLOYMENT_NAME"),
            api_version=os.environ.get("AZURE_API_VERSION", "2023-05-15"),
            azure_endpoint=os.environ.get("AZURE_ENDPOINT"),
        )
    elif ai_provider == "bedrock":
        return ChatAWSBedrock(
            model=os.environ.get(
                "BEDROCK_MODEL_ID", "anthropic.claude-3-sonnet-20240229-v1:0"
            )
        )
    else:  # default to OpenAI
        base_url = os.environ.get("OPENAI_BASE_URL")
        model = os.environ.get("OPENAI_MODEL_ID", "gpt-4o")

        if base_url:
            return ChatOpenAI(model=model, base_url=base_url)
        else:
            return ChatOpenAI(model=model)


def process_screenshot_data(screenshot_data) -> Optional[bytes]:
    """Convert screenshot data from various formats to bytes"""
    if not screenshot_data:
        logger.warning("No screenshot data provided")
        return None

    image_data = None
    if isinstance(screenshot_data, bytes):
        image_data = screenshot_data
    elif isinstance(screenshot_data, str):
        # Clean base64 data (remove data URL prefix if present)
        if screenshot_data.startswith("data:image/"):
            screenshot_data = screenshot_data.split(",", 1)[1]
        try:
            image_data = base64.b64decode(screenshot_data)
        except Exception as decode_error:
            logger.error(f"Failed to decode screenshot data: {decode_error}")
            return None
    else:
        logger.error(f"Unexpected screenshot data type: {type(screenshot_data)}")
        return None

    return image_data


def check_duplicate_screenshot(image_data: bytes, task_id: str) -> bool:
    """Check if screenshot is a duplicate based on size tolerance"""
    if not image_data:
        return False

    current_size = len(image_data)
    logger.debug(f"Current screenshot size: {current_size} bytes")

    # Check existing screenshots in the task media directory with size tolerance
    task_media_dir = MEDIA_DIR / task_id
    if task_media_dir.exists():
        existing_screenshots = list(task_media_dir.glob("*.png"))
        for existing_file in existing_screenshots:
            try:
                existing_size = existing_file.stat().st_size

                # Calculate size difference tolerance (0.5% of the larger size, min 1KB, max 10KB)
                larger_size = max(existing_size, current_size)
                size_tolerance = max(1024, min(10240, int(larger_size * 0.005)))
                size_diff = abs(existing_size - current_size)

                if size_diff <= size_tolerance:
                    logger.info(
                        f"Duplicate screenshot detected - size {current_size} bytes is within {size_tolerance} bytes of {existing_file.name} ({existing_size} bytes), difference: {size_diff} bytes"
                    )
                    return True
            except Exception as stat_error:
                logger.warning(f"Could not check size of {existing_file}: {stat_error}")
                continue

    return False


def validate_and_save_screenshot(
    image_data: bytes, task_id: str, user_id: str, task_status: str = None
) -> Optional[str]:
    """Validate PNG data and save screenshot with appropriate filename"""
    if not image_data:
        return None

    # Validate PNG header
    png_signature = b"\x89PNG\r\n\x1a\n"
    if not image_data.startswith(png_signature):
        logger.warning("Image data does not have valid PNG signature, skipping save")
        return None

    # Create task media directory
    task_media_dir = MEDIA_DIR / task_id
    task_media_dir.mkdir(exist_ok=True, parents=True)

    # Generate filename
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    task = task_storage.get_task(task_id, user_id)

    if task_status == TaskStatus.FINISHED or (
        task and task["status"] == TaskStatus.FINISHED
    ):
        screenshot_filename = f"final-{timestamp}.png"
    elif task_status == TaskStatus.RUNNING or (
        task and task["status"] == TaskStatus.RUNNING
    ):
        # Number running status screenshots from 1..n based on existing status-step captures.
        status_step_count = 0

        if task:
            for media_entry in task.get("media", []):
                if not isinstance(media_entry, dict):
                    continue
                filename = media_entry.get("filename", "")
                if isinstance(filename, str) and filename.startswith("status-step-"):
                    status_step_count += 1

        if status_step_count == 0 and task_media_dir.exists():
            status_step_count = len(list(task_media_dir.glob("status-step-*.png")))

        current_step = status_step_count + 1
        screenshot_filename = f"status-step-{current_step}-{timestamp}.png"
    else:
        task_status_str = task_status or (task["status"] if task else "unknown")
        screenshot_filename = f"status-{task_status_str}-{timestamp}.png"

    screenshot_path = task_media_dir / screenshot_filename

    # Save the screenshot
    try:
        with open(screenshot_path, "wb") as f:
            f.write(image_data)

        if screenshot_path.exists() and screenshot_path.stat().st_size > 0:
            logger.info(
                f"New unique screenshot saved: {screenshot_path} ({screenshot_path.stat().st_size} bytes)"
            )
            # Add to task storage
            screenshot_url = f"/media/{task_id}/{screenshot_filename}"
            media_entry = {
                "url": screenshot_url,
                "type": "screenshot",
                "filename": screenshot_filename,
                "created_at": datetime.now(UTC).isoformat() + "Z",
            }
            task_storage.add_task_media(task_id, media_entry, user_id)
            return screenshot_url
        else:
            logger.error(f"Screenshot file not created or empty: {screenshot_path}")
            return None
    except Exception as save_error:
        logger.error(f"Error saving screenshot: {save_error}")
        return None


def configure_browser_profile(
    task_browser_config: dict,
) -> tuple[Optional[Browser], dict]:
    """Configure browser based on task and environment settings"""
    # Configure browser headless/headful mode (task setting overrides env var)
    task_headful = task_browser_config.get("headful")
    if task_headful is not None:
        headful = task_headful
    else:
        headful = os.environ.get("BROWSER_USE_HEADFUL", "false").lower() == "true"

    # Get Chrome path and user data directory (task settings override env vars)
    use_custom_chrome = task_browser_config.get("use_custom_chrome")

    if use_custom_chrome is False:
        chrome_path = None
        chrome_user_data = None
    else:
        chrome_path = os.environ.get("CHROME_PATH")
        chrome_user_data = os.environ.get("CHROME_USER_DATA")

    browser = None
    browser_info = {
        "headful": headful,
        "chrome_path": chrome_path,
        "chrome_user_data": chrome_user_data,
    }

    # Only configure browser if we need custom setup
    if not headful or chrome_path:
        extra_chromium_args = ["--headless=new"]
        browser_config_args = {
            "headless": not headful,
            "chrome_instance_path": None,
            "viewport": {"width": 1280, "height": 720},
            "window_size": {"width": 1280, "height": 720},
        }

        if chrome_path and chrome_path.lower() != "false":
            browser_config_args["chrome_instance_path"] = chrome_path
            logger.info(f"Using custom Chrome executable: {chrome_path}")

        if chrome_user_data:
            extra_chromium_args.append(f"--user-data-dir={chrome_user_data}")
            logger.info(f"Using Chrome user data directory: {chrome_user_data}")

        browser_config = BrowserProfile(**browser_config_args)
        browser = Browser(browser_profile=browser_config)
        browser_info["browser_config_args"] = browser_config_args

    return browser, browser_info


def prepare_task_environment(task_id: str, user_id: str):
    """Prepare task environment and media directory"""
    # Create task media directory up front
    task_media_dir = MEDIA_DIR / task_id
    task_media_dir.mkdir(exist_ok=True, parents=True)
    logger.info(f"Created media directory for task {task_id}: {task_media_dir}")
    return task_media_dir


def get_sensitive_data():
    """Extract sensitive data from environment variables"""
    if not PASS_SENSITIVE_DATA_TO_AGENT:
        return {}

    sensitive_data = {}
    for key, value in os.environ.items():
        if key.startswith("X_") and value:
            sensitive_data[key] = value
    return sensitive_data


def resolve_use_vision(ai_provider: str) -> bool:
    """Resolve vision usage with env override and provider-safe defaults."""
    configured = os.environ.get("BROWSER_USE_VISION")
    if configured is not None:
        return configured.lower() == "true"

    # Safe default for providers/models that are often text-only.
    if ai_provider in {"ollama", "deepseek"}:
        return False

    return True


def build_agent_task(instruction: str) -> str:
    """Optionally wrap task with constraints that reduce tool-chatter loops."""
    json_output_contract = (
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

    if not AGENT_ENFORCE_CONCISE_EXECUTION:
        return f"{instruction}\n\n{json_output_contract}"

    guardrails = (
        "Execution constraints:\n"
        "- Prefer the shortest path to completion.\n"
        "- Do not use input_text, write_file, or unrelated actions unless explicitly required by the user task.\n"
        "- If the requested information is already available, return final answer immediately.\n"
        "- Avoid repeating the same action pattern after a failed attempt; choose a different strategy or finish with a clear failure reason.\n"
        "- Keep total actions minimal.\n"
        "- After any successful extraction (extract tool or find_elements), format result as JSON and call done() immediately—do not continue with more steps."
    )
    return f"{instruction}\n\n{guardrails}\n\n{json_output_contract}"


def stop_task_for_guardrail(
    agent,
    task_id: str,
    user_id: str,
    reason: str,
    event_type: str,
    event_details: dict,
):
    """Stop an active task when a guardrail condition is met."""
    logger.error(f"Task {task_id} {reason}")
    task_storage.set_task_error(task_id, reason, user_id)
    task_storage.update_task_status(task_id, TaskStatus.STOPPING, user_id)
    add_trajectory_event(task_id, user_id, event_type, event_details)

    try:
        agent.stop()
    except Exception as stop_error:
        logger.warning(f"Failed to stop agent after guardrail trigger: {stop_error}")


async def check_extraction_complete(
    agent,
    task_id: str,
    user_id: str,
    initial_extracted_count: int = 0,
) -> bool:
    """Stop only when *new* extracted content appears after task start.

    Some browser-use versions may have non-empty extracted history at initialization,
    so we compare against an initial baseline to avoid premature stop on step 1.
    """
    if not hasattr(agent, "history") or agent.history is None:
        return False

    try:
        extracted_items = agent.history.extracted_content()
        if not extracted_items:
            return False

        # Filter out empty/error artifacts
        meaningful = [
            item
            for item in extracted_items
            if item and not is_unhelpful_output(str(item))
        ]
        if not meaningful:
            return False

        latest_output = str(meaningful[-1])
        if not is_high_confidence_extraction_output(latest_output):
            logger.debug(
                f"Task {task_id}: latest extracted item is not a high-confidence final extraction signal; continuing"
            )
            return False

        new_item_count = len(meaningful) - max(0, initial_extracted_count)
        if new_item_count <= 0:
            return False

        logger.info(
            f"Task {task_id}: extraction complete with {new_item_count} new result(s), stopping agent to avoid unnecessary LLM calls"
        )

        # Capture final screenshot and format result in parallel (while agent is still active)
        try:
            results = await asyncio.gather(
                capture_screenshot(agent, task_id, user_id),
                format_extraction_result_with_llm(
                    latest_output, getattr(agent, "llm", None), task_id, user_id
                ),
                return_exceptions=True,
            )
            # results[1] is the formatted output from LLM
            clean_output = (
                results[1]
                if isinstance(results[1], str)
                else extract_result_from_wrapper(latest_output)
            )
        except Exception as e:
            logger.debug(f"Error in parallel extraction formatting: {e}")
            clean_output = extract_result_from_wrapper(latest_output)

        task_storage.set_task_output(task_id, clean_output, user_id)
        add_trajectory_event(
            task_id,
            user_id,
            "extraction_auto_stop",
            {
                "extracted_items_count": len(meaningful),
                "initial_extracted_count": initial_extracted_count,
                "new_extracted_items_count": new_item_count,
            },
        )
        agent.stop()
        return True
    except Exception as e:
        logger.debug(f"check_extraction_complete error for task {task_id}: {e}")
        return False


def extract_first_url(text: str) -> Optional[str]:
    """Extract the first URL from a task instruction."""
    match = re.search(r"https?://[^\s)\]]+", text)
    if not match:
        return None
    return match.group(0).rstrip(".,")


def is_unhelpful_output(output: Optional[str]) -> bool:
    """Determine whether model output is empty or clearly a tool-error artifact."""
    if not output:
        return True

    normalized = output.strip().lower()
    if not normalized:
        return True

    noisy_markers = [
        "file '",
        "not found",
        "error executing action",
        "action '",
    ]
    return any(marker in normalized for marker in noisy_markers)


async def format_extraction_result_with_llm(
    raw_output: str,
    llm,
    task_id: str,
    user_id: str,
) -> str:
    """Format extracted result as structured JSON with summary and raw_data sections."""
    if not llm or not raw_output:
        return extract_result_from_wrapper(raw_output)

    prompt = (
        """You are a data extraction formatter. Parse the following extracted information and return ONLY a valid JSON object with:
1. "summary": A brief 1-2 sentence summary of key findings
2. "structured_data": An object with clearly named fields for each value
3. "raw_data": The original extraction preserved exactly as provided

Return ONLY valid JSON, no markdown, no code fences.

Extracted information:
"""
        + raw_output
    )

    try:
        from langchain_core.messages import HumanMessage

        response = await llm.ainvoke([HumanMessage(content=prompt)])
        result_text = (
            response.content if hasattr(response, "content") else str(response)
        )

        try:
            parsed = json.loads(result_text)
            return json.dumps(parsed, indent=2)
        except json.JSONDecodeError:
            json_match = re.search(r"\{.*\}", result_text, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group(0))
                return json.dumps(parsed, indent=2)
            return result_text
    except Exception as e:
        logger.debug(f"LLM formatting failed for task {task_id}, falling back: {e}")
        return extract_result_from_wrapper(raw_output)


def extract_result_from_wrapper(raw_output: str) -> str:
    """Extract clean result content from tool wrapper XML.

    If output contains <result>...</result>, extract that.
    Otherwise return the input as-is.
    """
    if not raw_output:
        return raw_output

    result_match = re.search(
        r"<result>(.*?)</result>", raw_output, flags=re.IGNORECASE | re.DOTALL
    )
    if result_match:
        content = result_match.group(1).strip()
        if content.startswith("{") and content.endswith("}"):
            try:
                parsed = json.loads(content)
                return json.dumps(parsed, indent=2)
            except json.JSONDecodeError:
                pass
        return content
    return raw_output


def is_high_confidence_extraction_output(output: Optional[str]) -> bool:
    """Return True only for outputs that look like final extracted answers.

    This rejects common action/tool chatter (for example go_to_url navigation
    responses) so step-start guardrails do not stop tasks prematurely.
    """
    if not output:
        return False

    text = output.strip()
    if not text:
        return False

    normalized = text.lower()

    # Strong signal from extract_structured_data tool payload shape.
    if "<result>" in normalized and "</result>" in normalized:
        return True

    # Accept explicit JSON-shaped final answers.
    if text.startswith("{") and text.endswith("}"):
        return True

    # Reject common navigation/action artifacts.
    action_markers = [
        "navigated to",
        "go_to_url",
        "clicked",
        "typing",
        "typed",
        "scroll",
        "opened",
        "new tab",
        "switched tab",
    ]
    if any(marker in normalized for marker in action_markers):
        return False

    return False


async def try_simple_title_shortcut(
    agent,
    instruction: str,
    task_id: str,
    user_id: str,
) -> Optional[str]:
    """Handle simple title-extraction tasks with deterministic browser operations."""
    if not ENABLE_SIMPLE_TITLE_SHORTCUT:
        return None

    instruction_lc = instruction.lower()
    if "title" not in instruction_lc:
        return None

    target_url = extract_first_url(instruction)
    if not target_url:
        return None

    try:
        title = None
        observation = None

        if hasattr(agent, "browser_session") and agent.browser_session is not None:
            # Browser session already initialized – use it directly.
            await agent.browser_session.navigate_to(target_url)
            observation = await collect_browser_observation(agent)
            if observation:
                title = observation.get("title")

                if not title:
                    dom_html = observation.get("dom_html") or ""
                    if dom_html:
                        title_match = re.search(
                            r"<title[^>]*>(.*?)</title>",
                            dom_html,
                            flags=re.IGNORECASE | re.DOTALL,
                        )
                        if title_match:
                            title = title_match.group(1).strip() or None

                    if not title and dom_html:
                        h1_match = re.search(
                            r"<h1[^>]*>(.*?)</h1>",
                            dom_html,
                            flags=re.IGNORECASE | re.DOTALL,
                        )
                        if h1_match:
                            title = h1_match.group(1).strip() or None
        else:
            # Browser session not yet initialized (pre-run).  Use a lightweight
            # HTTP fetch so simple title tasks complete without invoking the LLM.
            import urllib.request as _urllib_req

            def _fetch_html(url: str) -> str:
                req = _urllib_req.Request(
                    url,
                    headers={
                        "User-Agent": "Mozilla/5.0 (compatible; browser-use-bridge/1.0)"
                    },
                )
                with _urllib_req.urlopen(req, timeout=10) as resp:
                    charset = resp.info().get_content_charset() or "utf-8"
                    content = resp.read().decode(charset, errors="replace")
                    return content[:MAX_DOM_SNAPSHOT_CHARS]

            try:
                dom_html = await asyncio.to_thread(_fetch_html, target_url)
            except Exception as fetch_error:
                logger.warning(
                    f"HTTP title shortcut fetch failed for {target_url}: {fetch_error}"
                )
                return None

            title_match = re.search(
                r"<title[^>]*>(.*?)</title>", dom_html, flags=re.IGNORECASE | re.DOTALL
            )
            if title_match:
                title = title_match.group(1).strip() or None

            if not title:
                h1_match = re.search(
                    r"<h1[^>]*>(.*?)</h1>", dom_html, flags=re.IGNORECASE | re.DOTALL
                )
                if h1_match:
                    title = h1_match.group(1).strip() or None

            observation = {
                "url": target_url,
                "title": title,
                "dom_html": dom_html,
                "dom_truncated": len(dom_html) >= MAX_DOM_SNAPSHOT_CHARS,
            }

        if observation:
            observation_entry = {
                "observation_id": 1,
                "step": 1,
                "source": "shortcut_title",
                "timestamp": datetime.now(UTC).isoformat() + "Z",
                "screenshot_url": None,
                **observation,
            }
            task_storage.add_task_observation(task_id, observation_entry, user_id)

        add_trajectory_event(
            task_id,
            user_id,
            "shortcut_title",
            {"url": target_url, "success": bool(title)},
        )

        if title:
            return f"Page title: {title}"
        return f"No page title found at {target_url}."
    except Exception as shortcut_error:
        logger.warning(
            f"Simple title shortcut failed for task {task_id}: {shortcut_error}"
        )
        return None


def create_agent_config(
    instruction: str,
    llm,
    sensitive_data: dict,
    ai_provider: str,
    browser: Optional[Browser] = None,
):
    """Create agent configuration dictionary"""
    agent_kwargs = {
        "task": build_agent_task(instruction),
        "llm": llm,
        "sensitive_data": sensitive_data,
        "use_vision": resolve_use_vision(ai_provider),
    }

    if browser:
        agent_kwargs["browser"] = browser

    return agent_kwargs


def add_trajectory_event(task_id: str, user_id: str, event_type: str, details: dict):
    """Append a normalized trajectory event for later inspection."""
    event = {
        "timestamp": datetime.now(UTC).isoformat() + "Z",
        "event_type": event_type,
        "details": details,
    }
    task_storage.add_task_trajectory_entry(task_id, event, user_id)


def compute_auto_reward(task: Optional[dict]) -> tuple[float, str]:
    """Compute a simple transparent heuristic reward for demo usage."""
    if not task:
        return 0.0, "no task context available"

    status = task.get("status")
    output = task.get("output")
    error = task.get("error")

    if status == TaskStatus.FINISHED and output:
        return 0.8, "task finished with non-empty output"
    if status == TaskStatus.FINISHED:
        return 0.4, "task finished without output"
    if status in [TaskStatus.STOPPED, TaskStatus.STOPPING]:
        return -0.2, "task was stopped before completion"
    if status == TaskStatus.FAILED or error:
        return -0.8, "task failed or produced an error"

    return 0.0, "task is non-terminal or lacks clear success signal"


async def collect_browser_observation(agent) -> Optional[dict]:
    """Capture a lightweight browser state snapshot, including DOM content."""
    if not hasattr(agent, "browser_session") or agent.browser_session is None:
        return None

    browser_session = agent.browser_session
    page = None
    observation = {
        "url": None,
        "title": None,
        "dom_html": None,
        "dom_truncated": False,
    }

    try:
        if hasattr(browser_session, "get_current_page"):
            page = await browser_session.get_current_page()
    except Exception as page_error:
        logger.debug(f"Unable to get current page from browser session: {page_error}")

    if page is not None:
        try:
            observation["url"] = getattr(page, "url", None)
        except Exception:
            observation["url"] = None

        try:
            if hasattr(page, "title"):
                observation["title"] = await page.title()
        except Exception as title_error:
            logger.debug(f"Unable to capture page title: {title_error}")

        try:
            if hasattr(page, "content"):
                dom_html = await page.content()
                if dom_html and len(dom_html) > MAX_DOM_SNAPSHOT_CHARS:
                    dom_html = dom_html[:MAX_DOM_SNAPSHOT_CHARS]
                    observation["dom_truncated"] = True
                observation["dom_html"] = dom_html
        except Exception as dom_error:
            logger.debug(f"Unable to capture page DOM: {dom_error}")

    if not observation["url"] and hasattr(browser_session, "current_url"):
        try:
            observation["url"] = browser_session.current_url
        except Exception:
            observation["url"] = None

    has_signal = any(
        [observation["url"], observation["title"], observation["dom_html"]]
    )
    return observation if has_signal else None


async def record_task_observation(
    agent,
    task_id: str,
    user_id: str,
    screenshot_url: Optional[str],
    source: str,
):
    """Persist a browser observation and corresponding trajectory event."""
    observation = await collect_browser_observation(agent)
    if not observation:
        return

    task = task_storage.get_task(task_id, user_id) or {}
    observation_index = len(task.get("observations", [])) + 1
    task_step_count = len(task.get("steps", []))

    observation_entry = {
        "observation_id": observation_index,
        "step": task_step_count,
        "source": source,
        "timestamp": datetime.now(UTC).isoformat() + "Z",
        "screenshot_url": screenshot_url,
        **observation,
    }
    task_storage.add_task_observation(task_id, observation_entry, user_id)
    add_trajectory_event(
        task_id,
        user_id,
        "observation",
        {
            "observation_id": observation_index,
            "step": task_step_count,
            "source": source,
            "url": observation_entry.get("url"),
            "screenshot_url": screenshot_url,
            "dom_truncated": observation_entry.get("dom_truncated", False),
        },
    )


async def process_task_result(result, task_id: str, user_id: str):
    """Process and store task execution result"""
    if isinstance(result, AgentHistoryList):
        final_result = result.final_result()
        if final_result:
            task_storage.set_task_output(task_id, str(final_result), user_id)
            return

        # Fallback: if model failed to emit done(), use the latest extracted content
        # captured from successful tool actions.
        extracted_items = result.extracted_content()
        if extracted_items:
            task_storage.set_task_output(task_id, str(extracted_items[-1]), user_id)
        else:
            task_storage.set_task_output(task_id, "", user_id)
    else:
        task_storage.set_task_output(task_id, str(result), user_id)


def infer_output_from_observations(task: Optional[dict]) -> Optional[str]:
    """Best-effort summary when model did not emit a final answer."""
    if not task:
        return None

    if task.get("output"):
        return None

    observations = task.get("observations", [])
    if not observations:
        return None

    for observation in reversed(observations):
        dom_html = observation.get("dom_html") or ""

        title = observation.get("title")
        if title:
            return f"Page title: {title}"

        if dom_html:
            title_match = re.search(
                r"<title[^>]*>(.*?)</title>", dom_html, flags=re.IGNORECASE | re.DOTALL
            )
            if title_match and title_match.group(1).strip():
                return f"Page title: {title_match.group(1).strip()}"

            h1_match = re.search(
                r"<h1[^>]*>(.*?)</h1>", dom_html, flags=re.IGNORECASE | re.DOTALL
            )
            if h1_match and h1_match.group(1).strip():
                return f"Page heading: {h1_match.group(1).strip()}"

    latest_url = observations[-1].get("url")
    if latest_url:
        return f"No final model output was returned. Last observed URL: {latest_url}"

    return "No final model output was returned."


async def collect_browser_cookies(agent, task_id: str, user_id: str):
    """Collect browser cookies if requested and available"""
    task = task_storage.get_task(task_id, user_id)
    if (
        not task
        or not task.get("save_browser_data")
        or not hasattr(agent, "browser_session")
    ):
        return

    try:
        cookies = []
        browser_session = None
        try:
            browser_session = agent.browser_session
        except (AssertionError, AttributeError):
            logger.warning(
                f"BrowserSession is not set up for task {task_id}, skipping cookie collection."
            )

        if browser_session and hasattr(browser_session, "get_cookies"):
            cookies = await browser_session.get_cookies()
        else:
            logger.warning(f"No method to collect cookies for task {task_id}")

        task_storage.update_task(
            task_id, {"browser_data": {"cookies": cookies}}, user_id
        )
    except Exception as e:
        logger.error(f"Failed to collect browser data: {str(e)}")
        task_storage.update_task(
            task_id, {"browser_data": {"cookies": [], "error": str(e)}}, user_id
        )


async def cleanup_task(browser: Optional[Browser], task_id: str, user_id: str):
    """Clean up task resources and take final screenshot"""
    if browser is not None:
        logger.info(f"Closing browser for task {task_id}")
        try:
            logger.info(f"Taking final screenshot for task {task_id} after completion")

            # Take final screenshot
            agent = task_storage.get_task_agent(task_id, user_id)
            if agent and hasattr(agent, "browser_session"):
                await capture_screenshot(agent, task_id, user_id)
        except Exception as e:
            logger.error(f"Error taking final screenshot: {str(e)}")
        finally:
            if browser:
                try:
                    if hasattr(browser, "close"):
                        await browser.close()
                    else:
                        logger.info(
                            f"Browser object for task {task_id} has no close() method; skipping explicit browser close"
                        )
                except Exception as e:
                    logger.error(f"Error closing browser for task {task_id}: {str(e)}")


async def execute_task(
    task_id: str, instruction: str, ai_provider: str, user_id: str = DEFAULT_USER_ID
):
    """Execute browser task in background - main orchestration function

    Chrome paths (CHROME_PATH and CHROME_USER_DATA) are only sourced from
    environment variables for security reasons.
    """
    browser = None

    try:
        # Update task status and prepare environment
        task_storage.update_task_status(task_id, TaskStatus.RUNNING, user_id)
        prepare_task_environment(task_id, user_id)

        # Get task configuration
        task = task_storage.get_task(task_id, user_id)
        task_browser_config = task.get("browser_config", {}) if task else {}

        # Set up LLM and browser
        llm = get_llm(ai_provider)
        logger.info(
            f"Task {task_id}: Using ai_provider={ai_provider}, llm_class={llm.__class__.__name__}"
        )
        browser, browser_info = configure_browser_profile(task_browser_config)
        logger.info(f"Task {task_id}: Browser configuration: {browser_info}")

        # Create agent
        sensitive_data = get_sensitive_data()
        agent_config = create_agent_config(
            instruction,
            llm,
            sensitive_data,
            ai_provider,
            browser,
        )
        logger.info(f"Agent config keys: {list(agent_config.keys())}")

        try:
            agent = Agent(**agent_config)
        except TypeError as type_error:
            if "use_vision" in str(type_error):
                logger.warning(
                    "Agent constructor does not support use_vision on this browser-use version; retrying without it"
                )
                agent_config.pop("use_vision", None)
                agent = Agent(**agent_config)
            else:
                raise
        task_storage.set_task_agent(task_id, agent, user_id)

        initial_extracted_count = 0
        try:
            if hasattr(agent, "history") and agent.history is not None:
                initial_extracted_count = len(agent.history.extracted_content() or [])
        except Exception as history_error:
            logger.debug(
                f"Unable to read initial extracted history for task {task_id}: {history_error}"
            )

        shortcut_output = await try_simple_title_shortcut(
            agent,
            instruction,
            task_id,
            user_id,
        )
        if shortcut_output:
            task_storage.set_task_output(
                task_id,
                shortcut_output,
                user_id,
            )
            task_storage.mark_task_finished(task_id, user_id, TaskStatus.FINISHED)
            finished_task = task_storage.get_task(task_id, user_id)
            auto_score, auto_reason = compute_auto_reward(finished_task)
            task_storage.set_task_reward(
                task_id,
                {
                    "auto_score": auto_score,
                    "effective_score": auto_score,
                    "source": "auto",
                    "reason": auto_reason,
                    "updated_at": datetime.now(UTC).isoformat() + "Z",
                },
                user_id,
            )
            add_trajectory_event(
                task_id,
                user_id,
                "reward_auto",
                {"score": auto_score, "reason": auto_reason},
            )
            await collect_browser_cookies(agent, task_id, user_id)
            return

        # Execute task with automated screenshots and guardrails
        async def _on_step_start(agent_instance):
            """Capture screenshots at the beginning of each step."""
            await automated_screenshot(agent_instance, task_id, user_id)

        async def _on_step_end(agent_instance):
            """Stop only after a completed step once extraction is confidently done."""
            await check_extraction_complete(
                agent_instance,
                task_id,
                user_id,
                initial_extracted_count=initial_extracted_count,
            )

        run_kwargs = {
            "on_step_start": lambda agent_instance: asyncio.create_task(
                _on_step_start(agent_instance)
            ),
            "on_step_end": lambda agent_instance: asyncio.create_task(
                _on_step_end(agent_instance)
            ),
        }
        if AGENT_MAX_STEPS > 0:
            run_kwargs["max_steps"] = AGENT_MAX_STEPS

        while True:
            try:
                run_coro = agent.run(**run_kwargs)
                break
            except TypeError as run_type_error:
                error_text = str(run_type_error)
                if "on_step_end" in error_text and "on_step_end" in run_kwargs:
                    logger.warning(
                        "agent.run() does not support on_step_end on this browser-use version; retrying without it"
                    )
                    run_kwargs.pop("on_step_end", None)
                    continue
                if "max_steps" in error_text and "max_steps" in run_kwargs:
                    logger.warning(
                        "agent.run() does not support max_steps on this browser-use version; retrying without it"
                    )
                    run_kwargs.pop("max_steps", None)
                    continue
                raise

        if TASK_RUN_TIMEOUT_SECONDS > 0:
            result = await asyncio.wait_for(run_coro, timeout=TASK_RUN_TIMEOUT_SECONDS)
        else:
            result = await run_coro

        # Process results
        await process_task_result(result, task_id, user_id)
        post_run_task = task_storage.get_task(task_id, user_id)
        inferred_output = infer_output_from_observations(post_run_task)
        if inferred_output and is_unhelpful_output((post_run_task or {}).get("output")):
            task_storage.set_task_output(
                task_id,
                inferred_output,
                user_id,
            )

        post_run_status = (post_run_task or {}).get("status")
        if post_run_status in [TaskStatus.STOPPING, TaskStatus.STOPPED]:
            task_storage.mark_task_finished(task_id, user_id, TaskStatus.STOPPED)
        else:
            task_storage.mark_task_finished(task_id, user_id, TaskStatus.FINISHED)

        finished_task = task_storage.get_task(task_id, user_id)
        auto_score, auto_reason = compute_auto_reward(finished_task)
        task_storage.set_task_reward(
            task_id,
            {
                "auto_score": auto_score,
                "effective_score": auto_score,
                "source": "auto",
                "reason": auto_reason,
                "updated_at": datetime.now(UTC).isoformat() + "Z",
            },
            user_id,
        )
        add_trajectory_event(
            task_id,
            user_id,
            "reward_auto",
            {"score": auto_score, "reason": auto_reason},
        )
        await collect_browser_cookies(agent, task_id, user_id)

    except Exception as e:
        if isinstance(e, asyncio.TimeoutError):
            logger.error(
                f"Task {task_id} timed out after {TASK_RUN_TIMEOUT_SECONDS}s; stopping task"
            )
            agent = task_storage.get_task_agent(task_id, user_id)
            timeout_extracted_output = None

            # Prefer already-extracted content from agent history when available.
            try:
                if agent and hasattr(agent, "history") and agent.history is not None:
                    extracted_items = agent.history.extracted_content()
                    if extracted_items:
                        timeout_extracted_output = str(extracted_items[-1])
            except Exception as history_error:
                logger.debug(
                    f"Could not read extracted content from history for task {task_id}: {history_error}"
                )

            if agent:
                try:
                    agent.stop()
                except Exception as stop_error:
                    logger.warning(f"Failed to stop timed out agent: {stop_error}")

            timed_out_task = task_storage.get_task(task_id, user_id)
            inferred_output = (
                timeout_extracted_output
                or infer_output_from_observations(timed_out_task)
            )
            if inferred_output and is_unhelpful_output(
                (timed_out_task or {}).get("output")
            ):
                output_to_store = inferred_output
                task_storage.set_task_output(task_id, output_to_store, user_id)

            timed_out_task = task_storage.get_task(task_id, user_id)
            has_useful_output = not is_unhelpful_output(
                (timed_out_task or {}).get("output")
            )

            if has_useful_output:
                # If we recovered meaningful output before timeout, finalize as success.
                task_storage.update_task(task_id, {"error": None}, user_id)
                task_storage.mark_task_finished(task_id, user_id, TaskStatus.FINISHED)
                add_trajectory_event(
                    task_id,
                    user_id,
                    "timeout_recovered",
                    {
                        "timeout_seconds": TASK_RUN_TIMEOUT_SECONDS,
                        "status": TaskStatus.FINISHED,
                    },
                )
            else:
                task_storage.update_task_status(task_id, TaskStatus.STOPPED, user_id)
                task_storage.set_task_error(
                    task_id,
                    f"Task timed out after {TASK_RUN_TIMEOUT_SECONDS} seconds",
                    user_id,
                )
                task_storage.mark_task_finished(task_id, user_id, TaskStatus.STOPPED)

            timed_out_task = task_storage.get_task(task_id, user_id)
            auto_score, auto_reason = compute_auto_reward(timed_out_task)
            task_storage.set_task_reward(
                task_id,
                {
                    "auto_score": auto_score,
                    "effective_score": auto_score,
                    "source": "auto",
                    "reason": auto_reason,
                    "updated_at": datetime.now(UTC).isoformat() + "Z",
                },
                user_id,
            )
            add_trajectory_event(
                task_id,
                user_id,
                "reward_auto",
                {"score": auto_score, "reason": auto_reason},
            )
            return

        logger.exception(f"Error executing task {task_id}")
        task_storage.update_task_status(task_id, TaskStatus.FAILED, user_id)
        task_storage.set_task_error(task_id, str(e), user_id)
        task_storage.mark_task_finished(task_id, user_id, TaskStatus.FAILED)
        failed_task = task_storage.get_task(task_id, user_id)
        auto_score, auto_reason = compute_auto_reward(failed_task)
        task_storage.set_task_reward(
            task_id,
            {
                "auto_score": auto_score,
                "effective_score": auto_score,
                "source": "auto",
                "reason": auto_reason,
                "updated_at": datetime.now(UTC).isoformat() + "Z",
            },
            user_id,
        )
        add_trajectory_event(
            task_id,
            user_id,
            "reward_auto",
            {"score": auto_score, "reason": auto_reason},
        )
    finally:
        await cleanup_task(browser, task_id, user_id)


# API Routes
@app.post("/api/v1/run-task", response_model=TaskResponse)
async def run_task(request: TaskRequest, user_id: str = Depends(get_user_id)):
    """Start a browser automation task"""
    task_id = str(uuid.uuid4())
    now = datetime.now(UTC).isoformat() + "Z"

    # Generate live URL
    live_url = f"/live/{task_id}"

    # Initialize task record
    task_data = {
        "id": task_id,
        "task": request.task,
        "ai_provider": request.ai_provider,
        "status": TaskStatus.CREATED,
        "created_at": now,
        "finished_at": None,
        "output": None,  # Final result
        "error": None,
        "steps": [],  # Will store step information
        "observations": [],  # Browser snapshots captured during execution
        "trajectory": [],  # Timeline of observation and reward events
        "reward": {
            "auto_score": None,
            "manual_score": None,
            "effective_score": None,
            "source": None,
            "reason": None,
            "updated_at": None,
        },
        "agent": None,
        "save_browser_data": request.save_browser_data,
        "browser_data": None,  # Will store browser cookies if requested
        # Store browser configuration options
        "browser_config": {
            "headful": request.headful,
            "use_custom_chrome": request.use_custom_chrome,
        },
        "last_status_capture_epoch": None,
        "consecutive_duplicate_screenshots": 0,
        "screenshot_error_count": 0,
        "live_url": live_url,
    }

    # Store the task in storage
    task_storage.create_task(task_id, task_data, user_id)

    # Start task in background
    ai_provider = request.ai_provider or "openai"
    asyncio.create_task(execute_task(task_id, request.task, ai_provider, user_id))

    return TaskResponse(id=task_id, status=TaskStatus.CREATED, live_url=live_url)


async def automated_screenshot(agent, task_id, user_id=DEFAULT_USER_ID):
    """Take automated screenshot during task execution with duplicate detection"""
    # Only proceed if browser_session is set up
    if not hasattr(agent, "browser_session") or agent.browser_session is None:
        logger.warning(
            f"Agent browser_session not set up for task {task_id}, skipping screenshot."
        )
        return

    try:
        # Take the screenshot
        try:
            screenshot_data = await agent.browser_session.take_screenshot(
                full_page=True
            )
            if not screenshot_data:
                logger.warning(f"No screenshot data returned for task {task_id}")
                return
        except Exception as screenshot_error:
            logger.error(
                f"Failed to take screenshot for task {task_id}: {screenshot_error}"
            )

            task = task_storage.get_task(task_id, user_id) or {}
            screenshot_error_count = int(task.get("screenshot_error_count", 0)) + 1
            task_storage.update_task(
                task_id,
                {"screenshot_error_count": screenshot_error_count},
                user_id,
            )

            if (
                LOOP_GUARD_MAX_SCREENSHOT_ERRORS > 0
                and screenshot_error_count >= LOOP_GUARD_MAX_SCREENSHOT_ERRORS
            ):
                stop_task_for_guardrail(
                    agent,
                    task_id,
                    user_id,
                    reason=(
                        "loop guard triggered: repeated screenshot capture errors indicate unstable page state"
                    ),
                    event_type="loop_guard",
                    event_details={
                        "screenshot_errors": screenshot_error_count,
                        "error": str(screenshot_error),
                    },
                )
            return

        # Process screenshot data
        image_data = process_screenshot_data(screenshot_data)
        if not image_data:
            logger.warning(f"No image data processed for task {task_id}")
            return

        # Check for duplicates
        if check_duplicate_screenshot(image_data, task_id):
            logger.info(f"Skipping duplicate screenshot for task {task_id}")
            task = task_storage.get_task(task_id, user_id) or {}
            duplicate_count = int(task.get("consecutive_duplicate_screenshots", 0)) + 1
            task_storage.update_task(
                task_id,
                {"consecutive_duplicate_screenshots": duplicate_count},
                user_id,
            )

            if (
                LOOP_GUARD_MAX_CONSECUTIVE_DUPLICATE_SCREENSHOTS > 0
                and duplicate_count >= LOOP_GUARD_MAX_CONSECUTIVE_DUPLICATE_SCREENSHOTS
            ):
                stop_task_for_guardrail(
                    agent,
                    task_id,
                    user_id,
                    reason=(
                        "loop guard triggered: repeated duplicate screenshots suggest no progress"
                    ),
                    event_type="loop_guard",
                    event_details={
                        "duplicate_screenshots": duplicate_count,
                        "reason": "repeated duplicate screenshots suggest no progress",
                    },
                )
                add_trajectory_event(
                    task_id,
                    user_id,
                    "loop_guard_metrics",
                    {"duplicate_screenshots": duplicate_count},
                )
            return

        logger.info(f"Taking screenshot for task {task_id}")

        # Progress observed: reset duplicate streak.
        task_storage.update_task(
            task_id,
            {"consecutive_duplicate_screenshots": 0, "screenshot_error_count": 0},
            user_id,
        )

        # Save the screenshot
        screenshot_url = validate_and_save_screenshot(image_data, task_id, user_id)
        if screenshot_url:
            logger.info(f"Screenshot saved successfully: {screenshot_url}")
            await record_task_observation(
                agent,
                task_id,
                user_id,
                screenshot_url,
                source="on_step_start",
            )

    except Exception as e:
        logger.error(f"Error in automated_screenshot for task {task_id}: {str(e)}")


@app.get("/api/v1/task/{task_id}/status", response_model=TaskStatusResponse)
async def get_task_status(task_id: str, user_id: str = Depends(get_user_id)):
    """Get status of a task"""
    task = task_storage.get_task(task_id, user_id)

    agent = task_storage.get_task_agent(task_id, user_id)

    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    # Optional step tracking on status polling (off by default for performance)
    if task["status"] == TaskStatus.RUNNING and STATUS_TRACK_STEPS_ON_POLL:
        # Initialize steps array if not present
        current_step = len(task.get("steps", [])) + 1

        # Add step info
        step_info = {
            "step": current_step,
            "timestamp": datetime.now(UTC).isoformat() + "Z",
            "next_goal": f"Progress check {current_step}",
            "evaluation_previous_goal": "In progress",
        }

        task_storage.add_task_step(task_id, step_info, user_id)
        logger.info(f"Added step {current_step} for task {task_id}")
        add_trajectory_event(
            task_id,
            user_id,
            "step",
            {
                "step": current_step,
                "next_goal": step_info.get("next_goal"),
                "evaluation_previous_goal": step_info.get("evaluation_previous_goal"),
            },
        )

    if STATUS_CAPTURE_SCREENSHOT:
        now_epoch = datetime.now(UTC).timestamp()
        last_capture_epoch = task.get("last_status_capture_epoch") or 0
        if (
            now_epoch - float(last_capture_epoch)
            >= STATUS_SCREENSHOT_MIN_INTERVAL_SECONDS
        ):
            try:
                _ = agent.browser_session
                await capture_screenshot(agent, task_id, user_id)
                task_storage.update_task(
                    task_id, {"last_status_capture_epoch": now_epoch}, user_id
                )
            except (AssertionError, AttributeError):
                logger.info(
                    f"BrowserSession not ready for task {task_id}, skipping screenshot."
                )

    return TaskStatusResponse(
        status=task["status"],
        result=task.get("output"),
        error=task.get("error"),
    )


async def capture_screenshot(agent_or_context, task_id, user_id=DEFAULT_USER_ID):
    """Capture screenshot with flexible input handling and duplicate detection"""
    logger.info(f"Capturing screenshot for task: {task_id}")

    # Handle different input types to get browser_session
    browser_session = None
    if hasattr(agent_or_context, "browser_session"):
        browser_session = getattr(agent_or_context, "browser_session", None)
    elif hasattr(agent_or_context, "take_screenshot"):
        browser_session = agent_or_context
    else:
        logger.warning(f"Unable to determine browser session type for task {task_id}")
        return

    if browser_session is None:
        logger.warning(f"No browser session available for task {task_id}")
        return

    if not hasattr(browser_session, "take_screenshot"):
        logger.error(
            f"browser_session does not have take_screenshot method for task {task_id}"
        )
        return

    try:
        # Check if browser session is still active before trying to take screenshot
        try:
            # Try to access a simple property to check if session is alive
            if (
                hasattr(browser_session, "is_connected")
                and not browser_session.is_connected()
            ):
                logger.info(
                    f"Browser session disconnected for task {task_id}, skipping screenshot"
                )
                return
        except Exception:
            # If we can't check connection status, we'll try the screenshot anyway
            pass

        # Take screenshot
        try:
            screenshot_data = await browser_session.take_screenshot(full_page=True)
            if not screenshot_data:
                logger.warning(f"No screenshot data returned for task {task_id}")
                return
        except Exception as screenshot_error:
            # Check if this is a CDP/connection related error
            error_msg = str(screenshot_error).lower()
            if any(
                keyword in error_msg
                for keyword in ["cdp", "connection", "websocket", "browser", "closed"]
            ):
                logger.info(
                    f"Browser session closed for task {task_id}, cannot take screenshot: {screenshot_error}"
                )
            else:
                logger.error(
                    f"Failed to take screenshot for task {task_id}: {screenshot_error}"
                )
            return

        # Process screenshot data
        image_data = process_screenshot_data(screenshot_data)
        if not image_data:
            logger.warning(f"No image data processed for task {task_id}")
            return

        # Check for duplicates
        if check_duplicate_screenshot(image_data, task_id):
            logger.info(
                f"Skipping duplicate screenshot in capture_screenshot for task {task_id}"
            )
            return

        # Save the screenshot
        screenshot_url = validate_and_save_screenshot(image_data, task_id, user_id)
        if screenshot_url:
            logger.info(f"Screenshot captured and saved successfully: {screenshot_url}")

    except Exception as e:
        logger.error(f"Error in capture_screenshot for task {task_id}: {str(e)}")


@app.get("/api/v1/task/{task_id}", response_model=dict)
async def get_task(task_id: str, user_id: str = Depends(get_user_id)):
    """Get full task details"""
    task = task_storage.get_task(task_id, user_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    return task


@app.post("/api/v1/task/{task_id}/reward")
async def set_task_reward(
    task_id: str,
    request: RewardRequest,
    user_id: str = Depends(get_user_id),
):
    """Set a manual reward score for a task and override effective reward."""
    task = task_storage.get_task(task_id, user_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    reward = task.get("reward", {})
    auto_score = reward.get("auto_score") if isinstance(reward, dict) else None
    manual_score = request.manual_score
    now = datetime.now(UTC).isoformat() + "Z"

    task_storage.set_task_reward(
        task_id,
        {
            "auto_score": auto_score,
            "manual_score": manual_score,
            "effective_score": manual_score,
            "source": "manual",
            "reason": request.reason,
            "updated_at": now,
        },
        user_id,
    )
    add_trajectory_event(
        task_id,
        user_id,
        "reward_manual",
        {
            "score": manual_score,
            "reason": request.reason,
        },
    )

    updated_task = task_storage.get_task(task_id, user_id)
    return {
        "message": "Reward updated",
        "task_id": task_id,
        "reward": updated_task.get("reward", {}),
    }


@app.put("/api/v1/stop-task/{task_id}")
async def stop_task(task_id: str, user_id: str = Depends(get_user_id)):
    """Stop a running task"""
    task = task_storage.get_task(task_id, user_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    if task["status"] in [
        TaskStatus.FINISHED,
        TaskStatus.FAILED,
        TaskStatus.STOPPED,
    ]:
        return {"message": f"Task already in terminal state: {task['status']}"}

    # Get agent
    agent = task_storage.get_task_agent(task_id, user_id)
    if agent:
        # Call agent's stop method
        agent.stop()
        task_storage.update_task_status(task_id, TaskStatus.STOPPING, user_id)
        return {"message": "Task stopping"}
    else:
        task_storage.update_task_status(task_id, TaskStatus.STOPPED, user_id)
        task_storage.mark_task_finished(task_id, user_id, TaskStatus.STOPPED)
        return {"message": "Task stopped (no agent found)"}


@app.put("/api/v1/pause-task/{task_id}")
async def pause_task(task_id: str, user_id: str = Depends(get_user_id)):
    """Pause a running task"""
    task = task_storage.get_task(task_id, user_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    if task["status"] != TaskStatus.RUNNING:
        return {"message": f"Task not running: {task['status']}"}

    # Get agent
    agent = task_storage.get_task_agent(task_id, user_id)
    if agent:
        # Call agent's pause method
        agent.pause()
        task_storage.update_task_status(task_id, TaskStatus.PAUSED, user_id)
        return {"message": "Task paused"}
    else:
        return {"message": "Task could not be paused (no agent found)"}


@app.put("/api/v1/resume-task/{task_id}")
async def resume_task(task_id: str, user_id: str = Depends(get_user_id)):
    """Resume a paused task"""
    task = task_storage.get_task(task_id, user_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    if task["status"] != TaskStatus.PAUSED:
        return {"message": f"Task not paused: {task['status']}"}

    # Get agent
    agent = task_storage.get_task_agent(task_id, user_id)
    if agent:
        # Call agent's resume method
        agent.resume()
        task_storage.update_task_status(task_id, TaskStatus.RUNNING, user_id)
        return {"message": "Task resumed"}
    else:
        return {"message": "Task could not be resumed (no agent found)"}


@app.get("/api/v1/list-tasks")
async def list_tasks(
    user_id: str = Depends(get_user_id),
    page: int = Query(1, ge=1),
    per_page: int = Query(100, ge=1, le=1000),
):
    """List all tasks"""
    return task_storage.list_tasks(user_id, page, per_page)


@app.get("/live/{task_id}", response_class=HTMLResponse)
async def live_view(task_id: str, user_id: str = Depends(get_user_id)):
    """Get a live view of a task that can be embedded in an iframe"""
    task = task_storage.get_task(task_id, user_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Browser Use Task {task_id}</title>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; }}
            .container {{ max-width: 1200px; margin: 0 auto; }}
            .status {{ padding: 10px; border-radius: 4px; margin-bottom: 20px; }}
            .{TaskStatus.RUNNING} {{ background-color: #e3f2fd; }}
            .{TaskStatus.FINISHED} {{ background-color: #e8f5e9; }}
            .{TaskStatus.FAILED} {{ background-color: #ffebee; }}
            .{TaskStatus.PAUSED} {{ background-color: #fff8e1; }}
            .{TaskStatus.STOPPED} {{ background-color: #eeeeee; }}
            .{TaskStatus.CREATED} {{ background-color: #f3e5f5; }}
            .{TaskStatus.STOPPING} {{ background-color: #fce4ec; }}
            .controls {{ margin-bottom: 20px; }}
            button {{ padding: 8px 16px; margin-right: 10px; cursor: pointer; }}
            pre {{ background-color: #f5f5f5; padding: 15px; border-radius: 4px; overflow: auto; }}
            .step {{ margin-bottom: 10px; padding: 10px; border: 1px solid #ddd; border-radius: 4px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Browser Use Task</h1>
            <div id="status" class="status">Loading...</div>
            
            <div class="controls">
                <button id="pauseBtn">Pause</button>
                <button id="resumeBtn">Resume</button>
                <button id="stopBtn">Stop</button>
            </div>
            
            <h2>Result</h2>
            <pre id="result">Loading...</pre>
            
            <h2>Steps</h2>
            <div id="steps">Loading...</div>
            
            <script>
                const taskId = '{task_id}';
                const FINISHED = '{TaskStatus.FINISHED}';
                const FAILED = '{TaskStatus.FAILED}';
                const STOPPED = '{TaskStatus.STOPPED}';
                const userId = '{user_id}';
                
                // Set user ID in request headers if available
                const headers = {{}};
                if (userId && userId !== 'default') {{
                    headers['X-User-ID'] = userId;
                }}
                
                // Update status function
                function updateStatus() {{
                    fetch(`/api/v1/task/${{taskId}}/status`, {{ headers }})
                        .then(response => response.json())
                        .then(data => {{
                            // Update status element
                            const statusEl = document.getElementById('status');
                            statusEl.textContent = `Status: ${{data.status}}`;
                            statusEl.className = `status ${{data.status}}`;
                            
                            // Update result if available
                            if (data.result) {{
                                document.getElementById('result').textContent = data.result;
                            }} else if (data.error) {{
                                document.getElementById('result').textContent = `Error: ${{data.error}}`;
                            }}
                            
                            // Continue polling if not in terminal state
                            if (![FINISHED, FAILED, STOPPED].includes(data.status)) {{
                                setTimeout(updateStatus, 2000);
                            }}
                        }})
                        .catch(error => {{
                            console.error('Error fetching status:', error);
                            setTimeout(updateStatus, 5000);
                        }});
                        
                    // Also fetch full task to get steps
                    fetch(`/api/v1/task/${{taskId}}`, {{ headers }})
                        .then(response => response.json())
                        .then(data => {{
                            if (data.steps && data.steps.length > 0) {{
                                const stepsHtml = data.steps.map(step => `
                                    <div class="step">
                                        <strong>Step ${{step.step}}</strong>
                                        <p>Next Goal: ${{step.next_goal || 'N/A'}}</p>
                                        <p>Evaluation: ${{step.evaluation_previous_goal || 'N/A'}}</p>
                                    </div>
                                `).join('');
                                document.getElementById('steps').innerHTML = stepsHtml;
                            }} else {{
                                document.getElementById('steps').textContent = 'No steps recorded yet.';
                            }}
                        }})
                        .catch(error => {{
                            console.error('Error fetching task details:', error);
                        }});
                }}
                
                // Setup control buttons
                document.getElementById('pauseBtn').addEventListener('click', () => {{
                    fetch(`/api/v1/pause-task/${{taskId}}`, {{ 
                        method: 'PUT',
                        headers
                    }})
                        .then(response => response.json())
                        .then(data => alert(data.message))
                        .catch(error => console.error('Error pausing task:', error));
                }});
                
                document.getElementById('resumeBtn').addEventListener('click', () => {{
                    fetch(`/api/v1/resume-task/${{taskId}}`, {{ 
                        method: 'PUT',
                        headers
                    }})
                        .then(response => response.json())
                        .then(data => alert(data.message))
                        .catch(error => console.error('Error resuming task:', error));
                }});
                
                document.getElementById('stopBtn').addEventListener('click', () => {{
                    if (confirm('Are you sure you want to stop this task? This action cannot be undone.')) {{
                        fetch(`/api/v1/stop-task/${{taskId}}`, {{ 
                            method: 'PUT',
                            headers
                        }})
                            .then(response => response.json())
                            .then(data => alert(data.message))
                            .catch(error => console.error('Error stopping task:', error));
                    }}
                }});
                
                // Start status updates
                updateStatus();
                
                // Refresh every 5 seconds
                setInterval(updateStatus, 5000);
            </script>
        </div>
    </body>
    </html>
    """

    return HTMLResponse(content=html_content)


@app.get("/api/v1/ping")
async def ping():
    """Health check endpoint"""
    return {"status": "success", "message": "API is running"}


@app.get("/api/v1/browser-config")
async def browser_config():
    """Get current browser configuration

    Note: Chrome paths (CHROME_PATH and CHROME_USER_DATA) can only be set via
    environment variables for security reasons and cannot be overridden in task requests.
    """
    headful = os.environ.get("BROWSER_USE_HEADFUL", "false").lower() == "true"
    chrome_path = os.environ.get("CHROME_PATH", None)
    chrome_user_data = os.environ.get("CHROME_USER_DATA", None)

    return {
        "headful": headful,
        "headless": not headful,
        "chrome_path": chrome_path,
        "chrome_user_data": chrome_user_data,
        "using_custom_chrome": chrome_path is not None,
        "using_user_data": chrome_user_data is not None,
    }


@app.get("/api/v1/task/{task_id}/media")
async def get_task_media(
    task_id: str, user_id: str = Depends(get_user_id), type: Optional[str] = None
):
    """Returns links to any recordings or media generated during task execution"""
    task = task_storage.get_task(task_id, user_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    # Check if task is completed
    if task["status"] not in [
        TaskStatus.FINISHED,
        TaskStatus.FAILED,
        TaskStatus.STOPPED,
    ]:
        raise HTTPException(
            status_code=400, detail="Media only available for completed tasks"
        )

    # Check if the media directory exists and contains files
    task_media_dir = MEDIA_DIR / task_id
    media_files = []

    if task_media_dir.exists():
        media_files = list(task_media_dir.glob("*"))
        logger.info(
            f"Media directory for task {task_id} contains {len(media_files)} files: {[f.name for f in media_files]}"
        )
    else:
        logger.warning(f"Media directory for task {task_id} does not exist")

    # If we have files but no media entries, create them now
    if media_files and (not task.get("media") or len(task.get("media", [])) == 0):
        for file_path in media_files:
            if file_path.suffix.lower() in [".png", ".jpg", ".jpeg"]:
                file_url = f"/media/{task_id}/{file_path.name}"
                media_entry = {
                    "url": file_url,
                    "type": "screenshot",
                    "filename": file_path.name,
                }
                task_storage.add_task_media(task_id, media_entry, user_id)

    # Get updated task with media
    task = task_storage.get_task(task_id, user_id)
    if task is not None:
        media_list = task.get("media", [])
    else:
        media_list = []

    # Filter by type if specified
    if type and isinstance(media_list, list):
        if all(isinstance(item, dict) for item in media_list):
            # Dictionary format with type info
            media_list = [item for item in media_list if item.get("type") == type]
            recordings = [item["url"] for item in media_list]
        else:
            # Just URLs without type info
            recordings = []
            logger.warning(
                f"Media list for task {task_id} doesn't contain type information"
            )
    else:
        # Return all media
        if isinstance(media_list, list):
            if media_list and all(isinstance(item, dict) for item in media_list):
                recordings = [item["url"] for item in media_list]
            else:
                recordings = media_list
        else:
            recordings = []

    logger.info(f"Returning {len(recordings)} media items for task {task_id}")
    return {"recordings": recordings}


@app.get("/api/v1/task/{task_id}/media/list")
async def list_task_media(
    task_id: str, user_id: str = Depends(get_user_id), type: Optional[str] = None
):
    """Returns detailed information about media files associated with a task"""
    # Check if the media directory exists
    task_media_dir = MEDIA_DIR / task_id

    if not task_storage.task_exists(task_id, user_id):
        raise HTTPException(status_code=404, detail="Task not found")

    if not task_media_dir.exists():
        return {
            "media": [],
            "count": 0,
            "message": f"No media found for task {task_id}",
        }

    media_info = []

    media_files = list(task_media_dir.glob("*"))
    logger.info(f"Found {len(media_files)} media files for task {task_id}")

    for file_path in media_files:
        # Determine media type based on file extension
        file_type = "unknown"
        if file_path.suffix.lower() in [".png", ".jpg", ".jpeg"]:
            file_type = "screenshot"
        elif file_path.suffix.lower() in [".mp4", ".webm"]:
            file_type = "recording"

        # Get file stats
        stats = file_path.stat()

        file_info = {
            "filename": file_path.name,
            "type": file_type,
            "size_bytes": stats.st_size,
            "created_at": datetime.fromtimestamp(stats.st_ctime).isoformat(),
            "url": f"/media/{task_id}/{file_path.name}",
        }
        media_info.append(file_info)

    # Filter by type if specified
    if type:
        media_info = [item for item in media_info if item["type"] == type]

    logger.info(f"Returning {len(media_info)} media items for task {task_id}")
    return {"media": media_info, "count": len(media_info)}


@app.get("/api/v1/media/{task_id}/{filename}")
async def get_media_file(
    task_id: str,
    filename: str,
    download: bool = Query(
        False, description="Force download instead of viewing in browser"
    ),
):
    """Serve a media file with options for viewing or downloading"""
    # Construct the file path
    file_path = MEDIA_DIR / task_id / filename

    # Check if file exists
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Media file not found")

    # Determine content type
    content_type, _ = mimetypes.guess_type(file_path)

    # Set headers based on download preference
    headers = {}
    if download:
        headers["Content-Disposition"] = f'attachment; filename="{filename}"'
    else:
        headers["Content-Disposition"] = f'inline; filename="{filename}"'

    # Return the file with appropriate headers
    return FileResponse(
        path=file_path, media_type=content_type, headers=headers, filename=filename
    )


@app.get("/api/v1/test-screenshot")
async def test_screenshot(ai_provider: str = "google"):
    """Test endpoint to verify screenshot functionality using refactored utility functions"""
    logger.info(f"Testing screenshot functionality with provider: {ai_provider}")

    browser = None
    try:
        # Use our get_llm utility (fallback for test providers)
        if ai_provider.lower() == "google":
            llm = ChatGoogle(model="gemini-1.5-flash")
        elif ai_provider.lower() == "openai":
            llm = ChatOpenAI(model="gpt-4o")
        else:
            # Use our standard get_llm function for consistency
            llm = get_llm(ai_provider)

        # Use our configure_browser_profile utility
        test_browser_config = {"headful": False}  # Force headless for testing
        browser, browser_info = configure_browser_profile(test_browser_config)
        logger.info(f"Test browser configuration: {browser_info}")

        # Use our create_agent_config utility
        task_instruction = "Navigate to example.com and take a screenshot"
        sensitive_data = get_sensitive_data()
        agent_config = create_agent_config(
            task_instruction, llm, sensitive_data, browser
        )

        agent = Agent(**agent_config)

        # Navigate to test page
        logger.info("Navigating to example.com for test")
        await agent.browser_session.navigate_to("https://example.com")

        # Test our capture_screenshot function
        test_task_id = "screenshot-test"
        logger.info("Testing screenshot capture with utility function")
        await capture_screenshot(agent, test_task_id, "test-user")

        # Check results using the same logic but simplified
        test_media_dir = MEDIA_DIR / test_task_id
        if test_media_dir.exists():
            screenshots = list(test_media_dir.glob("*.png"))
            if screenshots:
                latest_screenshot = max(screenshots, key=lambda x: x.stat().st_mtime)
                file_size = latest_screenshot.stat().st_size

                return {
                    "success": True,
                    "message": "Screenshot test completed using refactored utilities",
                    "file_size": file_size,
                    "file_path": str(latest_screenshot),
                    "url": f"/media/{test_task_id}/{latest_screenshot.name}",
                    "utilities_used": [
                        "configure_browser_profile",
                        "create_agent_config",
                        "capture_screenshot",
                    ],
                }
            else:
                return {"error": "No screenshots found after test"}
        else:
            return {"error": "Test media directory not created"}

    except Exception as e:
        logger.exception("Error in screenshot test")
        return {"error": f"Test failed: {str(e)}"}
    finally:
        # Use our cleanup utility approach
        await cleanup_task(browser, "screenshot-test", "test-user")


async def cleanup_all_tasks():
    """Clean up all running tasks on shutdown"""
    try:
        # Get all tasks from storage
        all_tasks = task_storage.list_tasks()
        if isinstance(all_tasks, dict) and "tasks" in all_tasks:
            tasks_list = all_tasks["tasks"]

            for task_summary in tasks_list:
                task_id = task_summary["id"]
                task = task_storage.get_task(task_id)

                if task and task.get("status") in [
                    TaskStatus.RUNNING,
                    TaskStatus.PAUSED,
                ]:
                    logger.info(f"Cleaning up running task: {task_id}")

                    # Get agent and try to stop it gracefully
                    agent = task_storage.get_task_agent(task_id)
                    if agent:
                        try:
                            agent.stop()
                        except Exception as e:
                            logger.warning(
                                f"Error stopping agent for task {task_id}: {e}"
                            )

                    # Update task status
                    task_storage.update_task_status(task_id, TaskStatus.STOPPED)
                    task_storage.mark_task_finished(task_id, status=TaskStatus.STOPPED)

        logger.info("Task cleanup completed")
    except Exception as e:
        logger.error(f"Error during task cleanup: {e}")


def setup_uvicorn_logging():
    """Configure uvicorn to suppress some of the noisy shutdown logs"""
    import logging

    # Reduce noise from uvicorn during shutdown
    uvicorn_logger = logging.getLogger("uvicorn.error")
    uvicorn_logger.setLevel(logging.WARNING)

    # Also suppress asyncio warnings during shutdown
    asyncio_logger = logging.getLogger("asyncio")
    asyncio_logger.setLevel(logging.WARNING)


async def run_server():
    """Run the server with proper asyncio signal handling"""
    port = int(os.environ.get("PORT", 8000))

    # Configure uvicorn server
    config = uvicorn.Config(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info",
        access_log=False,  # Reduce noise
        loop="asyncio",
    )
    server = uvicorn.Server(config)

    # Set up signal handlers for graceful shutdown
    def signal_handler():
        logger.info("\nReceived shutdown signal, initiating graceful shutdown...")
        server.should_exit = True

    # Register signal handlers (Unix-like systems only)
    if sys.platform != "win32":
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, signal_handler)

    # Start the server
    logger.info(f"Starting Browser Use Bridge API on port {port}")
    logger.info("Press Ctrl+C for graceful shutdown")

    try:
        await server.serve()
    except asyncio.CancelledError:
        logger.info("Server shutdown completed")
    except Exception as e:
        logger.error(f"Server error: {e}")
        raise


# Run server if executed directly
if __name__ == "__main__":
    setup_uvicorn_logging()

    try:
        # Use asyncio.run with proper exception handling
        asyncio.run(run_server())
    except KeyboardInterrupt:
        # This should rarely be reached due to signal handling above
        pass  # Silent shutdown - the signal handler already logged the message
    except Exception as e:
        logger.error(f"Error starting server: {e}")
        sys.exit(1)

    logger.info("Browser Use Bridge API stopped")
    sys.exit(0)
