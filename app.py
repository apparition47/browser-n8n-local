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

from typing import Literal, Optional
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
import jev_navigator


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
NAVIGATORS = ("browser-use", "jev")


def _navigator_setting(raw: Optional[str]) -> str:
    """DEFAULT_NAVIGATOR, browser-use unless set; an unknown value stops startup instead of failing every run."""
    value = (raw or "").strip().lower() or "browser-use"
    if value not in NAVIGATORS:
        raise SystemExit(f"DEFAULT_NAVIGATOR must be one of: {', '.join(NAVIGATORS)} (got {raw!r})")
    return value


DEFAULT_NAVIGATOR = _navigator_setting(os.environ.get("DEFAULT_NAVIGATOR"))
JEV_SCREENSHOTS = os.environ.get("JEV_SCREENSHOTS", "true").lower() == "true"
# Like jev-ultrafast, a Jev run returns its final state; an answer written by DEFAULT_AI_PROVIDER is opt-in.
JEV_EXTRACT = os.environ.get("JEV_EXTRACT", "false").lower() == "true"
JEV_PAGE_TEXT_MAX_CHARS = int(os.environ.get("JEV_PAGE_TEXT_MAX_CHARS", "20000"))


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
    navigator: Optional[Literal["browser-use", "jev"]] = None  # None means DEFAULT_NAVIGATOR
    extract: Optional[bool] = None  # Jev runs only; None means JEV_EXTRACT


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
        custom_headers = None
        custom_headers_raw = os.environ.get("OPENAI_CUSTOM_HEADERS")
        if custom_headers_raw:
            try:
                custom_headers = json.loads(custom_headers_raw)
            except json.JSONDecodeError as e:
                logger.error(f"Invalid OPENAI_CUSTOM_HEADERS JSON: {e}")

        kwargs = {"model": model}
        if base_url:
            kwargs["base_url"] = base_url
        if custom_headers:
            kwargs["default_headers"] = custom_headers

        # Some models reject non-default sampling params. Set to "none" to omit.
        for env_name, kwarg in (
            ("OPENAI_TEMPERATURE", "temperature"),
            ("OPENAI_FREQUENCY_PENALTY", "frequency_penalty"),
        ):
            raw = os.environ.get(env_name, "").strip().lower()
            if raw == "none":
                kwargs[kwarg] = None
            elif raw:
                kwargs[kwarg] = float(raw)

        return ChatOpenAI(**kwargs)


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
    downloads_dir: Optional[Path] = None,
    keep_alive: bool = False,
) -> tuple[Optional[Browser], dict]:
    """Configure browser based on task and environment settings.

    downloads_dir: when set, file downloads triggered during this browser's
    session are saved here instead of wherever Chrome's default download
    location is. Callers pass MEDIA_DIR / task_id so downloaded files show
    up in the run's attachments (GET /api/v4/runs/{run_id}/attachments),
    which glob that exact directory.
    """
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

        if downloads_dir:
            downloads_dir.mkdir(parents=True, exist_ok=True)
            browser_config_args["downloads_path"] = str(downloads_dir)
            logger.info(f"Using downloads directory: {downloads_dir}")

        if keep_alive:
            browser_config_args["keep_alive"] = True
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


JSON_OUTPUT_CONTRACT = (
    "Output contract:\n"
    "- Return the final answer as exactly one valid JSON object (no markdown, no code fences, no extra text).\n"
    "- Use dynamic keys that fit the task; do not rely on a fixed schema.\n"
    '- Include a short top-level "summary" string and put detailed values in other JSON fields.\n'
    '- If a requested value is unavailable, include the key with value null and explain briefly in "summary".\n'
)
# browser-use only: Jev runs answer with a single extraction call, so there's no done() to call.
BROWSER_USE_COMPLETION_RULE = (
    "\nCOMPLETION RULE (Critical):\n"
    "- Once the extract tool (or any tool) returns the requested data, IMMEDIATELY format as JSON and call done().\n"
    "- Do NOT attempt further browser navigation or tool calls after successful extraction.\n"
    "- Do NOT loop or retry; successful extraction = task complete."
)


def build_agent_task(instruction: str) -> str:
    """Optionally wrap task with constraints that reduce tool-chatter loops."""
    json_output_contract = JSON_OUTPUT_CONTRACT + BROWSER_USE_COMPLETION_RULE

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


async def _fetch_dataclub_login_code(max_wait: int = 90, max_age: int = 180) -> str:
    """Poll the mailbox (via the `gws` CLI) for the newest DataClub Tax verification email and return its code.

    Only mail from dataclubtax.ca is ever read. Accepts a message no older than `max_age` seconds so a
    stale code from an earlier attempt isn't reused; waits up to `max_wait` seconds for a fresh one.
    """
    import time as _time

    env = {**os.environ, "PATH": "/usr/local/bin:/usr/bin:/bin"}

    async def _gws(*args):
        proc = await asyncio.create_subprocess_exec(
            "/usr/local/bin/gws", "gmail", "users", "messages", *args,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, env=env,
        )
        out, err = await asyncio.wait_for(proc.communicate(), timeout=30)
        if proc.returncode != 0:
            raise RuntimeError(f"gws failed: {err.decode(errors='replace')[:200]}")
        return json.loads(out.decode())

    deadline = _time.time() + max_wait
    seen_stale = False
    while _time.time() < deadline:
        listing = await _gws("list", "--params", json.dumps(
            {"userId": "me", "q": "from:dataclubtax.ca newer_than:10m", "maxResults": 1}))
        msgs = listing.get("messages") or []
        if msgs:
            msg = await _gws("get", "--params", json.dumps(
                {"userId": "me", "id": msgs[0]["id"], "format": "metadata"}))
            age = _time.time() - int(msg.get("internalDate", 0)) / 1000
            m = re.search(r"verification code is:?\s*(\d{6})", msg.get("snippet", ""))
            if m and age <= max_age:
                return m.group(1)
            seen_stale = True
        await asyncio.sleep(5)
    raise RuntimeError(
        "No fresh DataClub verification email arrived in time"
        + (" (only an older code was found)" if seen_stale else "")
    )


def build_custom_tools():
    """Extra browser-use actions available to every agent run."""
    from browser_use import Tools, ActionResult

    tools = Tools()

    @tools.action(
        "Get the 6-digit DataClub Tax login verification code from the company mailbox. "
        "Call this ONLY after you have clicked 'Send code' and the code-entry screen is showing. "
        "It waits for the email and returns the code; then type that code into the field and submit. "
        "Never click 'Send code' or 'Resend' a second time to retry - every new request invalidates the previous code. "
        "If this action reports no email, call it again instead of requesting another code."
    )
    async def get_dataclub_login_code() -> ActionResult:
        try:
            code = await _fetch_dataclub_login_code()
        except Exception as e:  # surface to the agent so it can retry the action
            return ActionResult(extracted_content=f"Could not get the code: {e}")
        return ActionResult(extracted_content=f"The DataClub verification code is {code}")

    return tools


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

    try:
        agent_kwargs["tools"] = build_custom_tools()
    except Exception as e:  # never block a run because the extra tool failed to register
        logger.warning(f"Custom tools unavailable: {e}")

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


def _record_auto_reward(task_id: str, user_id: str):
    """Score a terminal task with the heuristic reward and log it on the trajectory."""
    task = task_storage.get_task(task_id, user_id)
    auto_score, auto_reason = compute_auto_reward(task)
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
    task_id: str,
    instruction: str,
    ai_provider: str,
    user_id: str = DEFAULT_USER_ID,
    session_id: Optional[str] = None,
    navigator: str = "browser-use",
):
    """Execute browser task in background - main orchestration function

    Chrome paths (CHROME_PATH and CHROME_USER_DATA) are only sourced from
    environment variables for security reasons.

    When session_id is set, the browser/agent are handed off to the
    session registry instead of being torn down, so a later queued
    message on the same session can continue in the same browser tab.
    """
    if navigator == "jev":
        await _execute_jev_task(task_id, instruction, ai_provider, user_id, session_id)
        return

    browser = None
    agent = None

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
        # Downloads are always saved under THIS task's media dir. When this
        # call starts a new session, that's the session's first task_id —
        # later continuations of the same session reuse this same browser
        # object (see _run_session_continuation) rather than recreating the
        # profile, so every download for the whole session lands here, not
        # under each individual continuation's own task_id.
        # Session-owned browsers must survive between runs (login, cookies, current page); without
        # keep_alive the agent resets the browser when each run ends, so a follow-up message in the
        # same session would find a blank tab.
        browser, browser_info = configure_browser_profile(
            task_browser_config, downloads_dir=MEDIA_DIR / task_id, keep_alive=bool(session_id)
        )
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
            _record_auto_reward(task_id, user_id)
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

        _record_auto_reward(task_id, user_id)
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

            _record_auto_reward(task_id, user_id)
            return

        logger.exception(f"Error executing task {task_id}")
        task_storage.update_task_status(task_id, TaskStatus.FAILED, user_id)
        task_storage.set_task_error(task_id, str(e), user_id)
        task_storage.mark_task_finished(task_id, user_id, TaskStatus.FAILED)
        _record_auto_reward(task_id, user_id)
    finally:
        if session_id:
            await _park_session_browser(session_id, agent, browser, user_id)
        else:
            await cleanup_task(browser, task_id, user_id)


def _fail_task(task_id: str, user_id: str, message: str):
    """Finish a task as failed with a reason, scored like any other terminal state."""
    task_storage.update_task_status(task_id, TaskStatus.FAILED, user_id)
    task_storage.set_task_error(task_id, message, user_id)
    task_storage.mark_task_finished(task_id, user_id, TaskStatus.FAILED)
    _record_auto_reward(task_id, user_id)


async def _save_jev_screenshot(browser: Browser, task_id: str, user_id: str):
    """Viewport PNG after a Jev action, taken between ticks so it can't race Jev's input."""
    try:
        png = await browser.take_screenshot()
    except Exception as screenshot_error:
        logger.warning(f"Task {task_id}: Jev screenshot failed: {screenshot_error}")
        return
    validate_and_save_screenshot(png, task_id, user_id, TaskStatus.RUNNING)


async def _run_jev(
    task_id: str,
    user_id: str,
    instruction: str,
    ai_provider: str,
    browser: Browser,
    controller: "jev_navigator.JevController",
    start_url: Optional[str],
):
    """Navigate with Jev on a started browser. On DONE the output is Jev's final state, or an
    LLM-written answer when the run opted in with extract."""
    if start_url:
        await browser.navigate_to(start_url)

    steps = []

    async def on_step(entry: dict):
        step = jev_navigator.step_record(entry, datetime.now(UTC).isoformat() + "Z")
        steps.append({key: step[key] for key in ("step", "operation", "action", "text", "url")})
        task_storage.add_task_step(task_id, step, user_id)
        add_trajectory_event(task_id, user_id, "jev_action", step)
        # The v4 API doesn't expose steps; this line is how to follow a Jev run live.
        logger.info(
            f"Task {task_id}: Jev step {step['step']} {step['operation']} {step['action']!r} -> {step['url']}"
        )
        if JEV_SCREENSHOTS:
            await _save_jev_screenshot(browser, task_id, user_id)

    outcome = await jev_navigator.navigate(
        browser,
        instruction,
        controller,
        timeout_seconds=TASK_RUN_TIMEOUT_SECONDS if TASK_RUN_TIMEOUT_SECONDS > 0 else None,
        on_step=on_step,
    )
    status, error = jev_navigator.terminal_status(outcome)
    if outcome.kind == "done":
        extract = (task_storage.get_task(task_id, user_id) or {}).get("extract", False)
        try:
            page = await jev_navigator.read_page(browser, outcome.target_id, JEV_PAGE_TEXT_MAX_CHARS)
            if extract:
                output = await jev_navigator.extract_output(
                    get_llm(ai_provider), instruction, page, JSON_OUTPUT_CONTRACT
                )
            else:
                output = jev_navigator.state_output(page, steps)
            task_storage.set_task_output(task_id, output, user_id)
        except Exception as extraction_error:
            logger.exception(f"Task {task_id}: Jev extraction failed")
            status, error = TaskStatus.FAILED, f"Extraction failed: {extraction_error}"
    if error:
        task_storage.set_task_error(task_id, error, user_id)
    task_storage.mark_task_finished(task_id, user_id, TaskStatus(status))
    await collect_browser_cookies(controller, task_id, user_id)
    _record_auto_reward(task_id, user_id)


async def _execute_jev_task(
    task_id: str,
    instruction: str,
    ai_provider: str,
    user_id: str,
    session_id: Optional[str] = None,
):
    """Jev run: fail fast before any browser launches, then navigate and extract on the bridge's own tab."""
    browser = None
    controller = None
    try:
        task_storage.update_task_status(task_id, TaskStatus.RUNNING, user_id)
        prepare_task_environment(task_id, user_id)
        try:
            jev_navigator.ensure_available()
        except jev_navigator.JevUnavailable as unavailable:
            logger.warning(f"Task {task_id}: Jev unavailable: {unavailable}")
            _fail_task(task_id, user_id, str(unavailable))
            return
        start_url = jev_navigator.find_start_url(instruction)
        if not start_url:
            _fail_task(
                task_id,
                user_id,
                "Jev needs a start URL: put a URL or domain (for example google.com) in the task text",
            )
            return

        task = task_storage.get_task(task_id, user_id) or {}
        downloads_dir = MEDIA_DIR / task_id
        # No keep_alive, even for sessions: it exists because the browser-use Agent resets its browser
        # when a run ends. Jev has no Agent, so its browser survives between session runs anyway, and
        # purge's close() still shuts it down.
        browser, browser_info = configure_browser_profile(
            task.get("browser_config", {}), downloads_dir=downloads_dir
        )
        if browser is None:
            # configure_browser_profile leaves headful runs without CHROME_PATH to browser-use's Agent; Jev has none.
            downloads_dir.mkdir(parents=True, exist_ok=True)
            browser = Browser(
                browser_profile=BrowserProfile(
                    headless=False,
                    viewport={"width": 1280, "height": 720},
                    window_size={"width": 1280, "height": 720},
                    downloads_path=str(downloads_dir),
                )
            )
        logger.info(f"Task {task_id}: Jev navigator, start_url={start_url}, browser={browser_info}")
        # Registered before the browser starts, so a cancel during launch stops the run before its first step.
        controller = jev_navigator.JevController(browser)
        task_storage.set_task_agent(task_id, controller, user_id)
        await browser.start()
        await _run_jev(task_id, user_id, instruction, ai_provider, browser, controller, start_url)
    except Exception as error:
        logger.exception(f"Error executing Jev task {task_id}")
        _fail_task(task_id, user_id, str(error))
    finally:
        if session_id:
            await _park_session_browser(session_id, controller, browser, user_id)
        else:
            await cleanup_task(browser, task_id, user_id)


# API Routes
def _new_task_record(
    task_id: str,
    task_text: str,
    ai_provider: Optional[str],
    user_id: str,
    save_browser_data: bool = False,
    headful: Optional[bool] = None,
    use_custom_chrome: Optional[bool] = None,
    session_id: Optional[str] = None,
    navigator: str = "browser-use",
    extract: bool = False,
) -> str:
    """Build and store a fresh task record. Returns its live_url."""
    now = datetime.now(UTC).isoformat() + "Z"
    live_url = f"/live/{task_id}"
    task_data = {
        "id": task_id,
        "task": task_text,
        "ai_provider": ai_provider,
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
        "save_browser_data": save_browser_data,
        "browser_data": None,  # Will store browser cookies if requested
        # Store browser configuration options
        "browser_config": {
            "headful": headful,
            "use_custom_chrome": use_custom_chrome,
        },
        "last_status_capture_epoch": None,
        "consecutive_duplicate_screenshots": 0,
        "screenshot_error_count": 0,
        "live_url": live_url,
        "session_id": session_id,
        "navigator": navigator,
        "extract": extract,
    }

    task_storage.create_task(task_id, task_data, user_id)
    return live_url


async def run_task(
    request: TaskRequest,
    user_id: str = Depends(get_user_id),
    session_id: Optional[str] = None,
):
    """Start a browser automation task"""
    task_id = str(uuid.uuid4())
    navigator = request.navigator or DEFAULT_NAVIGATOR
    extract = JEV_EXTRACT if request.extract is None else request.extract
    live_url = _new_task_record(
        task_id,
        request.task,
        request.ai_provider,
        user_id,
        save_browser_data=request.save_browser_data,
        headful=request.headful,
        use_custom_chrome=request.use_custom_chrome,
        session_id=session_id,
        navigator=navigator,
        extract=extract,
    )

    # Start task in background
    ai_provider = request.ai_provider or "openai"
    asyncio.create_task(
        execute_task(
            task_id, request.task, ai_provider, user_id, session_id=session_id, navigator=navigator
        )
    )

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


# --- Browser Use Cloud v4 API compatibility layer ---
# Lets the n8n-nodes-browser-use-cloud community node point at this local
# bridge instead of the real Browser Use Cloud.
#
# Run: create/get/get-many/cancel/status/events/attachments, mapped onto
# this bridge's existing task model.
#
# Session: the cloud keeps one browser open across queued follow-up
# messages. Here that's real too - the agent/browser created for a run
# started with a sessionId are kept alive (see _park_session_browser)
# instead of being closed. Later messages continue on the same tab: a
# browser-use follow-up drives the session's agent via Agent.add_new_task(),
# a Jev follow-up runs a fresh Jev goal. Each follow-up may pick its own
# navigator (e.g. log in with browser-use, then navigate with Jev).
#
# Browser: a standalone browser with no agent/task attached. Backed by a
# real browser_use Browser session with a periodic screenshot loop for a
# minimal live view, since nothing else drives it.

_CLOUD_RUN_TERMINAL_STATUS = {
    TaskStatus.FINISHED: "completed",
    TaskStatus.FAILED: "failed",
    TaskStatus.STOPPED: "cancelled",
}


def _to_cloud_run_status(local_status: str) -> str:
    return _CLOUD_RUN_TERMINAL_STATUS.get(local_status, local_status)


def _task_to_cloud_run(task: dict) -> dict:
    """Browser Use Cloud v4 RunSummary (the fields this bridge can fill)."""
    return {
        "id": task["id"],
        "task": task.get("task"),
        "status": _to_cloud_run_status(task["status"]),
        "model": os.environ.get("OPENAI_MODEL_ID") or task.get("ai_provider"),
        "result": task.get("output"),
        "output": task.get("output"),
        "error": task.get("error"),
        "sessionId": task.get("session_id"),
        "workspaceId": None,
        "createdAt": task.get("created_at"),
        "finishedAt": task.get("finished_at"),
    }


def _run_create_response(run_id: str, task_text: str, session_id: Optional[str] = None) -> dict:
    """Browser Use Cloud v4 RunCreateResponse."""
    return {
        "id": run_id,
        "status": "queued",
        "model": os.environ.get("OPENAI_MODEL_ID") or os.environ.get("DEFAULT_AI_PROVIDER", "openai"),
        "sessionId": session_id,
        "workspaceId": None,
        "eventsUrl": f"/api/v4/runs/{run_id}/events",
    }


@app.get("/api/v4/tasks")
async def cloud_tasks_probe():
    """Dummy endpoint the n8n credential's connection test hits."""
    return {"tasks": []}


@app.post("/api/v4/runs")
async def create_cloud_run(request: Request, user_id: str = Depends(get_user_id)):
    body = await request.json()
    task_text = body.get("task")
    if not task_text:
        raise HTTPException(status_code=400, detail='The "task" field is required.')

    # Bridge extensions, not in the cloud API: pick Jev or browser-use, and opt a Jev run into an LLM-written answer.
    navigator = body.get("navigator")
    if navigator is not None and navigator not in NAVIGATORS:
        raise HTTPException(
            status_code=400,
            detail=f'The "navigator" field must be one of: {", ".join(NAVIGATORS)}.',
        )
    extract = body.get("extract")
    if extract is not None and not isinstance(extract, bool):
        raise HTTPException(status_code=400, detail='The "extract" field must be true or false.')

    session_id = body.get("sessionId")
    if session_id:
        return await _create_or_continue_session_run(
            session_id, task_text, user_id, navigator=navigator, extract=extract
        )

    response = await run_task(TaskRequest(task=task_text, navigator=navigator, extract=extract), user_id)
    return _run_create_response(response.id, task_text)


@app.get("/api/v4/runs/{run_id}")
async def get_cloud_run(run_id: str, user_id: str = Depends(get_user_id)):
    task = task_storage.get_task(run_id, user_id)
    if not task:
        raise HTTPException(status_code=404, detail="Run not found")
    return _task_to_cloud_run(task)


@app.get("/api/v4/runs/{run_id}/status")
async def get_cloud_run_status(run_id: str, user_id: str = Depends(get_user_id)):
    task = task_storage.get_task(run_id, user_id)
    if not task:
        raise HTTPException(status_code=404, detail="Run not found")
    return {"id": run_id, "status": _to_cloud_run_status(task["status"])}


@app.post("/api/v4/runs/{run_id}/cancel")
async def cancel_cloud_run(run_id: str, user_id: str = Depends(get_user_id)):
    await stop_task(run_id, user_id)
    task = task_storage.get_task(run_id, user_id)
    return _task_to_cloud_run(task)


@app.get("/api/v4/runs")
async def list_cloud_runs(
    user_id: str = Depends(get_user_id),
    cursor: Optional[str] = None,
    limit: int = Query(50, ge=1, le=100),
):
    page = int(cursor) if cursor else 1
    result = task_storage.list_tasks(user_id, page, limit)
    runs = [_task_to_cloud_run(task) for task in result.get("tasks", [])]
    has_more = page * limit < result.get("total", 0)
    return {
        "runs": runs,
        "hasMore": has_more,
        "nextCursor": str(page + 1) if has_more else None,
    }


@app.get("/api/v4/runs/{run_id}/events")
async def get_cloud_run_events(run_id: str, user_id: str = Depends(get_user_id)):
    if not task_storage.task_exists(run_id, user_id):
        raise HTTPException(status_code=404, detail="Run not found")
    # No step-by-step event stream is recorded locally.
    return {"events": [], "hasMore": False, "nextAfter": None}


@app.get("/api/v4/runs/{run_id}/attachments")
async def get_cloud_run_attachments(run_id: str, user_id: str = Depends(get_user_id)):
    media = await list_task_media(run_id, user_id)
    attachments = [
        {"id": item["filename"], "filename": item["filename"], "url": f"/bridge{item['url']}"}
        for item in media.get("media", [])
    ]
    return {"attachments": attachments}


# --- Session resource: real cross-run continuity on one browser ---
# session_id -> {id, user_id, ai_provider, navigator (the default for follow-ups),
#                status, current_run_id, latest_run_id, first_run_id,
#                agent (whatever drove the last run: an Agent or a JevController),
#                browser_use_agent (kept for browser-use follow-ups), browser,
#                queue: [{id, text, run_id}], next_message_id, created_at}
_sessions: dict = {}


_SESSION_STATUS_TO_CLOUD = {"running": "running", "idle": "completed", "failed": "failed"}


def _session_to_cloud(session: dict) -> dict:
    """Browser Use Cloud v4 SessionInfo. `sessionId` is whatever string the caller chose (the real cloud wants a UUID)."""
    latest = session.get("current_run_id") or session.get("latest_run_id")
    task = task_storage.get_task(latest, session.get("user_id")) if latest else None
    return {
        "sessionId": session["id"],
        "workspaceId": None,
        "latestRunId": latest,
        "task": (task or {}).get("task"),
        "title": None,
        "status": _SESSION_STATUS_TO_CLOUD.get(session["status"], session["status"]),
        "createdAt": session["created_at"],
        "updatedAt": (task or {}).get("finished_at") or (task or {}).get("created_at") or session["created_at"],
    }


async def _park_session_browser(
    session_id: str, agent, browser: Optional[Browser], user_id: str
):
    """Called from execute_task's finally block for session-owned runs.

    Keeps the browser/agent alive in the session registry instead of
    closing them, then drains any messages queued while this run executed.
    """
    session = _sessions.get(session_id)
    if not session:
        # Session was purged while its first run was still executing.
        await cleanup_task(browser, f"session-{session_id}", user_id)
        return

    session["agent"] = agent
    if agent is not None and not isinstance(agent, jev_navigator.JevController):
        session["browser_use_agent"] = agent  # kept across Jev follow-ups for later browser-use ones
    session["browser"] = browser
    session["status"] = "idle"
    session["current_run_id"] = None
    asyncio.create_task(_drain_session_queue(session_id, user_id))


async def _run_session_continuation(
    session_id: str, task_id: str, message_text: str, user_id: str
):
    """Drive a follow-up message into a session's already-open agent/browser."""
    session = _sessions.get(session_id)
    task = task_storage.get_task(task_id, user_id) or {}
    if session and (task.get("navigator") or session.get("navigator")) == "jev":
        try:
            await _continue_jev_session(session, task_id, message_text, user_id)
        finally:
            session["status"] = "idle"
            session["current_run_id"] = None
            asyncio.create_task(_drain_session_queue(session_id, user_id))
        return

    agent = session.get("browser_use_agent") if session else None

    task_storage.update_task_status(task_id, TaskStatus.RUNNING, user_id)
    if agent is not None:
        session["agent"] = agent  # this run's driver, for purge and the /bridge/sessions endpoints
        task_storage.set_task_agent(task_id, agent, user_id)

    try:
        if agent is None and session and session.get("navigator") == "jev":
            raise RuntimeError(
                "This session started with Jev, so it has no browser-use agent; start the session with "
                '"navigator": "browser-use" to use browser-use follow-ups'
            )
        if agent is None:
            raise RuntimeError(
                "Session has no active browser/agent to continue (it may have failed or been purged)"
            )

        agent.add_new_task(message_text)

        async def _on_step_start(agent_instance):
            await automated_screenshot(agent_instance, task_id, user_id)

        run_kwargs = {
            "on_step_start": lambda agent_instance: asyncio.create_task(
                _on_step_start(agent_instance)
            ),
        }
        if AGENT_MAX_STEPS > 0:
            # browser-use counts steps cumulatively (`while n_steps <= max_steps`), so a follow-up in a
            # long session would otherwise return instantly with the previous answer once the agent has
            # used AGENT_MAX_STEPS in total. Give every follow-up its own budget.
            steps_so_far = int(getattr(getattr(agent, "state", None), "n_steps", 0) or 0)
            run_kwargs["max_steps"] = steps_so_far + AGENT_MAX_STEPS

        try:
            run_coro = agent.run(**run_kwargs)
        except TypeError:
            run_kwargs.pop("max_steps", None)
            run_coro = agent.run(**run_kwargs)

        if TASK_RUN_TIMEOUT_SECONDS > 0:
            result = await asyncio.wait_for(run_coro, timeout=TASK_RUN_TIMEOUT_SECONDS)
        else:
            result = await run_coro

        await process_task_result(result, task_id, user_id)
        task_storage.mark_task_finished(task_id, user_id, TaskStatus.FINISHED)
    except Exception as e:
        logger.exception(f"Error continuing session {session_id} run {task_id}")
        task_storage.update_task_status(task_id, TaskStatus.FAILED, user_id)
        task_storage.set_task_error(task_id, str(e), user_id)
        task_storage.mark_task_finished(task_id, user_id, TaskStatus.FAILED)
    finally:
        if session:
            session["status"] = "idle"
            session["current_run_id"] = None
            asyncio.create_task(_drain_session_queue(session_id, user_id))


async def _continue_jev_session(session: dict, task_id: str, message_text: str, user_id: str):
    """A Jev follow-up in a session: same tab, a fresh Jev goal, starting where the last run ended
    unless the message contains a full http(s) URL."""
    try:
        task_storage.update_task_status(task_id, TaskStatus.RUNNING, user_id)
        prepare_task_environment(task_id, user_id)
        # A headful browser-use first run without CHROME_PATH lets its Agent build the browser, so none is parked.
        browser = session.get("browser") or getattr(session.get("browser_use_agent"), "browser_session", None)
        if browser is None:
            _fail_task(
                task_id,
                user_id,
                "Session has no active browser to continue (it may have failed or been purged)",
            )
            return
        jev_navigator.ensure_available()
        controller = jev_navigator.JevController(browser)
        session["agent"] = controller
        task_storage.set_task_agent(task_id, controller, user_id)
        ai_provider = session.get("ai_provider") or os.environ.get("DEFAULT_AI_PROVIDER", "openai")
        await _run_jev(
            task_id,
            user_id,
            message_text,
            ai_provider,
            browser,
            controller,
            jev_navigator.find_start_url(message_text, bare_domains=False),
        )
    except Exception as error:
        logger.exception(f"Error continuing Jev session {session.get('id')} run {task_id}")
        _fail_task(task_id, user_id, str(error))


async def _drain_session_queue(session_id: str, user_id: str):
    """Pop and run the next queued message, if the session is idle and has one.

    A session runs one message at a time, matching the cloud API's own
    "a session runs one run at a time" semantics.
    """
    session = _sessions.get(session_id)
    if not session or session["status"] != "idle" or not session["queue"]:
        return

    message = session["queue"].pop(0)
    task_id = str(uuid.uuid4())
    _new_task_record(
        task_id,
        message["text"],
        session.get("ai_provider"),
        user_id,
        session_id=session_id,
        navigator=session["navigator"],
        extract=JEV_EXTRACT,
    )
    message["run_id"] = task_id
    session["status"] = "running"
    session["current_run_id"] = task_id
    session["latest_run_id"] = task_id
    await _run_session_continuation(session_id, task_id, message["text"], user_id)


async def _create_or_continue_session_run(
    session_id: str, task_text: str, user_id: str, navigator: Optional[str] = None, extract: Optional[bool] = None
):
    """POST /runs with a sessionId: start a brand-new session, or continue
    an idle one in its existing browser."""
    extract = JEV_EXTRACT if extract is None else extract  # per run, like a follow-up's navigator
    session = _sessions.get(session_id)

    if session and session["status"] == "running":
        raise HTTPException(
            status_code=409,
            detail="Session already has an active run. Wait for it to finish or cancel it first.",
        )

    if session is None:
        # Brand-new session: run normally (fresh browser); execute_task
        # will hand the browser off to the session registry when it's done.
        navigator = navigator or DEFAULT_NAVIGATOR
        task_id = str(uuid.uuid4())
        _sessions[session_id] = {
            "id": session_id,
            "user_id": user_id,
            "ai_provider": None,
            "navigator": navigator,
            "status": "running",
            "current_run_id": task_id,
            "latest_run_id": task_id,
            "first_run_id": task_id,
            "agent": None,
            "browser_use_agent": None,
            "browser": None,
            "queue": [],
            "next_message_id": 1,
            "created_at": datetime.now(UTC).isoformat() + "Z",
        }
        live_url = _new_task_record(
            task_id, task_text, None, user_id, session_id=session_id, navigator=navigator, extract=extract
        )
        ai_provider = os.environ.get("DEFAULT_AI_PROVIDER", "openai")
        _sessions[session_id]["ai_provider"] = ai_provider
        asyncio.create_task(
            execute_task(
                task_id, task_text, ai_provider, user_id, session_id=session_id, navigator=navigator
            )
        )
        return _run_create_response(task_id, task_text, session_id)

    # Idle session: continue in the same browser instead of starting a new one.
    # A follow-up may pick its own navigator (log in with browser-use, then navigate with Jev).
    task_id = str(uuid.uuid4())
    _new_task_record(
        task_id,
        task_text,
        session.get("ai_provider"),
        user_id,
        session_id=session_id,
        navigator=navigator or session["navigator"],
        extract=extract,
    )
    session["status"] = "running"
    session["current_run_id"] = task_id
    session["latest_run_id"] = task_id
    asyncio.create_task(_run_session_continuation(session_id, task_id, task_text, user_id))
    return _run_create_response(task_id, task_text, session_id)


@app.get("/api/v4/sessions/{session_id}")
async def get_cloud_session(session_id: str, user_id: str = Depends(get_user_id)):
    session = _sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return _session_to_cloud(session)


@app.get("/api/v4/sessions")
async def list_cloud_sessions(
    user_id: str = Depends(get_user_id),
    cursor: Optional[str] = None,
    limit: int = Query(50, ge=1, le=100),
):
    items = [s for s in _sessions.values() if s["user_id"] == user_id]
    page = int(cursor) if cursor else 1
    start = (page - 1) * limit
    end = start + limit
    page_items = items[start:end]
    has_more = end < len(items)
    return {
        "sessions": [_session_to_cloud(s) for s in page_items],
        "hasMore": has_more,
        "nextCursor": str(page + 1) if has_more else None,
    }


@app.post("/api/v4/sessions/{session_id}/queue")
async def queue_session_message(
    session_id: str, request: Request, user_id: str = Depends(get_user_id)
):
    session = _sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    body = await request.json()
    text = str(body.get("text") or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail='The "text" field is required.')

    message_id = session["next_message_id"]
    session["next_message_id"] += 1
    session["queue"].append({"id": message_id, "text": text, "run_id": None})

    if session["status"] == "idle":
        asyncio.create_task(_drain_session_queue(session_id, user_id))

    return {"id": message_id, "sessionId": session_id, "status": "queued"}


@app.get("/api/v4/sessions/{session_id}/queue")
async def get_session_queue(session_id: str, user_id: str = Depends(get_user_id)):
    session = _sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return {
        "queue": [
            {"id": m["id"], "text": m["text"], "runId": m["run_id"]}
            for m in session["queue"]
        ]
    }


@app.delete("/api/v4/sessions/{session_id}/queue/{message_id}")
async def cancel_queued_message(
    session_id: str, message_id: int, user_id: str = Depends(get_user_id)
):
    session = _sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    before = len(session["queue"])
    session["queue"] = [m for m in session["queue"] if m["id"] != message_id]
    if len(session["queue"]) == before:
        raise HTTPException(status_code=404, detail="Queued message not found")
    return {"success": True}


_FORM_VALUES_JS = """() => {
  const label = (el) => {
    if (el.getAttribute('aria-label')) return el.getAttribute('aria-label');
    if (el.id) { const l = document.querySelector('label[for="' + CSS.escape(el.id) + '"]'); if (l) return l.innerText.trim(); }
    const w = el.closest('label'); if (w) return w.innerText.trim();
    return '';
  };
  const visible = (el) => !!(el.offsetWidth || el.offsetHeight || el.getClientRects().length);
  const fields = [...document.querySelectorAll('input, select, textarea')]
    .filter((el) => el.type !== 'hidden' && visible(el))
    .map((el) => ({
      id: el.id || null, name: el.name || null, type: el.type || el.tagName.toLowerCase(),
      label: label(el).slice(0, 120),
      value: el.tagName === 'SELECT' ? ((el.selectedOptions[0] || {}).text || '') : el.value,
      checked: (el.type === 'checkbox' || el.type === 'radio') ? el.checked : null,
      group: el.type === 'radio' ? (((el.closest('.check-row') || {}).innerText || '').split('\\n')[0] || '').trim().slice(0, 160) : null,
    }));
  const h = document.querySelector('.step-heading') || document.querySelector('h1, h2');
  return JSON.stringify({ url: location.href, heading: h ? h.innerText.trim() : null, fields });
}"""


@app.get("/bridge/sessions/{session_id}/form-values")
async def session_form_values(session_id: str, user_id: str = Depends(get_user_id)):
    """Read the actual values of every visible form control on the session's current page.

    Lets callers VERIFY a fill from the DOM instead of trusting the agent's own report (which has
    claimed fields were filled when they were empty). Read-only; runs a fixed script.
    """
    session = _sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    if session.get("status") == "running":
        raise HTTPException(status_code=409, detail="Session has an active run")
    agent = session.get("agent")
    browser_session = getattr(agent, "browser_session", None) if agent else None
    if browser_session is None:
        raise HTTPException(status_code=409, detail="Session has no open browser")
    page = await browser_session.get_current_page()
    if page is None:
        raise HTTPException(status_code=409, detail="No open page")
    raw = await page.evaluate(_FORM_VALUES_JS)
    try:
        return json.loads(raw) if isinstance(raw, str) else raw
    except Exception:
        raise HTTPException(status_code=502, detail=f"Unexpected evaluate result: {str(raw)[:200]}")


_PAGE_OUTLINE_JS = """() => {
  const root = document.querySelector('.container') || document.body;
  const vis = (el) => !!(el.offsetWidth || el.offsetHeight || el.getClientRects().length);
  const text = (root.innerText || '').replace(/\\n{3,}/g, '\\n\\n').slice(0, 30000);
  const buttons = [...document.querySelectorAll('button, a.btn, [role=button]')].filter(vis)
    .map((b) => (b.innerText || b.getAttribute('aria-label') || '').trim()).filter(Boolean).slice(0, 80);
  const headings = [...document.querySelectorAll('h1, h2, h3, h4, .schedule-section-title')].filter(vis)
    .map((h) => h.innerText.trim()).filter(Boolean).slice(0, 80);
  const tables = [...document.querySelectorAll('table')].filter(vis).slice(0, 10)
    .map((t) => [...t.querySelectorAll('tr')].slice(0, 60).map((r) => [...r.children].map((c) => c.innerText.trim().slice(0, 80)).join(' | ')));
  return JSON.stringify({ url: location.href, headings, buttons, tables, text });
}"""


@app.get("/bridge/sessions/{session_id}/page-outline")
async def session_page_outline(session_id: str, user_id: str = Depends(get_user_id)):
    """Read-only structural dump of the session's current page (visible text, headings, buttons, tables)."""
    session = _sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    if session.get("status") == "running":
        raise HTTPException(status_code=409, detail="Session has an active run")
    agent = session.get("agent")
    browser_session = getattr(agent, "browser_session", None) if agent else None
    if browser_session is None:
        raise HTTPException(status_code=409, detail="Session has no open browser")
    page = await browser_session.get_current_page()
    if page is None:
        raise HTTPException(status_code=409, detail="No open page")
    raw = await page.evaluate(_PAGE_OUTLINE_JS)
    try:
        return json.loads(raw) if isinstance(raw, str) else raw
    except Exception:
        raise HTTPException(status_code=502, detail=f"Unexpected evaluate result: {str(raw)[:200]}")


@app.post("/api/v4/sessions/{session_id}/purge")
async def purge_cloud_session(session_id: str, user_id: str = Depends(get_user_id)):
    session = _sessions.pop(session_id, None)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    agent = session.get("agent")
    if agent:
        try:
            agent.stop()
        except Exception as e:
            logger.error(f"Error stopping agent for session {session_id}: {e}")

    await cleanup_task(session.get("browser"), f"session-{session_id}", user_id)
    return {"success": True, "sessionId": session_id}


# --- Browser resource: standalone browser, no agent/task attached ---
_browsers: dict = {}
BROWSER_SCREENSHOT_INTERVAL_SECONDS = 3


def _browser_to_cloud(entry: dict) -> dict:
    """Browser Use Cloud v4 BrowserSessionView (cost/proxy fields are always zero: nothing is metered locally)."""
    stopped = entry["status"] == "stopped"
    # Local Chrome's DevTools endpoint (127.0.0.1 only), so a client on this host can attach
    # with Playwright/Puppeteer: playwright.chromium.connect_over_cdp(cdpUrl).
    cdp_url = None if stopped else getattr(entry.get("browser"), "cdp_url", None)
    return {
        "id": entry["id"],
        "status": "stopped" if stopped else "active",
        "liveUrl": f"/live/browser/{entry['id']}",
        "cdpUrl": cdp_url,
        "timeoutAt": None,
        "startedAt": entry["created_at"],
        "finishedAt": entry.get("finished_at"),
        "proxyUsedMb": "0",
        "proxyCost": "0",
        "browserCost": "0",
        "agentSessionId": None,
        "recordingUrl": None,
        "metadata": {},
    }


async def _browser_screenshot_loop(browser_id: str, browser_session: Browser):
    media_dir = MEDIA_DIR / f"browser-{browser_id}"
    media_dir.mkdir(exist_ok=True, parents=True)
    while True:
        entry = _browsers.get(browser_id)
        if not entry or entry["status"] != "running":
            return
        try:
            png_bytes = await browser_session.take_screenshot()
            (media_dir / "latest.png").write_bytes(png_bytes)
        except Exception as e:
            logger.debug(f"Standalone browser {browser_id} screenshot failed: {e}")
        await asyncio.sleep(BROWSER_SCREENSHOT_INTERVAL_SECONDS)


@app.post("/api/v4/browsers")
async def create_cloud_browser(user_id: str = Depends(get_user_id)):
    browser_id = str(uuid.uuid4())
    downloads_dir = MEDIA_DIR / f"browser-{browser_id}" / "downloads"
    downloads_dir.mkdir(parents=True, exist_ok=True)
    headful = os.environ.get("BROWSER_USE_HEADFUL", "false").lower() == "true"
    # Same browser setup as ordinary runs (visible Chrome when configured), kept alive until stopped.
    browser_session, _info = configure_browser_profile({}, downloads_dir=downloads_dir, keep_alive=True)
    if browser_session is None:
        browser_session = Browser(
            browser_profile=BrowserProfile(
                headless=not headful, downloads_path=str(downloads_dir), keep_alive=True
            )
        )
    await browser_session.start()

    _browsers[browser_id] = {
        "id": browser_id,
        "browser": browser_session,
        "status": "running",
        "user_id": user_id,
        "created_at": datetime.now(UTC).isoformat() + "Z",
    }
    asyncio.create_task(_browser_screenshot_loop(browser_id, browser_session))
    return _browser_to_cloud(_browsers[browser_id])


@app.get("/api/v4/browsers/{browser_id}")
async def get_cloud_browser(browser_id: str, user_id: str = Depends(get_user_id)):
    entry = _browsers.get(browser_id)
    if not entry:
        raise HTTPException(status_code=404, detail="Browser session not found")
    return _browser_to_cloud(entry)


@app.get("/api/v4/browsers")
async def list_cloud_browsers(
    user_id: str = Depends(get_user_id),
    pageNumber: int = Query(1, ge=1),
    pageSize: int = Query(50, ge=1, le=100),
):
    items = [e for e in _browsers.values() if e["user_id"] == user_id]
    start = (pageNumber - 1) * pageSize
    end = start + pageSize
    return {
        "items": [_browser_to_cloud(e) for e in items[start:end]],
        "totalItems": len(items),
    }


@app.patch("/api/v4/browsers/{browser_id}")
async def stop_cloud_browser(browser_id: str, user_id: str = Depends(get_user_id)):
    entry = _browsers.get(browser_id)
    if not entry:
        raise HTTPException(status_code=404, detail="Browser session not found")

    entry["status"] = "stopped"
    entry["finished_at"] = datetime.now(UTC).isoformat() + "Z"
    try:
        await entry["browser"].stop()
    except Exception as e:
        logger.error(f"Error stopping standalone browser {browser_id}: {e}")
    return _browser_to_cloud(entry)


@app.get("/api/v4/browsers/{browser_id}/downloads")
async def get_cloud_browser_downloads(
    browser_id: str,
    user_id: str = Depends(get_user_id),
    cursor: Optional[str] = None,
    limit: int = Query(50, ge=1, le=100),
    includeUrls: bool = Query(False),
):
    """Matches the real Browser Use Cloud v4 "List Browser Session Downloads"
    contract (GET /api/v4/browsers/{session_id}/downloads), except `url` is a
    locally-served URL rather than a presigned S3 URL, since this bridge has
    no object storage."""
    entry = _browsers.get(browser_id)
    if not entry:
        raise HTTPException(status_code=404, detail="Browser session not found")

    downloads_dir = MEDIA_DIR / f"browser-{browser_id}" / "downloads"
    all_files = (
        sorted((p for p in downloads_dir.iterdir() if p.is_file()), key=lambda f: f.stat().st_mtime)
        if downloads_dir.exists()
        else []
    )

    start = int(cursor) if cursor else 0
    page = all_files[start : start + limit]
    has_more = start + limit < len(all_files)

    files = []
    for p in page:
        stat = p.stat()
        files.append(
            {
                "path": p.name,
                "size": stat.st_size,
                "lastModified": datetime.fromtimestamp(stat.st_mtime, UTC).isoformat().replace("+00:00", "Z"),
                "url": f"/bridge/browsers/{browser_id}/downloads/{p.name}" if includeUrls else None,
            }
        )

    return {
        "files": files,
        "hasMore": has_more,
        "nextCursor": str(start + limit) if has_more else None,
    }


@app.get("/bridge/browsers/{browser_id}/downloads/{filename}")
async def get_browser_download_file(
    browser_id: str, filename: str, user_id: str = Depends(get_user_id)
):
    """Serves a file previously downloaded during a standalone browser session."""
    entry = _browsers.get(browser_id)
    if not entry:
        raise HTTPException(status_code=404, detail="Browser session not found")

    file_path = MEDIA_DIR / f"browser-{browser_id}" / "downloads" / filename
    if not file_path.is_file():
        raise HTTPException(status_code=404, detail="Download not found")

    content_type, _ = mimetypes.guess_type(file_path)
    return FileResponse(file_path, media_type=content_type or "application/octet-stream", filename=filename)


@app.get("/live/browser/{browser_id}", response_class=HTMLResponse)
async def browser_live_view(browser_id: str, user_id: str = Depends(get_user_id)):
    """Minimal auto-refreshing screenshot view for a standalone browser session."""
    if browser_id not in _browsers:
        raise HTTPException(status_code=404, detail="Browser session not found")

    img_url = f"/bridge/media/browser-{browser_id}/latest.png"
    return f"""<!DOCTYPE html>
<html>
<head>
    <title>Standalone Browser {browser_id}</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
</head>
<body style="margin:0;background:#111;">
    <img id="shot" src="{img_url}" style="width:100%;height:auto;display:block;" />
    <script>
        setInterval(() => {{
            document.getElementById('shot').src = '{img_url}?t=' + Date.now();
        }}, {BROWSER_SCREENSHOT_INTERVAL_SECONDS * 1000});
    </script>
</body>
</html>"""


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
        <title>Browser Use Run {task_id}</title>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; }}
            .container {{ max-width: 1200px; margin: 0 auto; }}
            .status {{ padding: 10px; border-radius: 4px; margin-bottom: 20px; }}
            .queued {{ background-color: #f3e5f5; }}
            .running {{ background-color: #e3f2fd; }}
            .completed {{ background-color: #e8f5e9; }}
            .failed {{ background-color: #ffebee; }}
            .cancelled {{ background-color: #eeeeee; }}
            .controls {{ margin-bottom: 20px; }}
            button {{ padding: 8px 16px; margin-right: 10px; cursor: pointer; }}
            pre {{ background-color: #f5f5f5; padding: 15px; border-radius: 4px; overflow: auto; white-space: pre-wrap; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Browser Use Run</h1>
            <div id="status" class="status">Loading...</div>

            <div class="controls">
                <button id="cancelBtn">Cancel run</button>
            </div>

            <h2>Result</h2>
            <pre id="result">Loading...</pre>

            <script>
                const runId = '{task_id}';
                const TERMINAL = ['completed', 'failed', 'cancelled'];
                const userId = '{user_id}';

                // Set user ID in request headers if available
                const headers = {{}};
                if (userId && userId !== 'default') {{
                    headers['X-User-ID'] = userId;
                }}

                function updateStatus() {{
                    fetch(`/api/v4/runs/${{runId}}`, {{ headers }})
                        .then(response => response.json())
                        .then(data => {{
                            const statusEl = document.getElementById('status');
                            statusEl.textContent = `Status: ${{data.status}}`;
                            statusEl.className = `status ${{data.status}}`;

                            if (data.result) {{
                                document.getElementById('result').textContent = data.result;
                            }} else if (data.error) {{
                                document.getElementById('result').textContent = `Error: ${{data.error}}`;
                            }}

                            if (!TERMINAL.includes(data.status)) {{
                                setTimeout(updateStatus, 2000);
                            }}
                        }})
                        .catch(error => {{
                            console.error('Error fetching run:', error);
                            setTimeout(updateStatus, 5000);
                        }});
                }}

                document.getElementById('cancelBtn').addEventListener('click', () => {{
                    if (confirm('Cancel this run? This cannot be undone.')) {{
                        fetch(`/api/v4/runs/${{runId}}/cancel`, {{ method: 'POST', headers }})
                            .then(response => response.json())
                            .then(data => alert(`Run ${{data.status}}`))
                            .catch(error => console.error('Error cancelling run:', error));
                    }}
                }});

                updateStatus();
            </script>
        </div>
    </body>
    </html>
    """

    return HTMLResponse(content=html_content)


@app.get("/bridge/ping")
async def ping():
    """Health check endpoint"""
    return {"status": "success", "message": "API is running"}


@app.post("/bridge/pdf/layout-text")
async def pdf_layout_text(request: Request):
    """Extract layout-preserving text from a PDF sent as the raw request body.

    Used by the n8n T2 filing workflow: flattened/generated tax-return PDFs keep their
    filled-in values in the text layer, but n8n's built-in PDF extractor scrambles the reading
    order and drops them. `pdftotext -layout` keeps each value next to its label.
    Returns {"pageCount": n, "pages": [text, ...]} (one string per page).
    """
    body = await request.body()
    if not body.startswith(b"%PDF"):
        raise HTTPException(status_code=400, detail="Request body must be a PDF")
    proc = await asyncio.create_subprocess_exec(
        "/usr/local/bin/pdftotext", "-layout", "-", "-",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        out, err = await asyncio.wait_for(proc.communicate(body), timeout=60)
    except asyncio.TimeoutError:
        proc.kill()
        raise HTTPException(status_code=504, detail="pdftotext timed out")
    if proc.returncode != 0:
        raise HTTPException(status_code=422, detail=f"pdftotext failed: {err.decode(errors='replace')[:200]}")
    pages = out.decode("utf-8", errors="replace").split("\f")
    if pages and not pages[-1].strip():
        pages.pop()
    return {"pageCount": len(pages), "pages": pages}


@app.get("/bridge/browser-config")
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


@app.get("/bridge/media/{task_id}/{filename}")
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
