#!/usr/bin/env python3
"""Run a basic task and print summary fields including reward and observation counts."""

import argparse
import json
import sys
import time

import requests


TERMINAL_STATUSES = {"finished", "failed", "stopped"}


def run_task(base_url: str, task: str, provider: str | None, headful: bool) -> str:
    payload = {"task": task, "headful": headful}
    if provider:
        payload["ai_provider"] = provider

    response = requests.post(
        f"{base_url}/api/v1/run-task",
        json=payload,
        timeout=30,
    )
    response.raise_for_status()
    return response.json()["id"]


def wait_for_terminal_status(base_url: str, task_id: str, interval: float, timeout_s: int):
    started = time.time()

    while True:
        response = requests.get(f"{base_url}/api/v1/task/{task_id}/status", timeout=30)
        response.raise_for_status()
        data = response.json()
        status = data.get("status")
        print(f"status={status}")

        if status in TERMINAL_STATUSES:
            return data

        if time.time() - started > timeout_s:
            raise TimeoutError(f"Task {task_id} did not finish within {timeout_s}s")

        time.sleep(interval)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a basic Browser Use bridge flow")
    parser.add_argument("--base-url", default="http://localhost:8000", help="API base URL")
    parser.add_argument(
        "--task",
        default="Go to example.com and report the page title",
        help="Task instruction",
    )
    parser.add_argument(
        "--provider",
        default=None,
        help="LLM provider override (omit to use DEFAULT_AI_PROVIDER from server .env)",
    )
    parser.add_argument("--headful", action="store_true", help="Run browser in headful mode")
    parser.add_argument("--poll-interval", type=float, default=2.0, help="Status poll interval seconds")
    parser.add_argument("--timeout", type=int, default=300, help="Timeout in seconds")
    args = parser.parse_args()

    try:
        task_id = run_task(args.base_url, args.task, args.provider, args.headful)
        print(f"task_id={task_id}")

        terminal_status = wait_for_terminal_status(
            args.base_url,
            task_id,
            interval=args.poll_interval,
            timeout_s=args.timeout,
        )
        print(f"terminal_status={terminal_status.get('status')}")

        task_response = requests.get(f"{args.base_url}/api/v1/task/{task_id}", timeout=30)
        task_response.raise_for_status()
        task_data = task_response.json()

        summary = {
            "id": task_data.get("id"),
            "status": task_data.get("status"),
            "observations": len(task_data.get("observations", [])),
            "trajectory_events": len(task_data.get("trajectory", [])),
            "reward": task_data.get("reward", {}),
            "output": task_data.get("output"),
            "error": task_data.get("error"),
        }
        print(json.dumps(summary, indent=2))
        return 0
    except Exception as exc:
        print(f"error={exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
