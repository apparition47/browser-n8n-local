# Examples

These scripts demonstrate the new Web Eye + Memory + Reward flow.

## Prerequisites

1. Start the API server:

```bash
python app.py
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Configure your model provider in `.env`.

## Scripts

- `01_basic_flow.py`
  - Default task: go to `example.com` and report the page title.
  - Starts a task, waits for terminal status, then prints a summary JSON.
- `01_advanced_flow.py`
  - Default task: search for "huntrix golden lyrics" and extract YouTube video metadata.
  - Starts a task, polls until completion, then prints a summary JSON.

## Quick Start

```bash
python examples/01_basic_flow.py --base-url http://localhost:8000
python examples/01_advanced_flow.py --base-url http://localhost:8000
```

By default, scripts do not force any provider and will use the server's `DEFAULT_AI_PROVIDER`.
Use `--provider ollama` or `--provider deepseek` only when you want to override per run.

## Sample Results

### `01_basic_flow.py`

Default task argument:

```python
parser.add_argument(
    "--task",
    default="Go to example.com and report the page title",
    help="Task instruction",
)
```

Example output:

~~~bash
terminal_status=finished
{
  "id": "344a1b44-c747-4f56-b2a0-ef8382d58d19",
  "status": "finished",
  "observations": 0,
  "trajectory_events": 2,
  "reward": {
    "auto_score": 0.8,
    "manual_score": null,
    "effective_score": 0.8,
    "source": "auto",
    "reason": "task finished with non-empty output",
    "updated_at": "2026-06-22T07:49:52.889718+00:00Z"
  },
  "output": "<url>\nhttps://example.com/\n</url>\n<query>\npage title\n</query>\n<result>\nExample Domain\n</result>",
  "error": null
}
~~~

### `01_advanced_flow.py`

Default task argument:

```python
parser.add_argument(
    "--task",
    default=(
        "Go to google.com and search for 'huntrix golden lyrics'. "
        "On the search results page, click the first link that points to youtube.com/watch (or a YouTube video card) immediately; do not keep scrolling. "
        "If clicking a YouTube result fails after one attempt, navigate directly to https://www.youtube.com/results?search_query=huntrix+golden+lyrics and open the first video result. "
        "On the YouTube watch page, report the view count, like count, and upload/release date shown on the page."
        "Export the results as a JSON object with keys 'view_count', 'like_count', and 'upload_date'."
    ),
    help="Task instruction",
)
```

Example output:

~~~bash
{
  "id": "4be9cb39-e067-4d5f-b290-80ffb8358488",
  "status": "finished",
  "observations": 0,
  "trajectory_events": 2,
  "reward": {
    "auto_score": 0.8,
    "manual_score": null,
    "effective_score": 0.8,
    "source": "auto",
    "reason": "task finished with non-empty output",
    "updated_at": "2026-06-22T07:47:05.100077+00:00Z"
  },
  "output": "<url>\nhttps://www.youtube.com/watch?v=htk6MRjmcnQ\n</url>\n<query>\nPlease find the view count, the number of likes, and the upload/release date for the video shown on this page. Structure the output as a JSON object with keys 'view_count', 'like_count', and 'upload_date'.\n</query>\n<result>\n```json\n{\n  \"view_count\": \"164,669,057\",\n  \"like_count\": \"826K\",\n  \"upload_date\": \"Jul 1, 2025\"\n}\n```\n</result>",
  "error": null
}
~~~
