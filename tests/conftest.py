import os

# load env variables before imports
os.environ["WEBHOOK_SECRET"] = "test-secret"
os.environ["GITHUB_TOKEN"] = "test-token"
# Never let tests see the real Gemini key from .env: a test that forgets to patch
# the LLM then fails safely instead of spending quota and sending code to Google.
os.environ["GEMINI_API_KEY"] = "test-gemini-key"
os.environ["GEMINI_MODEL"] = "test-model"

import hashlib  # noqa: E402
import hmac  # noqa: E402
from pathlib import Path  # noqa: E402

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from reviewer.ingest import webhook  # noqa: E402

FIXTURES_DIR = Path(__file__).parent / "fixtures"

@pytest.fixture
def webhook_secret() -> bytes:
    """The TEST webhook secret as bytes, the same value the app loaded at import.

    Set at the top of this file, before the app is imported, because
    `settings = Settings()` reads the environment only once.
    """
    return os.environ["WEBHOOK_SECRET"].encode()

@pytest.fixture
def client() -> TestClient:
    """An in-process HTTP client for the FastAPI app (no real server or network).

    TestClient runs background tasks before `post()` returns, so tests can check
    what the task did right after the call.
    """
    return TestClient(webhook.app)

# runs this fixture before every test automatically without having to pass it as an argument
@pytest.fixture(autouse=True)
def _clear_deliveries():
    """Reset the module-level dedupe set around every test (autouse).

    `PROCESSED_DELIVERIES` lives on the module and survives between tests.
    Without this reset, a delivery ID used in one test would make a later test
    see "Duplicate delivery ignored", and tests would depend on their run order.
    """
    webhook.PROCESSED_DELIVERIES.clear() # runs during set (before the test)
    yield
    webhook.PROCESSED_DELIVERIES.clear() # runs after the test 

@pytest.fixture
def load_fixture():
    """Factory fixture: returns `_load(name) -> bytes` for files in tests/fixtures/.

    `_load("ping")` and `_load("ping.json")` both work. It returns raw bytes,
    because the signature covers exact bytes; `json.loads` them yourself when you
    need values or a modified variant. The path is resolved from this file, so
    pytest works from any directory.
    """
    def _load(name: str) -> bytes:
        # appends .json to file name if it doesn't already have it 
        filename = name if name.endswith(".json") else f"{name}.json"
        return (FIXTURES_DIR / filename).read_bytes()
    return _load

@pytest.fixture
def sign(webhook_secret):
    """Factory fixture: returns `_sign(body, secret=None) -> "sha256=<hex>"`.

    It computes the `X-Hub-Signature-256` value the way GitHub does: HMAC-SHA256
    of the raw body, as lowercase hex. By default it uses the test secret, so the
    app accepts the result. Pass `secret=b"..."` to forge one the app must reject.
    This is how saved fixtures get valid signatures without the real secret.
    """
    def _sign(body: bytes, secret: bytes | None = None) -> str:
        _secret = webhook_secret if secret is None else secret  # bytes either way
        digest = hmac.new(_secret, body, hashlib.sha256).hexdigest()
        return f"sha256={digest}"
    return _sign

@pytest.fixture
def post_webhook(client, sign):
    """Factory fixture: returns `_post(event, body, ...)`, which POSTs to /webhook/github like GitHub.

    It sets `X-GitHub-Event`, `X-GitHub-Delivery` and (unless told not to)
    `X-Hub-Signature-256`, and sends `body` byte-for-byte (`content=`, not
    `json=`), so the bytes signed are the bytes sent.

        _post(event, body)                          signed with the test secret
        _post(event, body, signed=False)            no signature header
        _post(event, body, secret=b"wrong")         signed with another secret
        _post(event, body, signature="sha256=...")  this exact header (tampering tests)
        _post(event, body, delivery="d-1")          choose the delivery ID (dedupe tests)

    Returns the client's response (`.status_code`, `.json()`).
    """
    def _post(event: str, body: bytes, delivery: str = "test-delivery", signed: bool = True, secret: bytes | None = None, signature: str | None = None,):
        headers = {
            "X-GitHub-Event": event,
            "X-GitHub-Delivery": delivery,
        }
        if signature is not None:
            headers["X-Hub-Signature-256"] = signature
        elif signed:
            headers["X-Hub-Signature-256"] = sign(body, secret=secret)

        return client.post("/webhook/github", content=body, headers=headers)
    return _post

@pytest.fixture
def queued(monkeypatch) -> list:
    """Replace the background task with a recorder; returns the list of recorded calls.

    It patches `webhook.process_pull_request`, which works because the endpoint
    looks the name up on the module each time it runs (patch where it's looked
    up). Each call's args are appended as a tuple, so tests assert `queued == []`
    (nothing queued) or `queued == [(delivery, repo, number, sha)]`. monkeypatch
    undoes the patch after each test. Don't use it in dedupe tests, which need
    the real task.
    """
    calls = []
    def fake(*args):
        calls.append(args)

    monkeypatch.setattr(webhook, "process_pull_request", fake) # module level patching
    return calls


class FakeGitHubClient:
    def __init__(self) -> None:
        self.diff_to_return = "diff --git a/a.py b/a.py\n+new line\n"
        self.error_to_raise: Exception | None = None
        self.comments_posted: list[tuple[str, int, str]] = []

    def __call__(self, token: str, **kwargs) -> "FakeGitHubClient":
        # Allows FakeGitHubClient instance to act as the GitHubClient(...) constructor
        return self

    async def __aenter__(self) -> "FakeGitHubClient":
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        pass

    async def get_pr_diff(self, repo: str, number: int) -> str:
        if self.error_to_raise is not None:
            raise self.error_to_raise
        return self.diff_to_return

    async def post_comment(self, repo: str, number: int, body: str) -> str:
        self.comments_posted.append((repo, number, body))
        return f"https://github.com/{repo}/pull/{number}#issuecomment-1"


@pytest.fixture
def fake_github(monkeypatch) -> FakeGitHubClient:
    fake = FakeGitHubClient()
    monkeypatch.setattr(webhook, "GitHubClient", fake)
    return fake


class FakeReview:
    """Stand-in for `review_diff`: records each diff it gets, returns a canned review.

    Set `error_to_raise` to make it fail like the real one would (e.g. a
    `ReviewError` for a Gemini rate limit).
    """

    def __init__(self) -> None:
        self.review_to_return = "FAKE REVIEW"
        self.error_to_raise: Exception | None = None
        self.diffs_reviewed: list[str] = []

    async def __call__(self, diff: str) -> str:
        # async, because the task does `await review_diff(diff=...)`
        self.diffs_reviewed.append(diff)
        if self.error_to_raise is not None:
            raise self.error_to_raise
        return self.review_to_return


@pytest.fixture
def fake_review(monkeypatch) -> FakeReview:
    """Replace `webhook.review_diff` with a FakeReview, so no test calls Gemini.

    Patched on `webhook` because that's where the task looks the name up
    (`from reviewer.agent.nodes import review_diff` copied it there).
    """
    fake = FakeReview()
    monkeypatch.setattr(webhook, "review_diff", fake)
    return fake
