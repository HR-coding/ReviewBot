import json
import logging
import random
from typing import Any

import httpx
import pytest

from reviewer.ingest import github_client
from reviewer.ingest.github_client import COMMENT_MARKER, GitHubAPIError, GitHubClient

TOKEN = "ghp_TEST_SECRET_123"
NOW = 1_700_000_000.0  # the fake wall clock (Unix time) every client in this file sees
MAX_ATTEMPTS = 4  # GitHubClient's default


class RecordingSleep:
    """Stand-in for `asyncio.sleep`: records each delay and returns at once.

    Retries then cost no real time, and tests can assert exactly how long the
    client *would* have waited.
    """

    def __init__(self) -> None:
        self.delays: list[float] = []

    async def __call__(self, delay: float) -> None:
        self.delays.append(delay)


@pytest.fixture
def sleeps() -> RecordingSleep:
    return RecordingSleep()


@pytest.fixture
def make_client(sleeps):
    """Factory: `make_client(handler, **overrides)` → a GitHubClient on a MockTransport.

    Time is fake (no real sleeps, the clock is fixed at NOW) and the jitter is
    seeded, so every run is identical and instant.
    """
    def _factory(handler, **overrides):
        options: dict[str, Any] = dict(sleep=sleeps, clock=lambda: NOW, rng=random.Random(0))
        options.update(overrides)
        return GitHubClient(TOKEN, transport=httpx.MockTransport(handler), **options)
    return _factory


def scripted(*steps):
    """A handler that answers each request with the next step, in order.

    A step is an `httpx.Response` to return or an exception to raise. Returns
    `(handler, requests)`; `requests` records every attempt the client made.
    """
    queue = list(steps)
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        step = queue.pop(0)
        if isinstance(step, Exception):
            raise step
        return step

    return handler, requests


@pytest.mark.anyio
async def test_get_pr_diff_request_shape(make_client):  # G1
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, text="diff --git a/x b/x\n+hello")

    async with make_client(handler) as gh:
        result = await gh.get_pr_diff("o/r", 1)

    assert result == "diff --git a/x b/x\n+hello"
    req = captured[0]
    assert req.method == "GET"
    assert req.url.path == "/repos/o/r/pulls/1"
    assert req.headers["Accept"] == "application/vnd.github.diff"
    assert req.headers["Authorization"] == f"Bearer {TOKEN}"
    assert req.headers["X-GitHub-Api-Version"] == "2022-11-28"


@pytest.mark.anyio
async def test_post_comment_request_shape(make_client):  # G2
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(201, json={"html_url": "https://github.com"})

    async with make_client(handler) as gh:
        url = await gh.post_comment("o/r", 1, "hi")

    assert url == "https://github.com"
    req = captured[0]
    assert req.method == "POST"
    assert req.url.path == "/repos/o/r/issues/1/comments"
    assert json.loads(req.content) == {"body": "hi"}


@pytest.mark.anyio
@pytest.mark.parametrize("status", [401, 404, 422, 500])
async def test_non_2xx_raises_github_api_error(make_client, sleeps, status):  # G3
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(status, text="boom")

    async with make_client(handler) as gh:
        with pytest.raises(GitHubAPIError) as exc_info:
            await gh.get_pr_diff("o/r", 1)

    assert exc_info.value.status_code == status
    assert "boom" in exc_info.value.message
    # 500 may be temporary → retried until the attempts run out; the rest are permanent.
    expected_attempts = MAX_ATTEMPTS if status == 500 else 1
    assert len(requests) == expected_attempts
    assert len(sleeps.delays) == expected_attempts - 1


@pytest.mark.anyio
async def test_network_error_raises_chained_github_api_error(make_client):  # G4
    original_err = httpx.ConnectError("down")
    handler, requests = scripted(*[original_err] * MAX_ATTEMPTS)

    async with make_client(handler) as gh:
        with pytest.raises(GitHubAPIError) as exc_info:
            await gh.get_pr_diff("o/r", 1)

    assert exc_info.value.status_code is None
    assert exc_info.value.__cause__ is original_err
    assert len(requests) == MAX_ATTEMPTS  # never sent → retried every time


@pytest.mark.anyio
async def test_token_never_leaks_in_logs(make_client, caplog):  # G5
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, text="Bad credentials")

    async with make_client(handler) as gh:
        with pytest.raises(GitHubAPIError):
            await gh.get_pr_diff("o/r", 1)

    assert TOKEN not in caplog.text


@pytest.mark.anyio
class TestRetries:
    """GET (fetching the diff) is safe to repeat, so temporary failures are retried."""

    async def test_temporary_errors_are_retried_until_success(self, make_client, sleeps):
        # G6
        handler, requests = scripted(
            httpx.Response(502), httpx.Response(502), httpx.Response(200, text="the diff")
        )

        async with make_client(handler) as gh:
            assert await gh.get_pr_diff("o/r", 1) == "the diff"

        assert len(requests) == 3
        assert len(sleeps.delays) == 2

    async def test_gives_up_after_max_attempts(self, make_client, sleeps):
        # G7
        handler, requests = scripted(*[httpx.Response(500, text="down")] * MAX_ATTEMPTS)

        async with make_client(handler) as gh:
            with pytest.raises(GitHubAPIError) as exc_info:
                await gh.get_pr_diff("o/r", 1)

        assert exc_info.value.status_code == 500
        assert len(requests) == MAX_ATTEMPTS
        assert len(sleeps.delays) == MAX_ATTEMPTS - 1  # no pointless sleep after the last try

    @pytest.mark.parametrize(
        "status, text",
        [
            (401, "Bad credentials"),
            (404, "Not Found"),
            (422, "Validation Failed"),
            # A 403 with no rate-limit signal = missing permission: retrying can't fix it.
            (403, "Resource not accessible by personal access token"),
        ],
    )
    async def test_permanent_errors_are_not_retried(self, make_client, sleeps, status, text):
        # G8
        handler, requests = scripted(httpx.Response(status, text=text))

        async with make_client(handler) as gh:
            with pytest.raises(GitHubAPIError) as exc_info:
                await gh.get_pr_diff("o/r", 1)

        assert exc_info.value.status_code == status
        assert len(requests) == 1
        assert sleeps.delays == []

    async def test_read_timeout_on_get_is_retried(self, make_client):
        # G15: for a GET, an unknown outcome is harmless: reading twice changes nothing.
        handler, requests = scripted(httpx.ReadTimeout("slow"), httpx.Response(200, text="the diff"))

        async with make_client(handler) as gh:
            assert await gh.get_pr_diff("o/r", 1) == "the diff"

        assert len(requests) == 2


@pytest.mark.anyio
class TestRateLimits:
    """GitHub says how long to wait (headers); the client waits exactly that, within limits."""

    async def test_retry_after_header_is_obeyed(self, make_client, sleeps):
        # G9: the "simulated 429 is retried" from Phase 1's done-when.
        handler, requests = scripted(
            httpx.Response(429, headers={"retry-after": "7"}), httpx.Response(200, text="the diff")
        )

        async with make_client(handler) as gh:
            assert await gh.get_pr_diff("o/r", 1) == "the diff"

        assert sleeps.delays == [7.0]
        assert len(requests) == 2

    async def test_rate_limited_post_is_retried(self, make_client):
        # A rate limit means GitHub refused the request BEFORE doing anything,
        # so even a POST is safe to send again.
        handler, requests = scripted(
            httpx.Response(429, headers={"retry-after": "1"}),
            httpx.Response(201, json={"html_url": "https://github.com/c/1"}),
        )

        async with make_client(handler) as gh:
            assert await gh.post_comment("o/r", 1, "hi") == "https://github.com/c/1"

        assert len(requests) == 2

    async def test_waits_until_rate_limit_reset(self, make_client, sleeps):
        # G10: x-ratelimit-reset is a Unix timestamp → wait = reset - now (the fake clock).
        handler, _ = scripted(
            httpx.Response(403, headers={"x-ratelimit-remaining": "0", "x-ratelimit-reset": str(int(NOW) + 30)}),
            httpx.Response(200, text="the diff"),
        )

        async with make_client(handler) as gh:
            await gh.get_pr_diff("o/r", 1)

        assert sleeps.delays == [30.0]

    async def test_reset_in_the_past_waits_zero(self, make_client, sleeps):
        # Clock skew: GitHub's reset can be "before" our now. Never sleep a negative time.
        handler, _ = scripted(
            httpx.Response(403, headers={"x-ratelimit-remaining": "0", "x-ratelimit-reset": str(int(NOW) - 5)}),
            httpx.Response(200, text="the diff"),
        )

        async with make_client(handler) as gh:
            await gh.get_pr_diff("o/r", 1)

        assert sleeps.delays == [0.0]

    async def test_reset_too_far_away_fails_fast(self, make_client, sleeps):
        # G11: don't park a background task for an hour; fail, and let a redelivery retry.
        handler, requests = scripted(
            httpx.Response(403, headers={"x-ratelimit-remaining": "0", "x-ratelimit-reset": str(int(NOW) + 3600)}),
        )

        async with make_client(handler) as gh:
            with pytest.raises(GitHubAPIError) as exc_info:
                await gh.get_pr_diff("o/r", 1)

        assert exc_info.value.status_code == 403
        assert len(requests) == 1
        assert sleeps.delays == []

    async def test_secondary_rate_limit_waits_at_least_a_minute(self, make_client, sleeps):
        # No headers, only the message: GitHub's docs say wait at least one minute.
        handler, _ = scripted(
            httpx.Response(403, json={"message": "You have exceeded a secondary rate limit."}),
            httpx.Response(200, text="the diff"),
        )

        async with make_client(handler) as gh:
            await gh.get_pr_diff("o/r", 1)

        assert sleeps.delays == [60.0]

    async def test_unparseable_retry_after_falls_back_to_backoff(self, make_client, sleeps):
        # Retry-After can be an HTTP date; we don't parse those, so use normal backoff.
        handler, _ = scripted(
            httpx.Response(429, headers={"retry-after": "Wed, 21 Oct 2026 07:28:00 GMT"}),
            httpx.Response(200, text="the diff"),
        )

        async with make_client(handler) as gh:
            await gh.get_pr_diff("o/r", 1)

        assert len(sleeps.delays) == 1
        assert 0.0 <= sleeps.delays[0] <= 1.0  # attempt 1's backoff ceiling = base_delay


@pytest.mark.anyio
class TestPostSafety:
    """A POST (creating a comment) is retried ONLY when it certainly had no effect.

    Otherwise a retry could post the same review twice (Phase 1 Q1).
    """

    async def test_read_timeout_on_post_is_not_retried(self, make_client, sleeps):
        # G12: GitHub may have saved the comment and only the reply was lost.
        timeout = httpx.ReadTimeout("slow")
        handler, requests = scripted(timeout, httpx.Response(201, json={"html_url": "x"}))

        async with make_client(handler) as gh:
            with pytest.raises(GitHubAPIError) as exc_info:
                await gh.post_comment("o/r", 1, "hi")

        assert len(requests) == 1
        assert sleeps.delays == []
        assert exc_info.value.__cause__ is timeout

    @pytest.mark.parametrize(
        "error",
        [httpx.ConnectError("refused"), httpx.ConnectTimeout("no route"), httpx.PoolTimeout("pool full")],
        ids=["connect-error", "connect-timeout", "pool-timeout"],
    )
    async def test_never_sent_post_is_retried(self, make_client, error):
        # G13: the request never left this machine, so sending it again can't duplicate it.
        handler, requests = scripted(error, httpx.Response(201, json={"html_url": "https://github.com/c/1"}))

        async with make_client(handler) as gh:
            assert await gh.post_comment("o/r", 1, "hi") == "https://github.com/c/1"

        assert len(requests) == 2

    async def test_server_error_on_post_is_not_retried(self, make_client):
        # G14: a 502 can come back AFTER GitHub processed the request.
        handler, requests = scripted(httpx.Response(502), httpx.Response(201, json={"html_url": "x"}))

        async with make_client(handler) as gh:
            with pytest.raises(GitHubAPIError) as exc_info:
                await gh.post_comment("o/r", 1, "hi")

        assert exc_info.value.status_code == 502
        assert len(requests) == 1


class TestBackoff:
    def test_full_jitter_stays_within_the_exponential_ceiling(self):
        # G16
        gh = GitHubClient(TOKEN, base_delay=1.0, max_delay=30.0, rng=random.Random(42))

        for attempt in range(1, 11):
            ceiling = min(30.0, 1.0 * 2 ** (attempt - 1))
            samples = [gh._backoff(attempt) for _ in range(50)]
            assert all(0.0 <= d <= ceiling for d in samples), attempt
            # Not vacuous: the delays really are random, and really use the range.
            assert len(set(samples)) > 1
            assert max(samples) > ceiling / 2

    def test_max_attempts_below_one_is_rejected(self):
        with pytest.raises(ValueError, match="max_attempts"):
            GitHubClient(TOKEN, max_attempts=0)


@pytest.mark.anyio
async def test_each_retry_is_logged_without_the_token(make_client, caplog):
    # G17
    caplog.set_level(logging.WARNING, logger="reviewer.ingest.github_client")
    handler, _ = scripted(httpx.Response(502), httpx.Response(502), httpx.Response(200, text="d"))

    async with make_client(handler) as gh:
        await gh.get_pr_diff("o/r", 1)

    assert caplog.text.count("retrying in") == 2
    assert TOKEN not in caplog.text


# ---------------------------------------------------------------- upsert_comment

BOT = "review-bot"  # the login the token acts as (GET /user)


def comment(comment_id: int, login: str, body: str) -> dict:
    """A comment in GitHub's REST shape (only the fields the client reads)."""
    return {"id": comment_id, "user": {"login": login}, "body": body}


class FakeGitHubAPI:
    """A tiny stateful GitHub: /user, list (paginated) / create / update issue comments.

    Use it as the MockTransport handler. `calls` records (method, path) for each
    request. Set `lose_next_post_reply = True` to make the next POST *succeed*
    (the comment is stored) but raise ReadTimeout, the Q1 scenario.
    """

    def __init__(self, comments: list[dict] | None = None, *, per_page: int = 100) -> None:
        self.comments = list(comments or [])
        self.per_page = per_page
        self.calls: list[tuple[str, str]] = []
        self.patched: dict[int, str] = {}
        self.lose_next_post_reply = False
        self._next_id = 1000

    def __call__(self, request: httpx.Request) -> httpx.Response:
        method, path = request.method, request.url.path
        self.calls.append((method, path))

        if method == "GET" and path == "/user":
            return httpx.Response(200, json={"login": BOT})

        if method == "GET" and path.endswith("/comments"):
            page = int(request.url.params.get("page", "1"))
            start = (page - 1) * self.per_page
            headers = {}
            if start + self.per_page < len(self.comments):
                headers["link"] = f'<https://api.github.com{path}?per_page={self.per_page}&page={page + 1}>; rel="next"'
            return httpx.Response(200, json=self.comments[start:start + self.per_page], headers=headers)

        if method == "POST" and path.endswith("/comments"):
            body = json.loads(request.content)["body"]
            self.comments.append(comment(self._next_id, BOT, body))
            self._next_id += 1
            if self.lose_next_post_reply:
                self.lose_next_post_reply = False
                raise httpx.ReadTimeout("saved, but the reply was lost")
            return httpx.Response(201, json={"html_url": f"https://github.com/c/{self._next_id - 1}"})

        if method == "PATCH" and "/issues/comments/" in path:
            comment_id = int(path.rsplit("/", 1)[1])
            self.patched[comment_id] = json.loads(request.content)["body"]
            return httpx.Response(200, json={"html_url": f"https://github.com/c/{comment_id}"})

        return httpx.Response(404, text=f"unexpected {method} {path}")

    def methods(self) -> list[str]:
        return [method for method, _ in self.calls]


@pytest.mark.anyio
class TestUpsertComment:
    """One review comment per PR: found by marker AND author, updated (PATCH) or created (POST)."""

    async def test_creates_comment_with_marker_when_none_exists(self, make_client):
        # U1
        api = FakeGitHubAPI()

        async with make_client(api) as gh:
            url = await gh.upsert_comment("o/r", 1, "the review")

        assert api.methods() == ["GET", "GET", "POST"]  # /user, list, create
        assert url == "https://github.com/c/1000"
        assert api.comments[0]["body"] == f"{COMMENT_MARKER}\nthe review"

    async def test_updates_own_comment_instead_of_posting(self, make_client):
        # U2: the second push of a PR edits the existing review.
        api = FakeGitHubAPI([comment(42, BOT, f"{COMMENT_MARKER}\nold review")])

        async with make_client(api) as gh:
            url = await gh.upsert_comment("o/r", 1, "new review")

        assert ("PATCH", "/repos/o/r/issues/comments/42") in api.calls
        assert "POST" not in api.methods()
        assert api.patched[42] == f"{COMMENT_MARKER}\nnew review"
        assert url == "https://github.com/c/42"

    async def test_marker_in_someone_elses_comment_is_ignored(self, make_client):
        # U3: anyone can type the invisible marker; we must not edit their comment.
        api = FakeGitHubAPI([comment(7, "stranger", f"{COMMENT_MARKER}\nlol")])

        async with make_client(api) as gh:
            await gh.upsert_comment("o/r", 1, "the review")

        assert "PATCH" not in api.methods()
        assert api.methods()[-1] == "POST"

    async def test_own_comment_without_marker_is_left_alone(self, make_client):
        # U4: with a PAT the bot IS you, so your normal comments share its login.
        api = FakeGitHubAPI([comment(7, BOT, "a normal human comment")])

        async with make_client(api) as gh:
            await gh.upsert_comment("o/r", 1, "the review")

        assert "PATCH" not in api.methods()
        assert api.methods()[-1] == "POST"

    async def test_finds_own_comment_on_a_later_page(self, make_client):
        # U5
        api = FakeGitHubAPI(
            [comment(1, "a", "x"), comment(2, "b", "y"), comment(3, BOT, f"{COMMENT_MARKER}\nold")],
            per_page=2,
        )

        async with make_client(api) as gh:
            await gh.upsert_comment("o/r", 1, "new")

        list_calls = [c for c in api.calls if c[0] == "GET" and c[1].endswith("/comments")]
        assert len(list_calls) == 2
        assert 3 in api.patched

    async def test_login_is_fetched_once_per_client(self, make_client):
        # U6
        api = FakeGitHubAPI()

        async with make_client(api) as gh:
            await gh.upsert_comment("o/r", 1, "first")
            await gh.upsert_comment("o/r", 1, "second")

        assert api.calls.count(("GET", "/user")) == 1

    async def test_stops_listing_after_the_page_cap(self, make_client):
        # U7: a huge thread can't keep the task paging forever.
        many = [comment(i, "someone", "hi") for i in range(github_client.MAX_COMMENT_PAGES + 5)]
        api = FakeGitHubAPI(many, per_page=1)

        async with make_client(api) as gh:
            await gh.upsert_comment("o/r", 1, "the review")

        list_calls = [c for c in api.calls if c[0] == "GET" and c[1].endswith("/comments")]
        assert len(list_calls) == github_client.MAX_COMMENT_PAGES

    async def test_retry_after_lost_post_reply_updates_instead_of_duplicating(self, make_client):
        # U8, the Q1 scenario end to end: the POST succeeds on GitHub but the reply
        # is lost (ReadTimeout). The client must not blindly re-POST; the next
        # upsert (a redelivery) finds the comment and PATCHes it → still ONE comment.
        api = FakeGitHubAPI()
        api.lose_next_post_reply = True

        async with make_client(api) as gh:
            with pytest.raises(GitHubAPIError):
                await gh.upsert_comment("o/r", 1, "the review")
            assert api.methods().count("POST") == 1  # not retried

            await gh.upsert_comment("o/r", 1, "the review")

        assert len(api.comments) == 1
        assert api.methods().count("POST") == 1
        assert api.methods()[-1] == "PATCH"
