import asyncio
from collections.abc import Awaitable, Callable
import logging
import random
import time

import httpx

logger = logging.getLogger(__name__)

# Hidden marker (invisible once GitHub renders the Markdown) that identifies the
# bot's review comment, so a later push updates it instead of adding another.
COMMENT_MARKER = "<!-- pr-reviewer -->"

# Upper bound on comment pages to scan (100 comments each) when looking for the
# bot's own comment, so a huge thread can't keep the task paging forever.
MAX_COMMENT_PAGES = 10

# Statuses that may be temporary, but may also mean GitHub already did the work
# (a 502 can arrive after the request was processed): retried for GET/PATCH only.
RETRYABLE_SERVER_ERRORS = (500, 502, 503, 504)

# Network errors raised before the request was sent, so retrying can't
# duplicate anything, even for POST.
NEVER_SENT_ERRORS = (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout)


class GitHubAPIError(Exception):
    def __init__(self, message: str, status_code: int | None = None) -> None:
        self.status_code = status_code
        self.message = message
        prefix = f"GitHub API Error ({self.status_code})" if self.status_code else "GitHub API Network Error"
        super().__init__(f"{prefix} : {message}")


def _is_secondary_rate_limit(response: httpx.Response) -> bool:
    if response.status_code not in (403, 429):
        return False
    return "secondary rate limit" in response.text.lower()


def _is_rate_limited(response: httpx.Response) -> bool:
    """429, or a 403 that carries a rate-limit signal (a plain 403 = no permission)."""
    if response.status_code == 429:
        return True
    if response.status_code == 403:
        if "retry-after" in response.headers:
            return True
        if response.headers.get("x-ratelimit-remaining") == "0":
            return True
        if _is_secondary_rate_limit(response):
            return True
    return False


def _retry_delay(response: httpx.Response, now: float) -> float | None:
    """Return the header-requested delay in seconds, or None to fall back to backoff.

    `now` must be wall-clock Unix time (`time.time()`): `x-ratelimit-reset` is an
    epoch timestamp, not a duration.
    """
    retry_after = response.headers.get("retry-after")
    if retry_after is not None:
        try:
            return float(retry_after)
        except ValueError:
            return None

    if response.headers.get("x-ratelimit-remaining") == "0":
        reset_raw = response.headers.get("x-ratelimit-reset")
        if reset_raw is not None:
            try:
                return float(reset_raw) - now
            except ValueError:
                return None

    return None


def _should_retry(method: str, *, response: httpx.Response | None = None, error: Exception | None = None) -> bool:
    """GET is always safe to repeat; POST only when it certainly had no effect.

    PATCH counts as safe here because ours sets the whole body: repeating it
    gives the same result (not true of every PATCH in general).
    """
    is_idempotent = method.upper() in ("GET", "PATCH")

    if error is not None:
        if isinstance(error, NEVER_SENT_ERRORS):
            return True
        if isinstance(error, httpx.RequestError):
            # ReadTimeout etc.: the request may have been processed.
            return is_idempotent
        return False

    if response is not None:
        if _is_rate_limited(response):
            return True  # GitHub refused it before doing anything
        if response.status_code in RETRYABLE_SERVER_ERRORS:
            return is_idempotent
        return False

    return False


def _error(response: httpx.Response | None, error: httpx.RequestError | None) -> GitHubAPIError:
    """The GitHubAPIError for a failed attempt (chain it with `raise ... from error`)."""
    if error is not None:
        return GitHubAPIError(f"Request Error: {error}")
    assert response is not None
    return GitHubAPIError(message=response.text[:500], status_code=response.status_code)


class GitHubClient:
    def __init__(
        self,
        token: str,
        *,
        base_url: str = "https://api.github.com",
        timeout: float = 10.0,
        transport: httpx.AsyncBaseTransport | None = None,
        max_attempts: int = 4,
        base_delay: float = 1.0,
        max_delay: float = 30.0,
        max_wait: float = 60.0,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        clock: Callable[[], float] = time.time,
        rng: random.Random | None = None,
    ) -> None:
        if max_attempts < 1:
            raise ValueError(f"max_attempts must be >= 1, got {max_attempts}")
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.max_wait = max_wait
        self._sleep = sleep
        self._clock = clock
        self._rng = rng or random.Random()
        self._login: str | None = None  # the token's own login, fetched once

        self._client = httpx.AsyncClient(
            base_url=base_url,
            transport=transport,
            timeout=timeout,
            headers={
                "Authorization": f"Bearer {token}",
                "X-GitHub-Api-Version": "2022-11-28",
            },
        )

    async def __aenter__(self) -> "GitHubClient":
        return self

    async def __aexit__(self, exc_type, exc_value, exc_tb) -> None:
        await self._client.aclose()

    def _backoff(self, attempt: int) -> float:
        """Full jitter: uniform(0, min(max_delay, base_delay * 2^(attempt-1)))."""
        ceiling = min(self.max_delay, self.base_delay * (2 ** (attempt - 1)))
        return self._rng.uniform(0.0, ceiling)

    def _compute_delay(self, response: httpx.Response | None, attempt: int) -> float:
        if response is not None:
            header_delay = _retry_delay(response, self._clock())
            if header_delay is not None:
                # Clock skew can make reset - now negative.
                return max(0.0, header_delay)

            if _is_secondary_rate_limit(response):
                # GitHub: wait at least a minute, then back off exponentially.
                secondary_ceiling = max(60.0, 60.0 * (2 ** (attempt - 1)))
                return self._rng.uniform(60.0, secondary_ceiling)

        return max(0.0, self._backoff(attempt))

    async def _request(self, method: str, url: str, **kwargs) -> httpx.Response:
        for attempt in range(1, self.max_attempts + 1):
            response: httpx.Response | None = None
            error: httpx.RequestError | None = None

            try:
                response = await self._client.request(method=method, url=url, **kwargs)
            except httpx.RequestError as err:
                error = err

            if response is not None and response.is_success:
                return response

            status_label = response.status_code if response is not None else type(error).__name__

            if not _should_retry(method, response=response, error=error) or attempt == self.max_attempts:
                logger.error("GitHub API failed %s %s: status=%s", method, url, status_label)
                raise _error(response, error) from error

            delay = self._compute_delay(response, attempt)
            if delay > self.max_wait:
                # e.g. the rate limit resets in 40 minutes: don't hold a background
                # task that long. Fail now; a later redelivery can try again.
                logger.error(
                    "GitHub API failed %s %s: status=%s, wait %.1fs exceeds max_wait %.1fs",
                    method, url, status_label, delay, self.max_wait,
                )
                raise _error(response, error) from error

            logger.warning(
                "GitHub %s %s: status=%s, retrying in %.1fs (attempt %d/%d)",
                method, url, status_label, delay, attempt, self.max_attempts,
            )
            await self._sleep(delay)

        raise AssertionError("unreachable: the loop returns or raises")

    async def get_pr_diff(self, repo: str, number: int) -> str:
        response = await self._request(
            "GET",
            f"/repos/{repo}/pulls/{number}",
            headers={"Accept": "application/vnd.github.diff"},
        )
        return response.text

    async def post_comment(self, repo: str, number: int, body: str) -> str:
        response = await self._request(
            "POST",
            f"/repos/{repo}/issues/{number}/comments",
            headers={"Accept": "application/vnd.github+json"},
            json={"body": body},
        )
        return response.json()["html_url"]

    async def _get_login(self) -> str:
        """The login the token acts as (with a PAT, that's you; with an App, `name[bot]`)."""
        if self._login is None:
            response = await self._request("GET", "/user", headers={"Accept": "application/vnd.github+json"})
            login: str = response.json()["login"]
            self._login = login
            return login
        return self._login

    async def _list_comments(self, repo: str, number: int) -> list[dict]:
        comments: list[dict] = []
        url: str | None = f"/repos/{repo}/issues/{number}/comments?per_page=100"
        pages = 0

        while url is not None:
            if pages == MAX_COMMENT_PAGES:
                logger.warning("%s#%s: stopped listing comments after %d pages", repo, number, pages)
                break
            response = await self._request(
                "GET",
                url,
                headers={"Accept": "application/vnd.github+json"},
            )
            comments.extend(response.json())
            pages += 1
            next_link = response.links.get("next")
            url = next_link["url"] if next_link else None

        return comments

    async def _update_comment(self, repo: str, comment_id: int, body: str) -> str:
        response = await self._request(
            "PATCH",
            f"/repos/{repo}/issues/comments/{comment_id}",
            headers={"Accept": "application/vnd.github+json"},
            json={"body": body},
        )
        return response.json()["html_url"]

    async def upsert_comment(self, repo: str, number: int, body: str) -> str:
        """Update the bot's review comment on the PR, or create it if there isn't one.

        Safe to repeat after a timeout: if a "lost" POST actually succeeded, the
        listing finds that comment and it gets updated, not duplicated.
        A comment counts as ours only if it has the marker AND was written by the
        token's own login: anyone can type the (invisible) marker.
        """
        marked_body = body if COMMENT_MARKER in body else f"{COMMENT_MARKER}\n{body}"
        login = await self._get_login()

        for comment in await self._list_comments(repo, number):
            author = (comment.get("user") or {}).get("login")
            if author == login and COMMENT_MARKER in (comment.get("body") or ""):
                return await self._update_comment(repo, comment["id"], marked_body)

        return await self.post_comment(repo, number, marked_body)
