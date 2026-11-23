import json
import httpx
import pytest
from reviewer.ingest.github_client import GitHubAPIError, GitHubClient

TOKEN = "ghp_TEST_SECRET_123"

@pytest.fixture
def make_client():
    def _factory(handler):
        return GitHubClient(TOKEN, transport=httpx.MockTransport(handler))
    return _factory


@pytest.mark.anyio
async def test_get_pr_diff_request_shape(make_client):  
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
async def test_non_2xx_raises_github_api_error(make_client, status):  # G3
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, text="boom")

    async with make_client(handler) as gh:
        with pytest.raises(GitHubAPIError) as exc_info:
            await gh.get_pr_diff("o/r", 1)

    assert exc_info.value.status_code == status
    assert "boom" in exc_info.value.message


@pytest.mark.anyio
async def test_network_error_raises_chained_github_api_error(make_client):  
    original_err = httpx.ConnectError("down")

    def handler(request: httpx.Request) -> httpx.Response:
        raise original_err

    async with make_client(handler) as gh:
        with pytest.raises(GitHubAPIError) as exc_info:
            await gh.get_pr_diff("o/r", 1)

    assert exc_info.value.status_code is None
    assert exc_info.value.__cause__ is original_err


@pytest.mark.anyio
async def test_token_never_leaks_in_logs(make_client, caplog):  
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, text="Bad credentials")

    async with make_client(handler) as gh:
        with pytest.raises(GitHubAPIError):
            await gh.get_pr_diff("o/r", 1)

    assert TOKEN not in caplog.text


