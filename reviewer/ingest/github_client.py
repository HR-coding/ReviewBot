import httpx
import logging

logger = logging.getLogger(__name__)

class GitHubAPIError(Exception):
    def __init__(self, message: str, status_code: int | None = None) -> None:
        self.status_code = status_code
        self.message = message
        prefix = f"GitHub API Error ({self.status_code})" if self.status_code else "GitHub API Network Error"
        super().__init__(f"{prefix} : {message}")

class GitHubClient:
    def __init__(self, token: str, *, base_url: str = "https://api.github.com", 
                timeout: float = 10.0, transport: httpx.AsyncBaseTransport | None = None) -> None:
        self._client = httpx.AsyncClient(base_url=base_url, transport=transport, timeout=timeout,
                                         headers={
                                             "Authorization": f"Bearer {token}",
                                             "X-GitHub-Api-Version": "2022-11-28"
                                         })

    async def __aenter__(self) -> "GitHubClient":
        return self

    async def __aexit__(self, exc_type, exc_value, exc_tb) -> None:
        await self._client.aclose()

    async def _request(self, method: str, url: str, **kwargs) -> httpx.Response:
        try:
            response = await self._client.request(method=method, url=url, **kwargs)
        except httpx.RequestError as err:
            logger.error("GitHub network failure %s %s: %s", method, url, type(err).__name__)
            raise GitHubAPIError(f"Request Error: {err}") from err
        if not response.is_success:
            logger.error("GitHub API failed %s %s: status=%s", method, url, response.status_code)
            raise GitHubAPIError(message=response.text[:500], status_code=response.status_code)
        return response

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