from reviewer.ingest.signature import verify_github_signature
from reviewer.ingest.github_client import GitHubClient, GitHubAPIError
from reviewer.config import settings
from reviewer.ingest.events import PREvent
from reviewer.agent.nodes import review_diff, ReviewError
from reviewer.ingest.deduplication import DeliveryStore

import logging
from fastapi import FastAPI, Request, Response, status, BackgroundTasks, Header, HTTPException
from pydantic import ValidationError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Allowed list of actions 
ALLOWED_ACTIONS = {"opened", "synchronize", "reopened"}


# In-memory store of claimed delivery IDs (lost on restart; see DeliveryStore).
# GitHub can redeliver a webhook (same X-GitHub-Delivery) for up to 3 days,
# so remember IDs at least that long.
DELIVERY_TTL_SECONDS = 3 * 24 * 60 * 60
PROCESSED_DELIVERIES = DeliveryStore(ttl_seconds=DELIVERY_TTL_SECONDS)

app = FastAPI()


async def process_pull_request(delivery_id: str, repo: str, pr_number: int, head_sha: str) -> None:
    '''Background task: review one PR at one commit.

    It receives only the three values the review needs, already validated by the
    endpoint, never the raw payload dict.

    Args:
        delivery_id (str): X-GitHub-Delivery GUID, used for dedupe and log correlation.
        repo (str): "owner/name", e.g. "HR-coding/pr-reviewer-sandbox".
        pr_number (int): The PR number.
        head_sha (str): The commit the PR points at right now (the one being reviewed).

    Returns:
        None
    '''
    if not PROCESSED_DELIVERIES.claim(delivery_id=delivery_id):
        logger.warning("[%s] Duplicate delivery ignored", delivery_id)
        return

    # On ANY failure, un-mark the delivery so a GitHub "Redeliver" is retried
    # instead of being dropped as a duplicate.
    try:
        logger.info("[%s] Reviewing %s#%s at %s", delivery_id, repo, pr_number, head_sha[:7])
        token = settings.github_token.get_secret_value()
        async with GitHubClient(token=token) as gh:
            diff = await gh.get_pr_diff(repo=repo, number=pr_number)            
            review = await review_diff(diff=diff)
            # Upsert, not post: one review comment per PR, updated on every push,
            # and safe to repeat if an earlier POST timed out after succeeding.
            comment_url = await gh.upsert_comment(repo=repo, number=pr_number, body=f"### 🤖 Review of {head_sha[:7]}\n\n{review}")
        logger.info("[%s] Posted comment on %s#%s: %s", delivery_id, repo, pr_number, comment_url)

    except ReviewError as exc:
        # LLM review failure due to rate limit issues etc
        PROCESSED_DELIVERIES.release(delivery_id=delivery_id)
        logger.error("[%s] LLM review failed (status=%s): %s", delivery_id, exc.status_code, exc.message)

    except GitHubAPIError as exc:
        # Expected failure (bad token, 404, GitHub down): one clear line, no traceback.
        PROCESSED_DELIVERIES.release(delivery_id=delivery_id)
        logger.error("[%s] GitHub API failed (status=%s): %s", delivery_id, exc.status_code, exc.message)

    except Exception:
        # Unexpected failure = a bug in our code: log the full traceback.
        PROCESSED_DELIVERIES.release(delivery_id=delivery_id)
        logger.exception("[%s] Error while processing pull request", delivery_id)


@app.post("/webhook/github", status_code=status.HTTP_202_ACCEPTED)
async def process_github_webhook(request: Request, response: Response, background_tasks: BackgroundTasks, x_hub_signature_256: str | None = Header(default=None),
                                 x_github_event: str = Header(default="unknown"), x_github_delivery: str | None = Header(default=None)) -> dict[str, str]:
    '''Verify, filter and queue a GitHub webhook delivery.

    Returns:
        200 for ping; 202 for everything else that is signed and well-formed
        (queued, or ignored on purpose).

    Raises:
        HTTPException: 401 if the signature is invalid, 400 if a pull_request
        payload doesn't match PREvent.
    '''
    raw_body = await request.body()

    # 1. Verify the HMAC signature before doing anything else.
    if not verify_github_signature(settings.webhook_secret.get_secret_value().encode(), raw_body, x_hub_signature_256):
        logger.warning(
            "[%s] Rejected webhook with invalid signature (event=%s)",
            x_github_delivery,
            x_github_event,
        )
        raise HTTPException(status_code=401, detail="Invalid webhook signature")

    # 2. Decide by event type, in the endpoint, so ignored events never get queued.
    if x_github_event == "ping":
        logger.info("[%s] PING received", x_github_delivery)
        response.status_code = status.HTTP_200_OK
        return {"status": "pong"}

    if x_github_event != "pull_request":
        logger.info("[%s] Ignoring event=%s", x_github_delivery, x_github_event)
        return {"status": "ignored"}

    # The delivery ID is the dedupe key: without it, every such request would
    # share one fake ID and all but the first would be dropped as "duplicates".
    if not x_github_delivery:
        logger.warning("Rejected pull_request without X-GitHub-Delivery")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Missing X-GitHub-Delivery header")

    # 3. Parse + validate straight from the raw bytes (bad JSON also raises ValidationError).
    try:
        event = PREvent.model_validate_json(raw_body)
    except ValidationError:
        logger.exception("[%s] Invalid pull_request payload", x_github_delivery)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid pull_request payload")

    # 4. Filter on action.
    if event.action not in ALLOWED_ACTIONS:
        logger.info("[%s] Ignoring pull_request action=%s", x_github_delivery, event.action)
        return {"status": "ignored"}

    # 5. Queue only the values the background task needs.
    logger.info(
        "[%s] Queued | repo=%s | action=%s | pr=#%s | author=%s | title=%r",
        x_github_delivery, event.repository.full_name, event.action,
        event.number, event.pull_request.user.login, event.pull_request.title,
    )
    background_tasks.add_task(
        process_pull_request,
        x_github_delivery,
        event.repository.full_name,
        event.number,
        event.pull_request.head.sha,
    )
    return {"status": "queued"}
