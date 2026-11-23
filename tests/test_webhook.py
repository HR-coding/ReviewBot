import pytest
import json
import logging
from typing import Any
from collections.abc import Callable

from reviewer.ingest import webhook
from reviewer.ingest.github_client import GitHubAPIError

def modified(body: bytes, change: Callable[[dict[str, Any]], None]) -> bytes:
    """Parse JSON bytes, mutate the dict via `change(data)`, and re-encode."""
    data = json.loads(body)
    change(data)
    return json.dumps(data).encode("utf-8")


def count_logs(caplog: pytest.LogCaptureFixture, text: str) -> int:
    """How many captured log lines contain `text` (with %s args filled in)."""
    return sum(text in record.getMessage() for record in caplog.records)


@pytest.fixture
def opened(load_fixture) -> bytes:
    """Raw bytes of the real `pull_request` / `opened` payload, the base for most variants."""
    return load_fixture("pull_request_opened")


class TestSecurity:
    """The signature is verified FIRST, before routing, parsing or queueing.

    Every request here lacks a valid signature, so every one must get a 401 and
    queue nothing:
    - missing header, a header made with the wrong secret (authenticity), and a
      body changed after signing (integrity);
    - an unsigned `ping`: 401, not 200, which proves verification happens before
      branching on the (unsigned) event header;
    - unsigned malformed JSON: 401, not 400, which proves verification happens
      before parsing, so unauthenticated input never reaches the parser.
    """

    def test_missing_signature_returns_401(self, post_webhook, opened, queued):
        resp = post_webhook("pull_request", opened, signed=False)

        assert resp.status_code == 401
        assert queued == []

    def test_wrong_secret_returns_401(self, post_webhook, opened, queued):
        resp = post_webhook("pull_request", opened, secret=b"wrong-secret")

        assert resp.status_code == 401
        assert queued == []

    def test_tampered_body_returns_401(self, post_webhook, sign, opened, queued):
        sig = sign(opened)
        tampered_body = opened + b" "  # single trailing space breaks HMAC

        resp = post_webhook("pull_request", tampered_body, signature=sig)

        assert resp.status_code == 401
        assert queued == []

    def test_unsigned_ping_returns_401(self, post_webhook, load_fixture, queued):
        # Signature check must happen BEFORE event routing: ping is the branch that
        # returns 200 early, so it's where an unverified answer would slip out.
        resp = post_webhook("ping", load_fixture("ping"), signed=False)

        assert resp.status_code == 401
        assert queued == []

    def test_unauthenticated_malformed_json_returns_401(self, post_webhook, queued):
        # Signature check must happen BEFORE JSON parsing (otherwise 400/422)
        resp = post_webhook("pull_request", b"{not-valid-json", signed=False)

        assert resp.status_code == 401
        assert queued == []

class TestEventRouting:
    """Signed requests are routed by the `X-GitHub-Event` header.

    - `ping` (sent when the webhook is created) → 200 `pong`; no work is queued.
    - Any event other than `pull_request` → 202 `ignored`; nothing is queued,
      even if its body looks like a PR event.
    """

    def test_ping_event_returns_200(self, post_webhook, load_fixture, queued):
        body = load_fixture("ping")

        resp = post_webhook("ping", body)

        assert resp.status_code == 200
        assert resp.json()["status"] == "pong"
        assert queued == []

    def test_unhandled_event_returns_202_ignored(self, post_webhook, opened, queued):
        resp = post_webhook("issues", opened)

        assert resp.status_code == 202
        assert resp.json()["status"] == "ignored"
        assert queued == []

class TestActionFilter:
    """Only `pull_request` actions that change the code get a review.

    - `opened`, `synchronize` (a new push) and `reopened` → 202 `queued`. The
      background task is called exactly once with (delivery_id, repo full_name,
      PR number, head sha), and the expected values are read from the payload,
      not hard-coded. For `synchronize`, the sha must be the NEW head commit.
    - `closed`, `labeled`, `edited`, `assigned`, ... → 202 `ignored`, nothing
      queued. Each queued task will mean a paid LLM call and a posted comment,
      so the filter runs in the endpoint, before `add_task`.

    Uses the `queued` fixture (a recorder in place of the real task).
    """

    def test_opened_queues_review(self, post_webhook, opened, queued):
        payload = json.loads(opened)
        expected_repo = payload["repository"]["full_name"]
        expected_number = payload["number"]
        expected_sha = payload["pull_request"]["head"]["sha"]
        delivery = "delivery"

        resp = post_webhook("pull_request", opened, delivery=delivery)

        assert resp.status_code == 202
        assert resp.json()["status"] == "queued"
        assert queued == [(delivery, expected_repo, expected_number, expected_sha)]

    def test_synchronize_queues_review_at_new_sha(self, post_webhook, load_fixture, opened, queued):
        # Real synchronize payload: a new push, so the head sha must be the NEW one.
        body = load_fixture("pull_request_synchronize")
        payload = json.loads(body)
        assert payload["pull_request"]["head"]["sha"] != json.loads(opened)["pull_request"]["head"]["sha"]
        delivery = "deliv-w9"

        resp = post_webhook("pull_request", body, delivery=delivery)

        assert resp.status_code == 202
        assert resp.json()["status"] == "queued"
        assert queued == [
            (
                delivery,
                payload["repository"]["full_name"],
                payload["number"],
                payload["pull_request"]["head"]["sha"],
            )
        ]

    def test_reopened_queues_review(self, post_webhook, opened, queued):
        body = modified(opened, lambda d: d.update({"action": "reopened"}))
        payload = json.loads(body)
        delivery = "deliv-w10"

        resp = post_webhook("pull_request", body, delivery=delivery)

        assert resp.status_code == 202
        assert resp.json()["status"] == "queued"
        assert queued == [
            (
                delivery,
                payload["repository"]["full_name"],
                payload["number"],
                payload["pull_request"]["head"]["sha"],
            )
        ]

    def test_closed_action_ignored(self, post_webhook, opened, queued):
        body = modified(opened, lambda d: d.update({"action": "closed"}))

        resp = post_webhook("pull_request", body)

        assert resp.status_code == 202
        assert resp.json()["status"] == "ignored"
        assert queued == []

    @pytest.mark.parametrize("action", ["labeled", "unlabeled", "assigned", "edited", "review_requested"])
    def test_other_actions_ignored(self, post_webhook, opened, queued, action):
        body = modified(opened, lambda d: d.update({"action": action}))

        resp = post_webhook("pull_request", body)

        assert resp.status_code == 202
        assert resp.json()["status"] == "ignored"
        assert queued == []

class TestValidation:
    """Signed `pull_request` events whose payload doesn't fit `PREvent` → 400, nothing queued.

    Covers malformed JSON, missing required fields (repository, number) and an
    empty head sha (`min_length=1`). The request is authentic, but the content
    is unusable, so the endpoint rejects it honestly instead of queueing a task
    that would crash later. Each case is a function that builds its bad body from
    the `opened` fixture, because parametrize values can't use fixtures directly.
    """

    @pytest.mark.parametrize(
        "make_body",
        [
            pytest.param(lambda _: b"{not-json", id="malformed-json"),
            pytest.param(lambda b: modified(b, lambda d: d.pop("repository", None)), id="missing-repository"),
            pytest.param(lambda b: modified(b, lambda d: d.pop("number", None)), id="missing-number"),
            pytest.param(lambda b: modified(b, lambda d: d["pull_request"]["head"].update({"sha": ""})),id="empty-sha"),
        ],
    )
    def test_invalid_payloads_return_400(
        self, post_webhook, opened, queued, make_body
    ):
        body = make_body(opened)

        resp = post_webhook("pull_request", body)

        assert resp.status_code == 400
        assert queued == []

class TestDeduplication:
    """The background task processes each `X-GitHub-Delivery` ID at most once.

    - Same delivery twice → reviewed and posted once, then "Duplicate delivery ignored".
    - Same payload under two different delivery IDs → reviewed twice (dedupe is
      per delivery, not per PR).
    - A delivery whose task FAILED is un-marked, so a redelivery is retried.

    Uses the REAL `process_pull_request` (not `queued`) with `fake_github` in
    place of `GitHubClient`, so no request reaches GitHub. Dedupe happens after
    the response is sent, so the endpoint answers "queued" both times; only the
    logs and the fake's recorded comments show the difference. The asserts count
    lines only the task produces (`"[id] Reviewing"`), because the endpoint's own
    "Queued" line also contains the delivery ID.
    """

    def test_duplicate_delivery_is_ignored(self, post_webhook, opened, caplog, fake_github):
        caplog.set_level(logging.INFO, logger="reviewer.ingest.webhook")
        delivery = "deliv-w17"

        first = post_webhook("pull_request", opened, delivery=delivery)
        second = post_webhook("pull_request", opened, delivery=delivery)

        # Dedupe runs in the background task, after the response is sent,
        # so the endpoint can only say "queued" both times.
        assert first.status_code == second.status_code == 202
        assert first.json()["status"] == second.json()["status"] == "queued"

        # Count the lines ONLY the task produces (the endpoint's "Queued" line also
        # contains the delivery ID, so "contains the ID" would prove nothing).
        assert count_logs(caplog, f"[{delivery}] Reviewing") == 1
        assert count_logs(caplog, f"[{delivery}] Duplicate delivery ignored") == 1
        # The side effect that matters: exactly one comment on the PR.
        assert len(fake_github.comments_posted) == 1

    def test_distinct_deliveries_both_process(self, post_webhook, opened, caplog, fake_github):
        caplog.set_level(logging.INFO, logger="reviewer.ingest.webhook")
        delivery_1 = "deliv-w18-a"
        delivery_2 = "deliv-w18-b"  # must differ, or this is W17 again

        post_webhook("pull_request", opened, delivery=delivery_1)
        post_webhook("pull_request", opened, delivery=delivery_2)

        # Dedupe is per delivery ID, not per PR: same payload, two IDs → two reviews.
        assert count_logs(caplog, f"[{delivery_1}] Reviewing") == 1
        assert count_logs(caplog, f"[{delivery_2}] Reviewing") == 1
        assert count_logs(caplog, "Duplicate delivery ignored") == 0
        assert len(fake_github.comments_posted) == 2

    def test_failed_delivery_is_retried(self, post_webhook, opened, caplog, fake_github):
        # GitHub's API fails → the task must un-mark the delivery ID, so when
        # GitHub (or you, via "Redeliver") sends the same delivery again, it runs again.
        caplog.set_level(logging.INFO, logger="reviewer.ingest.webhook")
        fake_github.error_to_raise = GitHubAPIError("server error", 500)
        delivery = "deliv-w19"

        post_webhook("pull_request", opened, delivery=delivery)
        post_webhook("pull_request", opened, delivery=delivery)  # the redelivery

        # Both attempts got past the dedupe check: the ID was discarded after attempt 1.
        assert count_logs(caplog, f"[{delivery}] Reviewing") == 2
        assert count_logs(caplog, f"[{delivery}] Duplicate delivery ignored") == 0
        assert count_logs(caplog, f"[{delivery}] GitHub API failed (status=500)") == 2
        assert fake_github.comments_posted == []  # the diff fetch failed, so nothing posted
        assert delivery not in webhook.PROCESSED_DELIVERIES


class TestPostedComment:
    """What the task posts: the placeholder summary, on the right PR.

    The fake returns a known diff (`FakeGitHubClient.diff_to_return`: 1 file,
    1 added line), so the expected counts are exact. The repo, number and sha
    are read from the payload, not hard-coded.
    """

    def test_posts_summary_on_the_pr(self, post_webhook, opened, fake_github):
        payload = json.loads(opened)

        post_webhook("pull_request", opened, delivery="deliv-w20")

        assert len(fake_github.comments_posted) == 1
        repo, number, body = fake_github.comments_posted[0]
        assert repo == payload["repository"]["full_name"]
        assert number == payload["number"]
        assert payload["pull_request"]["head"]["sha"][:7] in body
        assert "**Files changed:** 1" in body
        assert "+1 / -0" in body
