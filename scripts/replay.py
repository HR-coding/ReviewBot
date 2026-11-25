import argparse
import hashlib
import hmac
from pathlib import Path
import sys
import uuid

import httpx

from reviewer.config import settings

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "tests" / "fixtures"


def load_fixture(name: str) -> bytes:
    """Read `tests/fixtures/<name>.json` as raw bytes ("ping" or "ping.json")."""
    filename = name if name.endswith(".json") else f"{name}.json"
    return (FIXTURES_DIR / filename).read_bytes()


def sign(body: bytes, secret: bytes) -> str:
    """Compute GitHub's HMAC-SHA256 webhook signature (`sha256=<hex>`)."""
    digest = hmac.new(secret, body, hashlib.sha256).hexdigest()
    return f"sha256={digest}"


def build_headers(event: str, delivery_id: str, signature: str | None) -> dict[str, str]:
    """Assemble the GitHub webhook headers."""
    headers = {
        "Content-Type": "application/json",
        "X-GitHub-Event": event,
        "X-GitHub-Delivery": delivery_id,
    }
    if signature is not None:
        headers["X-Hub-Signature-256"] = signature
    return headers


def send(url: str, body: bytes, headers: dict[str, str]) -> httpx.Response:
    """POST raw fixture bytes to the webhook endpoint with a timeout."""
    return httpx.post(url, content=body, headers=headers, timeout=10.0)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay a saved GitHub webhook fixture.")
    parser.add_argument("fixture", help='Fixture name (e.g. "ping" or "pull_request_opened")')
    parser.add_argument(
        "--event",
        default=None,
        help='X-GitHub-Event header (defaults to "ping" for ping fixture, else "pull_request")',
    )
    parser.add_argument(
        "--delivery-id",
        default=None,
        help="X-GitHub-Delivery UUID (defaults to a fresh uuid4 each run)",
    )
    parser.add_argument(
        "--url",
        default="http://localhost:8000/webhook/github",
        help="Target webhook URL (default: http://localhost:8000/webhook/github)",
    )
    parser.add_argument(
        "--bad-signature",
        action="store_true",
        help="Sign with an invalid secret to test 401 rejection",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        body = load_fixture(args.fixture)
    except FileNotFoundError:
        print(f"Error: fixture not found: {args.fixture}", file=sys.stderr)
        return 1

    fixture_stem = Path(args.fixture).stem
    event = args.event if args.event is not None else ("ping" if fixture_stem == "ping" else "pull_request")
    delivery_id = args.delivery_id or str(uuid.uuid4())

    if args.bad_signature:
        secret = b"wrong-secret"
    else:
        secret = settings.webhook_secret.get_secret_value().encode("utf-8")

    signature = sign(body, secret)
    headers = build_headers(event=event, delivery_id=delivery_id, signature=signature)

    try:
        response = send(url=args.url, body=body, headers=headers)
    except httpx.HTTPError as exc:
        print(f"Request failed: {exc}", file=sys.stderr)
        return 1

    print(f"{response.status_code} {response.text}")
    return 0 if response.is_success else 1


if __name__ == "__main__":
    sys.exit(main())
