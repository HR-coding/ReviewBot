import hashlib
import hmac

PREFIX = "sha256="


def verify_github_signature(secret_key: bytes, body: bytes, signature_header: str | None) -> bool:
    """Verify GitHub's `X-Hub-Signature-256: sha256=<hex>` header against the raw body.

    Args:
        secret_key: The webhook secret, encoded to bytes.
        body: The raw request body, exactly as received (before any JSON parsing).
        signature_header: The X-Hub-Signature-256 header value, or None if absent.

    Returns:
        True only if the header is a sha256 signature matching the body.
        Never raises on bad header input; it returns False instead.
    """
    # Missing header or wrong algorithm (e.g. legacy "sha1=") -> reject (no downgrade).
    if signature_header is None or not signature_header.startswith(PREFIX):
        return False

    received = signature_header.removeprefix(PREFIX)
    expected = hmac.new(secret_key, body, hashlib.sha256).hexdigest()

    # Compare as bytes: compare_digest on str only accepts ASCII, so a 
    # non-ASCII header would raise TypeError. Encoding to UTF-8 always works.
    return hmac.compare_digest(expected.encode(), received.encode())
