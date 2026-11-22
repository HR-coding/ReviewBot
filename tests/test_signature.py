import hashlib
import hmac
import pytest
from reviewer.ingest.signature import verify_github_signature

# GitHub's official webhook signature test vector
SECRET = b"It's a Secret to Everybody"
BODY = b"Hello, World!"
VALID = "sha256=757107ea0eb2509fc211221cce984b8a37570b6d7586c22c46f4379c8b043e17"
BARE_HEX = VALID.removeprefix("sha256=")

class TestSignature:
    '''Unit tests for `verify_github_signature` (reviewer/ingest/signature.py).

    Uses GitHub's documented test vector (SECRET, BODY, VALID above), so the
    positive case checks our HMAC against GitHub's own numbers, not our own.

    - Valid signature → True. This is the one positive test, and it gives the
      negative tests their meaning: a function that always returned False would
      pass every other test here.
    - Tampered body → False: integrity (the content changed in transit).
    - Signature made with a different secret → False: authenticity (a forger
      without the secret).
    - Bad headers (missing, `sha1=` downgrade, no prefix, empty digest) → False.
    - Non-ASCII header → False *without raising*: the verifier never crashes on
      hostile input.

    Every negative case asserts `is False`, not just "falsy", so a function that
    accidentally returns None still fails.
    '''

    def test_valid_signature_accepted(self):
        assert verify_github_signature(body=BODY, secret_key=SECRET, signature_header=VALID) is True


    def test_tampered_body_rejected(self):
        tampered_body = b"Hello, World?"
        assert verify_github_signature(body=tampered_body, secret_key=SECRET, signature_header=VALID) is False


    def test_wrong_secret_rejected(self):
        wrong_mac = hmac.new(b"wrong-secret", msg=BODY, digestmod=hashlib.sha256).hexdigest()
        wrong_header = f"sha256={wrong_mac}"
        assert verify_github_signature(body=BODY, secret_key=SECRET, signature_header=wrong_header) is False


    @pytest.mark.parametrize("bad_header",
        [
            None,#  missing, 
            f"sha1={BARE_HEX}",# sha1 downgrade attempt
            BARE_HEX,#  no prefix
            "sha256=",#  empty digest
        ],
        ids=["missing", "sha1", "no-prefix", "empty-digest"],)
    def test_bad_headers_rejected(self, bad_header: str | None):
        assert verify_github_signature(body=BODY, secret_key=SECRET, signature_header=bad_header) is False


    def test_non_ascii_header_rejected_without_exception(self):
        non_ascii_header = "sha256=é" + "0" * 63
        assert verify_github_signature(body=BODY, secret_key=SECRET, signature_header=non_ascii_header) is False