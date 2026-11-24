import pytest
from pydantic import ValidationError

from reviewer.config import Settings


class TestSettings:
    """Settings refuse to start with secrets that would leave the app unprotected.

    `_env_file=None` stops `.env` from being read, so only the values passed in
    (and the test env from conftest) count.
    """

    @pytest.mark.parametrize("secret", ["", "short-secret"], ids=["empty", "under-16"])
    def test_weak_webhook_secret_is_rejected(self, monkeypatch, secret):
        # an empty HMAC key lets anyone forge a valid signature.
        monkeypatch.setenv("WEBHOOK_SECRET", secret)

        with pytest.raises(ValidationError, match="webhook_secret"):
            Settings(_env_file=None)  # pyright: ignore[reportCallIssue]

    def test_empty_github_token_is_rejected(self, monkeypatch):
        monkeypatch.setenv("GITHUB_TOKEN", "")

        with pytest.raises(ValidationError, match="github_token"):
            Settings(_env_file=None)  # pyright: ignore[reportCallIssue]

    def test_valid_settings_load(self):
        # the conftest test values are valid (otherwise every test would fail at import)
        s = Settings(_env_file=None)  # pyright: ignore[reportCallIssue]

        assert len(s.webhook_secret.get_secret_value()) >= 16
