from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

# The project root: this file is <root>/reviewer/config.py, so two levels up.
# Anchoring .env here (instead of a bare ".env", which pydantic-settings resolves
# against the CURRENT directory) means the same .env is found no matter where
# the process is started from: the repo root, scripts/, an IDE, a test runner.
PROJECT_ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    github_token: SecretStr = Field(min_length=1)
    # An empty secret would make HMAC signatures forgeable by anyone (the key
    # is ""), so refuse to start rather than run unprotected.
    webhook_secret: SecretStr = Field(min_length=16)
    gemini_api_key: SecretStr = SecretStr("")
    # The API model ID (from `client.models.list()`), not the display name.
    # Must be usable on your key's free tier; override with GEMINI_MODEL in .env.
    gemini_model: str = "gemini-3.5-flash"
    anthropic_api_key: SecretStr = SecretStr("")

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


# pydantic-settings fills the required fields from env vars / .env at runtime,
# but Pylance only sees a class with required fields called with no arguments.
settings = Settings()  # pyright: ignore[reportCallIssue]
