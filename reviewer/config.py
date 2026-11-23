from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    github_token: SecretStr
    webhook_secret: SecretStr
    gemini_api_key: SecretStr = SecretStr("")
    # The API model ID (from `client.models.list()`), not the display name.
    # Must be usable on your key's free tier; override with GEMINI_MODEL in .env.
    gemini_model: str = "gemini-3.5-flash"
    anthropic_api_key: SecretStr = SecretStr("")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


# pydantic-settings fills the required fields from env vars / .env at runtime,
# but Pylance only sees a class with required fields called with no arguments.
settings = Settings()  # pyright: ignore[reportCallIssue]
