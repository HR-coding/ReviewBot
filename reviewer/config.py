from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    github_token: SecretStr
    webhook_secret: SecretStr
    gemini_api_key: SecretStr = SecretStr("")
    anthropic_api_key: SecretStr = SecretStr("")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


# pydantic-settings fills the required fields from env vars / .env at runtime,
# but Pylance only sees a class with required fields called with no arguments.
settings = Settings()  # pyright: ignore[reportCallIssue]
