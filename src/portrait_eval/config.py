from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="PORTRAIT_EVAL_", env_file=".env", extra="ignore")

    database_url: str = "sqlite:///./workspace/portrait_eval.db"
    workspace: Path = Path("./workspace")
    allowed_roots: list[Path] = Field(default_factory=list)
    api_token: str | None = None
    report_share_secret: str = "local-only-change-me"
    vlm_provider: Literal["local", "openai"] = "local"
    primary_vlm_url: str | None = None
    primary_vlm_model: str = "Qwen/Qwen3-VL-32B-Instruct"
    reviewer_vlm_url: str | None = None
    reviewer_vlm_model: str = "OpenGVLab/InternVL3_5-38B"
    openai_api_key: str | None = None
    openai_auth_file: Path | None = None
    openai_base_url: str = "https://api.openai.com"
    openai_primary_model: str = "gpt-5.6-sol"
    openai_reviewer_model: str = "gpt-5.6-sol"
    openai_max_output_tokens: int = Field(default=4000, ge=256, le=128000)
    search_provider: str = "disabled"
    searxng_url: str | None = None
    task_max_attempts: int = 3
    vlm_max_image_edge: int = 1536

    def prepare(self) -> None:
        self.workspace.mkdir(parents=True, exist_ok=True)
