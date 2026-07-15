from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="PORTRAIT_EVAL_", env_file=".env", extra="ignore")

    database_url: str = "sqlite:///./workspace/portrait_eval.db"
    workspace: Path = Path("./workspace")
    allowed_roots: list[Path] = Field(default_factory=list)
    api_token: str | None = None
    report_share_secret: str = "local-only-change-me"
    primary_vlm_url: str | None = None
    primary_vlm_model: str = "Qwen/Qwen3-VL-32B-Instruct"
    reviewer_vlm_url: str | None = None
    reviewer_vlm_model: str = "OpenGVLab/InternVL3_5-38B"
    search_provider: str = "disabled"
    searxng_url: str | None = None
    task_max_attempts: int = 3
    vlm_max_image_edge: int = 1536

    def prepare(self) -> None:
        self.workspace.mkdir(parents=True, exist_ok=True)
