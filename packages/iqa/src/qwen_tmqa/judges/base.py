from __future__ import annotations

from typing import Protocol

from ..domain import ModelEvaluation, PromptTrace


class QualityJudge(Protocol):
    def evaluate(self, prompt_trace: PromptTrace) -> ModelEvaluation:
        """Evaluate one scene using the fully rendered and traceable prompt."""
