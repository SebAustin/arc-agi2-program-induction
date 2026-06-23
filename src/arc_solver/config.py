"""Runtime configuration.

All tunables live in a single frozen dataclass so behaviour is explicit and
reproducible. The LLM provider and key are read from the environment only — no
secret is ever hard-coded. When no key is present the engine simply omits the
LLM proposer and runs the offline path.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace


# --- Search budget --------------------------------------------------------- #


@dataclass(frozen=True)
class Budget:
    """Resource limits for a single task solve.

    Attributes:
        max_programs: Cap on candidate programs expanded by a proposer.
        max_seconds: Wall-clock budget per task (enforced by the engine/harness).
        max_refine_rounds: Refinement iterations for the feedback loop.
        max_depth: Maximum DSL pipeline length (number of ops).
    """

    max_programs: int = 30_000
    max_seconds: float = 20.0
    max_refine_rounds: int = 2
    max_depth: int = 4


# --- Global settings ------------------------------------------------------- #

# Hard guards used by the interpreter to avoid runaway memory on pathological
# programs (e.g. repeated tiling). ARC grids never exceed 30x30 by design.
MAX_GRID_DIM = 60
MAX_GRID_CELLS = 2_500


@dataclass(frozen=True)
class Settings:
    """Top-level configuration resolved from defaults + environment."""

    budget: Budget = field(default_factory=Budget)
    llm_provider: str | None = None  # "openai" | "anthropic" | None
    llm_model: str | None = None
    use_llm: bool = False

    @staticmethod
    def from_env() -> "Settings":
        """Build settings from environment variables.

        Recognised variables:
            ARC_LLM_PROVIDER — "openai" or "anthropic"
            ARC_LLM_MODEL    — provider-specific model id
            ARC_LLM_KEY / OPENAI_API_KEY / ANTHROPIC_API_KEY — credentials
        """
        provider = os.environ.get("ARC_LLM_PROVIDER")
        model = os.environ.get("ARC_LLM_MODEL")
        has_key = bool(
            os.environ.get("ARC_LLM_KEY")
            or os.environ.get("OPENAI_API_KEY")
            or os.environ.get("ANTHROPIC_API_KEY")
        )
        return Settings(
            llm_provider=provider,
            llm_model=model,
            use_llm=bool(provider) and has_key,
        )

    def with_budget(self, **kwargs) -> "Settings":
        """Return a copy with selected budget fields overridden."""
        return replace(self, budget=replace(self.budget, **kwargs))
