"""Writeback adapter interface for KRMM v0.4 (fixed standard_kl primary + ablations; CPU mockable)."""
from __future__ import annotations

from typing import Dict, Optional

SUPPORTED_WRITE_STRATEGIES = frozenset(
    {"standard_kl", "orthogonal_retain_proj", "unconstrained_lora", "nonparametric_memory"}
)


class RepairAdapter:
    """Unified repair writeback adapter; CPU-safe mock mode for pipeline & policy testing."""

    def __init__(self, strategy: str = "standard_kl", mock_mode: bool = False) -> None:
        if strategy not in SUPPORTED_WRITE_STRATEGIES:
            raise ValueError(f"Unsupported write strategy: {strategy}")
        self.strategy = strategy
        self.mock_mode = bool(mock_mode)

    def apply_repair(
        self,
        unit_id: str,
        candidate_answer: str,
        mock_executor=None,
        **kwargs,
    ) -> Dict[str, object]:
        if not self.mock_mode:
            raise RuntimeError(
                "Live GPU parameter writeback is disabled during PLAN/CPU validation; "
                "requires explicit GPU authorization block."
            )
        if mock_executor is not None:
            return mock_executor(unit_id, candidate_answer, self.strategy)
        return {
            "unit_id": unit_id,
            "strategy": self.strategy,
            "candidate_written": candidate_answer,
            "adapter_saved": True,
            "mock_executed": True,
        }
