"""Strict query/token/call budget proxy separating logical queries vs physical forwards."""
from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple


class BudgetExhaustedError(RuntimeError):
    """Raised when a method attempts to exceed its allocated query budget."""


class StrictBudgetProxy:
    """Wraps a model backend with hard query budget, real tokenizer token counting, and cache accounting."""

    def __init__(
        self,
        backend,
        query_budget: int,
        use_cache: bool = True,
        shared_prob_cache: Optional[Dict[str, Tuple[List[float], float]]] = None,
        shared_gen_cache: Optional[Dict[Tuple[str, int], str]] = None,
    ) -> None:
        self._backend = backend
        self.query_budget = int(query_budget)
        self.use_cache = bool(use_cache)
        self._prob_cache = shared_prob_cache if shared_prob_cache is not None else {}
        self._gen_cache = shared_gen_cache if shared_gen_cache is not None else {}

        self.logical_queries: int = 0
        self.physical_calls: int = 0
        self.cache_hits: int = 0
        self.cache_misses: int = 0
        self.generate_calls: int = 0
        self.total_tokens: int = 0
        self.token_counter_mode: str = self._detect_token_mode()
        self.wall_seconds_total: float = 0.0
        self.wall_seconds_physical: float = 0.0

    def _detect_token_mode(self) -> str:
        tok = getattr(self._backend, "tokenizer", None)
        if tok is not None and hasattr(tok, "encode"):
            return "real_tokenizer"
        proc = getattr(self._backend, "processor", None)
        if proc is not None and hasattr(proc, "tokenizer") and hasattr(proc.tokenizer, "encode"):
            return "real_processor_tokenizer"
        return "approx_whitespace_fallback"

    def _count_tokens(self, text: str) -> int:
        tok = getattr(self._backend, "tokenizer", None)
        if tok is not None and hasattr(tok, "encode"):
            return len(tok.encode(text, add_special_tokens=False))
        proc = getattr(self._backend, "processor", None)
        if proc is not None and hasattr(proc, "tokenizer") and hasattr(proc.tokenizer, "encode"):
            return len(proc.tokenizer.encode(text, add_special_tokens=False))
        return max(1, len(text.split()))

    def remaining(self) -> int:
        return max(0, self.query_budget - self.logical_queries)

    def probabilities(self, prompt: str, media=None) -> Tuple[List[float], float]:
        if self.logical_queries >= self.query_budget:
            raise BudgetExhaustedError(
                f"Logical query budget exhausted ({self.logical_queries}/{self.query_budget})"
            )
        t0 = time.perf_counter()
        self.logical_queries += 1
        self.total_tokens += self._count_tokens(prompt)
        if self.use_cache and prompt in self._prob_cache:
            self.cache_hits += 1
            res = self._prob_cache[prompt]
            self.wall_seconds_total += time.perf_counter() - t0
            return res
        self.cache_misses += 1
        self.physical_calls += 1
        tp0 = time.perf_counter()
        probs, mass = self._backend.probabilities(prompt)
        dt_p = time.perf_counter() - tp0
        self.wall_seconds_physical += dt_p
        res = ([float(p) for p in probs], float(mass))
        if self.use_cache:
            self._prob_cache[prompt] = res
        self.wall_seconds_total += time.perf_counter() - t0
        return res

    def generate(self, prompt: str, max_new_tokens: int = 24, media=None) -> str:
        if self.logical_queries >= self.query_budget:
            raise BudgetExhaustedError(
                f"Logical query budget exhausted ({self.logical_queries}/{self.query_budget})"
            )
        t0 = time.perf_counter()
        self.logical_queries += 1
        self.generate_calls += 1
        self.total_tokens += self._count_tokens(prompt)
        key = (prompt, int(max_new_tokens))
        if self.use_cache and key in self._gen_cache:
            self.cache_hits += 1
            out = self._gen_cache[key]
            self.total_tokens += self._count_tokens(out)
            self.wall_seconds_total += time.perf_counter() - t0
            return out
        self.cache_misses += 1
        self.physical_calls += 1
        tp0 = time.perf_counter()
        out = str(self._backend.generate(prompt, max_new_tokens=max_new_tokens))
        dt_p = time.perf_counter() - tp0
        self.wall_seconds_physical += dt_p
        self.total_tokens += self._count_tokens(out)
        if self.use_cache:
            self._gen_cache[key] = out
        self.wall_seconds_total += time.perf_counter() - t0
        return out

    def accounting(self) -> Dict[str, object]:
        return {
            "query_budget": self.query_budget,
            "logical_queries": self.logical_queries,
            "physical_calls": self.physical_calls,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "generate_calls": self.generate_calls,
            "total_tokens": self.total_tokens,
            "token_counter_mode": self.token_counter_mode,
            "wall_seconds_total": self.wall_seconds_total,
            "wall_seconds_physical": self.wall_seconds_physical,
            "cold_cache_timing_valid": self.cache_hits == 0,
        }
