"""Provisional committed candidate ledger & capped history replay for W3/W4 (Section 5.2)."""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple
from krmm.v3.knowledge_registry import question_prompt
from krmm.v6.schemas import PublicUnit
from .state_ledger import LedgerEntry


def build_history_replay_pairs(
    committed_entries: Sequence[LedgerEntry],
    pub_map: Dict[str, PublicUnit],
    max_pairs: int = 8,
) -> List[Tuple[str, str]]:
    """Build deterministic capped (prompt, displayed_candidate_letter) replay pairs from provisional ledger.

    Uses ONLY provisional committed candidates (never evaluator gold!), excludes conflicted entries,
    and alternates canonical + reversed option permutation to prevent letter-bias overfitting.
    """
    active = [e for e in committed_entries if not e.conflicted and e.unit_id in pub_map]
    # Keep most recent up to max_pairs deterministically
    selected = active[-max_pairs:]
    pairs: List[Tuple[str, str]] = []
    perms = [("A", "B", "C", "D"), ("D", "C", "B", "A")]
    for idx, entry in enumerate(selected):
        pu = pub_map[entry.unit_id]
        ku = pu.to_knowledge_unit("test")
        perm = perms[idx % len(perms)]
        prompt, _, c2d = question_prompt(ku, variant=idx % 2, permutation=perm)
        disp_cand = c2d[entry.candidate]
        pairs.append((prompt, disp_cand))
    return pairs
