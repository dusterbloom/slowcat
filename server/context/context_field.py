"""
Deterministic Context Field Planner (MVP)

Computes a reproducible context "plan" with tier activations and token
budgets based on simple, deterministic features of the query, facts, and
recent conversation. Produces a reproducibility hash over canonicalized
inputs and outputs for verification.

This module intentionally avoids randomness and wall-clock time in the
planning path to guarantee reproducibility for the same inputs.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple
import hashlib
import json
import math


# ---- Data structures ----


@dataclass(frozen=True)
class FactMini:
    """Minimal fact record used by the planner (canonicalizable)."""
    subject: str
    predicate: str
    value: Optional[str] = None
    fidelity: int = 0
    last_seen: float = 0.0  # ignored in hash; kept for possible heuristics


@dataclass
class ContextPlan:
    """A deterministic plan for building context."""
    tier_activations: Dict[str, float]
    token_budgets: Dict[str, int]  # keys: system, facts, snippets, dth, recent, current, buffer
    token_budget_total: int
    notes: Dict[str, Any]


@dataclass
class PlanMetrics:
    """Plan metrics with reproducibility information."""
    reproducibility_hash: str
    total_tokens_budgeted: int
    tier_activations: Dict[str, float]
    token_budgets: Dict[str, int]


def _normalize_text(s: str) -> str:
    s = (s or "").strip()
    # Collapse whitespace deterministically
    return " ".join(s.split())


def _canonical_inputs(
    query: str,
    facts: List[FactMini],
    recents: List[Tuple[str, Optional[str]]],
    dth_texts: Optional[List[str]],
    base_total_budget: int,
) -> Dict[str, Any]:
    # Canonicalize facts (sorted by subject, predicate, value); we do not use last_seen in hash
    facts_sorted = sorted(
        (
            {
                "subject": (f.subject or ""),
                "predicate": (f.predicate or ""),
                "value": (f.value or None),
                "fidelity": int(f.fidelity),
            }
            for f in facts
        ),
        key=lambda x: (x["subject"], x["predicate"], x["value"] or "", -x["fidelity"]),
    )

    # Canonicalize recents (normalize strings)
    recents_norm = [
        [
            _normalize_text(pair[0] if len(pair) > 0 and pair[0] else ""),
            _normalize_text(pair[1] if len(pair) > 1 and pair[1] else ""),
        ]
        for pair in recents
    ]

    # Canonicalize DTH texts if any
    dth_norm = sorted([_normalize_text(t) for t in (dth_texts or [])])

    return {
        "query": _normalize_text(query),
        "facts": facts_sorted,
        "recents": recents_norm,
        "dth": dth_norm,
        "budget_total": int(base_total_budget),
    }


def _sha256(obj: Any) -> str:
    enc = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(enc.encode("utf-8")).hexdigest()


class ContextField:
    """
    Deterministic field-based planner that outputs tier activations and budgets.

    Tiers (logical):
    - core: system + session info
    - dynamic: facts + summary/snippets + recent window
    - retrieval: dth/snippets (verbatim when available)
    - tools: tool docs budget hint (not injected here)
    """

    def __init__(self, total_budget: int = 4096) -> None:
        self.total_budget = int(total_budget)
        # Floors and caps (tweakable, deterministic)
        import os
        self._unified = os.getenv('SC_UNIFIED_MEMORY', 'false').lower() == 'true'
        # Default floors/caps for classic (non-unified) mode
        self.floors = {
            "system": 400,
            "facts": 300,
            "recent": 1000,
            "current": 200,
            "buffer": 96,
        }
        self.caps = {
            "facts": 1200,
            "recent": 2600,
            "snippets": 400,
            "dth": 400,
        }

    # ---- Feature extraction ----
    def _features(self, query: str) -> Dict[str, Any]:
        q = _normalize_text(query).lower()
        is_question = ("?" in q) or any(q.startswith(w) for w in ("what", "where", "when", "who", "why", "how"))
        is_command = any(w in q for w in ("play ", "stop ", "pause ", "skip ", "start "))
        length = max(1, len(q.split()))
        return {"is_question": is_question, "is_command": is_command, "len": length}

    def _tier_activations(self, feats: Dict[str, Any]) -> Dict[str, float]:
        # Simple deterministic rules
        a_core = 0.6 + (0.2 if feats["is_question"] else 0.0)
        a_dynamic = 0.7
        a_retrieval = 0.2 + (0.2 if feats["is_question"] else 0.0)
        a_tools = 0.2 + (0.3 if feats["is_command"] else 0.0)
        # Normalize for readability; not required for functioning
        total = a_core + a_dynamic + a_retrieval + a_tools
        return {
            "core": round(a_core / total, 4),
            "dynamic": round(a_dynamic / total, 4),
            "retrieval": round(a_retrieval / total, 4),
            "tools": round(a_tools / total, 4),
        }

    # ---- Planning ----
    def create_plan(
        self,
        query: str,
        facts: List[FactMini],
        recents: List[Tuple[str, Optional[str]]],
        dth_texts: Optional[List[str]] = None,
    ) -> Tuple[ContextPlan, PlanMetrics]:
        """Create a deterministic plan for budgets and tier activations.

        The plan itself is reproducible for identical inputs.
        """
        # Pre-compute canonical inputs for reproducibility and debugging
        inputs = _canonical_inputs(query, facts, recents, dth_texts, self.total_budget)

        feats = self._features(query)
        activ = self._tier_activations(feats)

        # Unified memory mode: follow friend's allocation exactly for input
        if self._unified:
            # Friend's proposal (10% system, 35% contextual, 10% current, 45% generation)
            # We only budget the input side here (55% of total). We still expose full total
            # and report an implicit generation workspace of ~45% (not part of messages).
            sys_b = int(math.floor(self.total_budget * 0.10))
            ctx_b = int(math.floor(self.total_budget * 0.35))
            cur_b = int(math.floor(self.total_budget * 0.10))
            # We do not include generation in message budgets; keep a small buffer for safety.
            # Effective input = sys+ctx+cur; buffer absorbs rounding.
            used = sys_b + ctx_b + cur_b
            buf_b = max(0, self.total_budget - used)
            budgets = {
                "system": sys_b,
                "contextual_memory": ctx_b,
                "current": cur_b,
                "buffer": buf_b,
            }
        else:
            # Base allocations guided by activations + floors in classic mode
            budgets = {
                "system": self.floors["system"],
                "facts": self.floors["facts"],
                "snippets": 0,
                "dth": 0,
                "recent": self.floors["recent"],
                "current": self.floors["current"],
                "buffer": self.floors["buffer"],
            }

        if not self._unified:
            remaining = self.total_budget - sum(budgets.values())
            if remaining < 0:
                # Reduce recent first, deterministically
                over = -remaining
                reduce_recent = min(over, budgets["recent"] // 2)
                budgets["recent"] -= reduce_recent
                remaining += reduce_recent
                if remaining < 0:
                    # Reduce facts next
                    over2 = -remaining
                    reduce_facts = min(over2, budgets["facts"] // 2)
                    budgets["facts"] -= reduce_facts
                    remaining += reduce_facts

            # Distribute remaining proportionally based on tier weights
            # Map tiers to budget slots
            slots = {
                "core": ["system"],
                "dynamic": ["facts", "recent", "snippets"],
                "retrieval": ["dth"],
                # tools does not map to a direct message budget here; hint only
            }

            weights = {
                "core": activ["core"],
                "dynamic": activ["dynamic"],
                "retrieval": activ["retrieval"],
            }

            total_w = sum(weights.values()) or 1.0
            for tier, w in weights.items():
                share = int(math.floor(remaining * (w / total_w)))
                # Split the share evenly across the tier's slots respecting caps
                tier_slots = slots[tier]
                per = max(0, share // len(tier_slots))
                for s in tier_slots:
                    cap = self.caps.get(s, 10 ** 9)
                    budgets[s] = min(cap, budgets[s] + per)

            # Minor deterministic rounding pass to use leftover remainder, prioritizing dynamic → retrieval → core
            used = sum(budgets.values())
            remainder = self.total_budget - used
            order = ["facts", "recent", "snippets", "dth", "system"]
            i = 0
            while remainder > 0 and i < len(order):
                s = order[i]
                cap = self.caps.get(s, 10 ** 9)
                if budgets[s] < cap:
                    budgets[s] += 1
                    remainder -= 1
                else:
                    i += 1

        # Canonical inputs for debugging / verification
        canonical = inputs

        plan = ContextPlan(
            tier_activations=activ,
            token_budgets=budgets,
            token_budget_total=self.total_budget,
            notes={
                "features": feats,
                "counts": {
                    "facts": len(facts),
                    "recents": len(recents),
                    "dth": len(dth_texts or []),
                },
                "activations": activ,
                "canonical_inputs": canonical,
            },
        )

        # Reproducibility hash over canonical inputs + plan outputs
        outputs = {
            "activ": plan.tier_activations,
            "budgets": plan.token_budgets,
            "total": plan.token_budget_total,
        }
        repro_hash = _sha256({"inputs": inputs, "outputs": outputs})

        metrics = PlanMetrics(
            reproducibility_hash=repro_hash,
            total_tokens_budgeted=sum(plan.token_budgets.values()),
            tier_activations=plan.tier_activations,
            token_budgets=plan.token_budgets,
        )

        return plan, metrics
