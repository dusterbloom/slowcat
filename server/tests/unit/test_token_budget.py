#!/usr/bin/env python3
"""
Token budget invariants for SmartContextManager.
"""

import importlib


def test_token_budget_invariants():
    scm = importlib.import_module('processors.smart_context_manager')
    TokenBudget = getattr(scm, 'TokenBudget')

    b = TokenBudget()
    # The context total should be sum of system + memory + input
    assert b.total == b.system_prompt + b.contextual_memory + b.current_input
    # Total with generation includes generation workspace
    assert b.total_with_generation == b.total + b.generation_workspace

