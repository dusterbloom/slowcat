"""
Unit tests for the deterministic ContextField planner (MVP).

Focus:
- Determinism: same inputs -> same reproducibility hash and budgets
- Budgeting: sum of budgets <= total; non-negative per-slot
- Tier activations: questions vs commands influence core/tools/retrieval
"""

from context.context_field import ContextField, FactMini


def _sample_facts():
    return [
        FactMini(subject="user", predicate="pet", value="Potola", fidelity=3, last_seen=1000.0),
        FactMini(subject="user", predicate="location", value="San Francisco", fidelity=2, last_seen=900.0),
    ]


def _sample_recents():
    return [
        ("My dog name is Potola", "That's a nice name!"),
        ("We live in San Francisco", "Great city for walks."),
    ]


def test_determinism_same_inputs_same_hash():
    field = ContextField(total_budget=4096)
    facts = _sample_facts()
    recents = _sample_recents()
    q = "What's my dog's name?"

    plan1, metrics1 = field.create_plan(q, facts, recents, dth_texts=["She is a golden retriever."])
    plan2, metrics2 = field.create_plan(q, facts, recents, dth_texts=["She is a golden retriever."])

    assert metrics1.reproducibility_hash == metrics2.reproducibility_hash
    assert plan1.token_budgets == plan2.token_budgets
    assert plan1.tier_activations == plan2.tier_activations


def test_budget_sum_and_non_negative():
    field = ContextField(total_budget=4096)
    plan, metrics = field.create_plan("hello", _sample_facts(), _sample_recents(), dth_texts=None)

    total = sum(plan.token_budgets.values())
    assert total <= field.total_budget
    for k, v in plan.token_budgets.items():
        assert v >= 0


def test_tier_activation_question_vs_command():
    field = ContextField(total_budget=4096)
    facts = _sample_facts()
    recents = _sample_recents()

    # Question should weight core/retrieval higher
    plan_q, _ = field.create_plan("What is this?", facts, recents)
    # Command-like phrase should weight tools higher
    plan_c, _ = field.create_plan("please play music", facts, recents)

    assert plan_q.tier_activations["core"] > 0
    assert plan_q.tier_activations["retrieval"] >= plan_c.tier_activations["retrieval"]
    assert plan_c.tier_activations["tools"] > plan_q.tier_activations["tools"]

