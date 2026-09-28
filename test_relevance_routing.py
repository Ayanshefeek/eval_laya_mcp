"""
test_relevance_routing.py

Phase 1 follow-up #2: relevance and routing weren't negated in the original
single-example test (test_judge_questions.py), so the fix that rescued
hallucination/accuracy doesn't obviously apply here. But both showed a weak
or wrong signal on their "good" case (relevant: 0.568, needs_agent_call:
0.225) from a sample size of one -- not enough to tell a real weakness from
noise. This runs 5 pairs per dimension, each against the original wording AND
one alternate phrasing, using the same pass/fail-by-direction scoring as
test_wording_variants.py.
"""

import logging

import laya

logging.basicConfig(level=logging.WARNING)


def load_judge_model():
    print("Loading Laya typed-decisions checkpoint...")
    return laya.load("convaiinnovations/laya", subfolder="typed-decisions")


# --- Test cases ---

RELEVANCE_CASES = [
    {
        "query": "What is the capital of France?",
        "on_topic": "The capital of France is Paris.",
        "off_topic": "France has a population of about 68 million people.",
    },
    {
        "query": "How do I reset my password?",
        "on_topic": "Go to Settings > Security > Reset Password, then check your email for a confirmation link.",
        "off_topic": "Our support hours are 9 AM to 6 PM on weekdays.",
    },
    {
        "query": "What's the weather like in Mumbai tomorrow?",
        "on_topic": "Tomorrow in Mumbai will be sunny with a high of 34C.",
        "off_topic": "Mumbai is the financial capital of India.",
    },
    {
        "query": "Can I get a refund for my order?",
        "on_topic": "Yes, refunds are available within 30 days of purchase.",
        "off_topic": "Our return policy was last updated in March.",
    },
    {
        "query": "What time does the store close?",
        "on_topic": "The store closes at 9 PM every day.",
        "off_topic": "We have over 50 stores across the country.",
    },
]

ROUTING_CASES = [
    {"needs": "What's the current status of order #48213?", "direct": "What's 15% of 200?"},
    {"needs": "Can you check my account balance?", "direct": "What's the plural of 'cactus'?"},
    {"needs": "Search for recent news about the stock market crash.", "direct": "Translate 'good morning' to French."},
    {"needs": "Look up the tracking number for my package.", "direct": "What's the boiling point of water in Celsius?"},
    {"needs": "Pull up my last three invoices.", "direct": "Summarize the plot of Romeo and Juliet in one sentence."},
]

RELEVANCE_VARIANTS = {
    "original": (
        {"relevant": {
            "type": "noul",
            "instructions": "Does `response` directly answer the question asked in `query`?",
        }},
        "relevant",
    ),
    "specific-info": (
        {"answers_query": {
            "type": "noul",
            "instructions": "Does `response` provide the specific information that `query` is asking for?",
        }},
        "answers_query",
    ),
}

ROUTING_VARIANTS = {
    "original": (
        {"needs_agent_call": {
            "type": "noul",
            "instructions": "Does answering `request` require calling an external tool, agent, search, or private data lookup, rather than being answerable directly?",
        }},
        "needs_agent_call",
    ),
    "simplified": (
        {"needs_lookup": {
            "type": "noul",
            "instructions": "Does `request` ask for information that must be looked up from an external system, account, or database, rather than something already known?",
        }},
        "needs_lookup",
    ),
}


def evaluate_relevance(agent, variant_name: str, questions: dict, answer_key: str) -> None:
    correct = 0
    total = 0
    print(f"\n=== RELEVANCE :: {variant_name} ===")
    for case in RELEVANCE_CASES:
        for label, resp_key, is_good in [("ON-TOPIC ", "on_topic", True), ("OFF-TOPIC", "off_topic", False)]:
            state = {"query": case["query"], "response": case[resp_key]}
            result = laya.decide(agent, state, questions=questions, return_details=True)
            v = result.answers[answer_key]["noul"]
            ok = (v > 0.5) if is_good else (v <= 0.5)
            correct += int(ok)
            total += 1
            print(f"  [{'OK' if ok else 'WRONG'}] {label}: {v:.3f}")
    print(f"  --> {correct}/{total} correct ({100 * correct / total:.0f}%)")


def evaluate_routing(agent, variant_name: str, questions: dict, answer_key: str) -> None:
    correct = 0
    total = 0
    print(f"\n=== ROUTING :: {variant_name} ===")
    for case in ROUTING_CASES:
        for label, req_key, is_good in [("NEEDS-CALL", "needs", True), ("DIRECT    ", "direct", False)]:
            state = {"request": case[req_key]}
            result = laya.decide(agent, state, questions=questions, return_details=True)
            v = result.answers[answer_key]["noul"]
            ok = (v > 0.5) if is_good else (v <= 0.5)
            correct += int(ok)
            total += 1
            print(f"  [{'OK' if ok else 'WRONG'}] {label}: {v:.3f}")
    print(f"  --> {correct}/{total} correct ({100 * correct / total:.0f}%)")


def main():
    agent = load_judge_model()

    for variant_name, (questions, answer_key) in RELEVANCE_VARIANTS.items():
        evaluate_relevance(agent, variant_name, questions, answer_key)

    for variant_name, (questions, answer_key) in ROUTING_VARIANTS.items():
        evaluate_routing(agent, variant_name, questions, answer_key)


if __name__ == "__main__":
    main()