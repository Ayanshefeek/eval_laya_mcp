"""
test_wording_variants.py

Phase 1 follow-up: hallucination and accuracy came back backwards (with high
confidence, on both good/bad cases) in test_judge_questions.py, while
injection and routing -- questions adapted closely from laya's own built-in
presets -- worked well. Two things need ruling out before concluding Laya
can't do hallucination/accuracy at all:

  1. Wording: the original hallucination/accuracy questions used a NEGATED
     proposition ("does response state a claim NOT supported by context?").
     This tests a POSITIVE-framed alternative ("is response fully grounded
     in context?") against the same cases, in case negation itself confuses
     the checkpoint.
  2. Sample size: one pair per dimension isn't enough to separate a real
     reversal from noise. This runs 5 pairs per dimension instead of 1.

For each dimension we score BOTH phrasings against the same 5 good/bad pairs,
then report how often each phrasing got the direction right (good case above
0.5, bad case below 0.5, in the polarity that phrasing implies). This turns
"it looked backwards" into a number we can act on.
"""

import logging

import laya

logging.basicConfig(level=logging.WARNING)  # keep noise down, we print our own summary
logger = logging.getLogger(__name__)


def load_judge_model():
    print("Loading Laya typed-decisions checkpoint...")
    return laya.load("convaiinnovations/laya", subfolder="typed-decisions")


# --- Test cases: 5 good/bad pairs per dimension ---

HALLUCINATION_CASES = [
    {
        "context": "The company's Q3 revenue was $4.2 million, up 12% year over year.",
        "grounded": "Q3 revenue was $4.2 million, up 12% from last year.",
        "hallucinated": "Q3 revenue was $4.2 million, and the CEO announced plans to double headcount next year.",
    },
    {
        "context": "Tomorrow's forecast for Mumbai is sunny with a high of 34C and no chance of rain.",
        "grounded": "Expect sunny skies tomorrow in Mumbai with a high around 34C.",
        "hallucinated": "Tomorrow will be sunny in Mumbai with a high of 34C, and there's a 70% chance of rain in the evening.",
    },
    {
        "context": "The laptop has 16GB RAM, a 512GB SSD, and a 14-inch display.",
        "grounded": "This laptop comes with 16GB of RAM and a 512GB SSD.",
        "hallucinated": "This laptop comes with 16GB of RAM, a 512GB SSD, and a dedicated GPU for 4K gaming.",
    },
    {
        "context": "The team agreed to launch the new feature on October 15th, pending QA sign-off.",
        "grounded": "Launch is planned for October 15th, assuming QA approves it.",
        "hallucinated": "Launch is planned for October 15th, and marketing has already started running ads for it.",
    },
    {
        "context": "Order #48213 shipped on September 25th and is expected to arrive by September 30th.",
        "grounded": "Your order shipped on September 25th and should arrive by September 30th.",
        "hallucinated": "Your order shipped on September 25th and includes a free gift card for your next purchase.",
    },
]

ACCURACY_CASES = [
    {
        "gold_answer": "The refund will be processed within 5-7 business days.",
        "accurate": "You'll see the refund in your account within 5 to 7 business days.",
        "inaccurate": "Refunds are processed instantly, you'll see it within the hour.",
    },
    {
        "gold_answer": "Our office is open Monday through Friday, 9 AM to 6 PM.",
        "accurate": "We're open weekdays from 9 in the morning until 6 in the evening.",
        "inaccurate": "We're open every day, including weekends, from 9 AM to 9 PM.",
    },
    {
        "gold_answer": "The subscription costs $12 per month, billed annually.",
        "accurate": "It's $12 a month, charged as one annual payment.",
        "inaccurate": "The subscription is a one-time payment of $12 with no recurring charges.",
    },
    {
        "gold_answer": "Password resets require a verified email address on file.",
        "accurate": "To reset your password you need a verified email on your account.",
        "inaccurate": "You can reset your password using just your phone number, no email needed.",
    },
    {
        "gold_answer": "The warranty covers manufacturing defects for 2 years from purchase.",
        "accurate": "Manufacturing defects are covered for two years after you buy it.",
        "inaccurate": "The warranty covers any damage, including accidental drops, for 5 years.",
    },
]

# Each variant: (question_dict, answer_key, "high_on_good" | "high_on_bad")
# "high_on_good" = a positively-framed question where the GOOD case should score > 0.5
# "high_on_bad"  = a negatively-framed question where the BAD case should score > 0.5

HALLUCINATION_VARIANTS = {
    "negated (original)": (
        {"hallucinated": {
            "type": "noul",
            "instructions": "Does `response` state any claim or fact that is NOT supported by `context`?",
        }},
        "hallucinated",
        "high_on_bad",
    ),
    "positive (grounded)": (
        {"grounded": {
            "type": "noul",
            "instructions": "Is every factual claim in `response` fully and directly supported by `context`?",
        }},
        "grounded",
        "high_on_good",
    ),
}

ACCURACY_VARIANTS = {
    "negated (contradicts)": (
        {"contradicts": {
            "type": "noul",
            "instructions": "Does `response` contradict `gold_answer` or add facts that are not in `gold_answer`?",
        }},
        "contradicts",
        "high_on_bad",
    ),
    "positive (matches)": (
        {"matches": {
            "type": "noul",
            "instructions": "Does `response` correctly match the factual content stated in `gold_answer`?",
        }},
        "matches",
        "high_on_good",
    ),
}


def score_direction(noul_value: float, expect: str, is_good_case: bool) -> bool:
    """Return True if this single score is on the correct side of 0.5."""
    above = noul_value > 0.5
    if expect == "high_on_good":
        return above if is_good_case else not above
    else:  # high_on_bad
        return (not above) if is_good_case else above


def evaluate_variant(agent, dimension: str, variant_name: str, questions: dict,
                      answer_key: str, expect: str, cases: list, good_key: str, bad_key: str,
                      context_key: str) -> None:
    correct = 0
    total = 0
    print(f"\n=== {dimension} :: {variant_name} ===")
    for case in cases:
        for label, text_key, is_good in [("GOOD", good_key, True), ("BAD", bad_key, False)]:
            state = {context_key: case[context_key], "response": case[text_key]}
            result = laya.decide(agent, state, questions=questions, return_details=True)
            noul_value = result.answers[answer_key]["noul"]
            ok = score_direction(noul_value, expect, is_good)
            correct += int(ok)
            total += 1
            mark = "OK" if ok else "WRONG"
            print(f"  [{mark}] {label:4s} case: {noul_value:.3f}")
    print(f"  --> {correct}/{total} correct ({100 * correct / total:.0f}%)")


def main():
    agent = load_judge_model()

    for variant_name, (questions, answer_key, expect) in HALLUCINATION_VARIANTS.items():
        evaluate_variant(
            agent, "HALLUCINATION", variant_name, questions, answer_key, expect,
            HALLUCINATION_CASES, good_key="grounded", bad_key="hallucinated", context_key="context",
        )

    for variant_name, (questions, answer_key, expect) in ACCURACY_VARIANTS.items():
        evaluate_variant(
            agent, "ACCURACY", variant_name, questions, answer_key, expect,
            ACCURACY_CASES, good_key="accurate", bad_key="inaccurate", context_key="gold_answer",
        )


if __name__ == "__main__":
    main()