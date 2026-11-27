"""Private evaluator for a frozen DEV-V8 episode; no evaluator feedback reaches the writer."""
from collections import Counter
import random
import re

from krmm.v3.knowledge_registry import AUDIT_GENERAL_CONTROLS, DOMAIN_OPTIONS, question_prompt
from krmm.v3.mastery import HELDOUT_PERM, HELDOUT_VARIANTS, evaluate_unit_target, screen_clean_mastery
from krmm.v3.schemas import EvaluatorUnit
from krmm.v5.v5_catalog import DOMAIN_RELATION
from krmm.v6.schemas import PublicUnit
from krmm.v6.truth_audit import classify_paired_truth

LETTERS = "ABCD"
ORDERS = ("ABCD", "BCDA", "CDAB", "DABC")


def knowledge(row):
    domain = row["domain"]
    return PublicUnit(row["unit_id"], row["entity_id"], domain, DOMAIN_RELATION[domain], DOMAIN_OPTIONS[domain], row["source_group"]).to_knowledge_unit("test")


def screen(backend, public, private):
    return {
        row["unit_id"]: screen_clean_mastery(backend, EvaluatorUnit(
            knowledge(row), private[row["unit_id"]]["gold"], [], [], False,
        ))
        for row in public
    }


def targets(backend, public, private):
    return {
        row["unit_id"]: evaluate_unit_target(
            backend, knowledge(row), private[row["unit_id"]]["gold"],
            HELDOUT_VARIANTS, (HELDOUT_PERM,),
        )
        for row in public
    }


def control_specs(public, private):
    rows = []
    for entry in public:
        uid, entity = entry["unit_id"], entry["entity_id"]
        for index, fact in enumerate(private[uid]["controls"]):
            rows.append({
                **fact, "id": f"{uid}-immutable-{index + 1}", "entity_id": entity,
                "kind": "same_entity_other_relation",
                "choices": {"A": fact["answer"], "B": fact["distractor"], "C": "Neither description fits", "D": "Insufficient information"},
                "canonical_gold": "A", "order": ORDERS[(len(rows) * 3 + len(rows) // 2) % 4],
            })
        # Same target relation asked independently, scored outside locality.
        rows.append({
            "id": f"{uid}-ripple", "entity_id": entity, "kind": "expected_ripple",
            "question": f"Subject: {entity}. Select the broader category that follows from the entity's defining properties.",
            "choices": DOMAIN_OPTIONS[entry["domain"]], "canonical_gold": private[uid]["gold"], "order": ORDERS[len(rows) % 4],
        })
    for index, general in enumerate(AUDIT_GENERAL_CONTROLS):
        rows.append({**general, "id": f"general-{index:02}", "entity_id": "__general__", "kind": "general_commonsense"})
    return rows


def evaluate_choices(backend, prompt):
    probs, mass = backend.probabilities(prompt)
    return {"probabilities": probs, "mass": mass, "pred": LETTERS[max(range(4), key=probs.__getitem__)]}


def display_prompt(stem, choices, order):
    rows = [f"{key}: {choices[value]}" for key, value in zip(LETTERS, order, strict=True)]
    return "\n".join([stem, *rows, "Reply with one letter only."])


def score_open(text, expected, distractor):
    def matches(answer):
        words = re.findall(r"\w+", answer.casefold())
        pattern = r"\b" + r"\W+".join(re.escape(word) for word in words) + r"\b"
        return bool(words and re.search(pattern, text.casefold()))
    if re.search(r"\b(?:not|neither|never)\b", text.casefold()):
        return "unclassified"
    right, wrong = matches(expected), matches(distractor)
    if right == wrong:
        return "unclassified"
    return "correct" if right else "wrong"


def controls(backend, specs, semantic):
    rows = []
    for spec in specs:
        if "choices" not in spec:
            question, gold = spec["question"], spec["gold"]
        else:
            question = display_prompt(spec["question"], spec["choices"], spec["order"])
            gold = LETTERS[spec["order"].index(spec["canonical_gold"])]
        scored = evaluate_choices(backend, question)
        row = {"id": spec["id"], "entity_id": spec["entity_id"], "kind": spec["kind"],
               "question": question, "gold": gold, "correct": scored["pred"] == gold, **scored}
        if semantic and spec["kind"] == "same_entity_other_relation":
            semantic_rows = []
            for order in ORDERS:
                prompt = display_prompt(spec["question"], spec["choices"], order)
                observed = evaluate_choices(backend, prompt)
                semantic_rows.append({"order": order, **observed, "canonical_answer": order[LETTERS.index(observed["pred"])]})
            response = backend.generate(spec["question"] + " Answer in a short phrase.", max_new_tokens=12)
            row["semantic"] = {"permutations": semantic_rows, "open": response,
                               "open_score": score_open(response, spec["answer"], spec["distractor"])}
        rows.append(row)
    return rows


def anchor_truth(public, private, clean, damaged, planned):
    return [classify_paired_truth(
        row["unit_id"], row["entity_id"], row["domain"], row["source_group"], "known_real",
        row["unit_id"] in planned, row["unit_id"] in planned,
        clean[row["unit_id"]]["stably_mastered"], clean[row["unit_id"]]["screen"]["accuracy"],
        clean[row["unit_id"]]["heldout_clean_audit"]["accuracy"], damaged[row["unit_id"]]["accuracy"],
    ).to_dict() for row in public]


def accuracy(rows):
    return {key: value["accuracy"] for key, value in rows.items()}


def semantic_metrics(m0, final):
    pairs = [(left["semantic"], right["semantic"]) for left, right in zip(m0, final, strict=True) if "semantic" in left]
    if not pairs:
        return {"n": 0, "coverage": 0.0, "confirmed_harm": 0}
    covered, harmful, ambiguous = 0, 0, 0
    for before, after in pairs:
        scores = [sum(row["canonical_answer"] == "A" for row in item["permutations"]) for item in (before, after)]
        if "unclassified" in (before["open_score"], after["open_score"]):
            ambiguous += 1
        else:
            covered += 1
            harmful += scores[0] >= 3 and scores[1] <= 1 and (before["open_score"], after["open_score"]) == ("correct", "wrong")
    return {"n": len(pairs), "coverage": covered / len(pairs), "confirmed_harm": harmful, "unclassified_pairs": ambiguous}


def select_damage(public, clean, fraction, seed):
    by_domain = {}
    for entry in public:
        row = clean[entry["unit_id"]]
        if row["stably_mastered"] and row["heldout_clean_mastered"]:
            by_domain.setdefault(entry["domain"], []).append(entry)
    rng = random.Random(seed)
    planned = []
    for domain in sorted(by_domain):
        items = list(by_domain[domain])
        rng.shuffle(items)
        planned.extend(items[:max(1, round(len(items) * fraction))])
    rng.shuffle(planned)
    return planned, dict(Counter(item["domain"] for items in by_domain.values() for item in items))
