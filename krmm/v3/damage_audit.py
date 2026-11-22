"""Independent retention and locality auditing across four control categories (M1.3)."""
from __future__ import annotations

from typing import Dict, List, Sequence
import numpy as np
import torch


def measure_controls(omni, controls: Sequence[Dict[str, str]], batch_size: int = 8) -> Dict[str, object]:
    """Evaluate model logits/predictions on an independent control suite."""
    probs: List[List[float]] = []
    top_tokens: List[int] = []
    masses: List[float] = []
    ids = torch.tensor(
        [omni.tokenizer.encode(c, add_special_tokens=False)[0] for c in "ABCD"],
        device="cuda",
    )
    with torch.inference_mode():
        for i in range(0, len(controls), batch_size):
            chunk = controls[i : i + batch_size]
            batch = omni.encode_many([c["question"] for c in chunk])
            logits = omni.model(**batch).logits[:, -1].float()
            p = logits[:, ids].softmax(-1).cpu().tolist()
            top = logits.argmax(-1).cpu().tolist()
            mass = (logits[:, ids].logsumexp(-1) - logits.logsumexp(-1)).exp().cpu().tolist()
            probs.extend(p)
            top_tokens.extend(top)
            masses.extend(mass)
    preds = ["ABCD"[int(np.argmax(p))] for p in probs]
    return {
        "probabilities": probs,
        "top_token": top_tokens,
        "choice_mass": masses,
        "prediction": preds,
    }


def _kl_div(p: Sequence[float], q: Sequence[float]) -> float:
    pa = np.clip(np.asarray(p, dtype=np.float64), 1e-8, 1.0)
    qa = np.clip(np.asarray(q, dtype=np.float64), 1e-8, 1.0)
    pa = pa / pa.sum()
    qa = qa / qa.sum()
    return float(np.sum(pa * np.log(pa / qa)))


def compare_locality(
    baseline: Dict[str, object],
    after: Dict[str, object],
    controls: Sequence[Dict[str, str]],
    target_entity: str,
    min_token_agreement: float = 0.90,
    max_accuracy_drop: float = 0.05,
) -> Dict[str, object]:
    """Compare post-update control behavior against baseline across all four M1.3 categories."""
    categories = [
        "all",
        "same_entity_other_relation",
        "other_entity_unseen_template",
        "near_neighbor_invariant",
        "general_commonsense",
    ]
    out: Dict[str, object] = {}
    for kind in categories:
        if kind == "all":
            indices = list(range(len(controls)))
        else:
            indices = [i for i, c in enumerate(controls) if c.get("kind") == kind]
        if not indices:
            out[kind] = {
                "n": 0,
                "token_agreement": 1.0,
                "baseline_accuracy": 1.0,
                "after_accuracy": 1.0,
                "accuracy_drop": 0.0,
                "mean_kl_shift": 0.0,
            }
            continue
        agree = float(np.mean([baseline["top_token"][i] == after["top_token"][i] for i in indices]))
        before_acc = float(np.mean([baseline["prediction"][i] == controls[i]["gold"] for i in indices]))
        after_acc = float(np.mean([after["prediction"][i] == controls[i]["gold"] for i in indices]))
        kl_mean = float(
            np.mean([_kl_div(baseline["probabilities"][i], after["probabilities"][i]) for i in indices])
        )
        out[kind] = {
            "n": len(indices),
            "token_agreement": agree,
            "baseline_accuracy": before_acc,
            "after_accuracy": after_acc,
            "accuracy_drop": before_acc - after_acc,
            "mean_kl_shift": kl_mean,
        }

    # Check overall token agreement and per-category accuracy drops
    cat_drops_ok = all(
        out[k]["accuracy_drop"] <= max_accuracy_drop + 1e-12
        for k in ("same_entity_other_relation", "other_entity_unseen_template", "near_neighbor_invariant", "general_commonsense")
        if out[k]["n"] > 0
    )
    # Note: for same_entity_other_relation (often n=1 per entity), a single flip is 1.0 drop; check token_agreement & drop!
    # Wait: if n=1 and baseline was correct, any drop is 1.0 > 0.05, which properly enforces preserving the same entity's other relations!
    out["independent_locality_pass"] = bool(
        out["all"]["token_agreement"] >= min_token_agreement - 1e-12 and cat_drops_ok
    )
    out["target_entity"] = target_entity
    return out
