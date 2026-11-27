"""Reproduce the archived first WP update with three merge precisions."""
import hashlib

from scripts.v8.dev_workers import configs, data_paths, isolate
from scripts.v8.rebuild_replay import read, write


def probe_probabilities(model, prompts):
    import math

    results = []
    for prompt in prompts:
        probabilities, mass = model.probabilities(prompt)
        if not all(math.isfinite(x) for x in [*probabilities, mass]):
            raise FloatingPointError("Nonfinite precision probe")
        results.append({"probabilities": probabilities, "choice_mass": mass,
                        "label": "ABCD"[max(range(4), key=probabilities.__getitem__)]})
    return results


def compare_probes(before, after):
    pairs = list(zip(before, after, strict=True))
    return {
        "n": len(pairs),
        "max_probability_difference": max(
            abs(a - b) for left, right in pairs
            for a, b in zip(left["probabilities"], right["probabilities"], strict=True)
        ),
        "label_changes": sum(left["label"] != right["label"] for left, right in pairs),
    }


def compare_step(out, cfg, precision, independent):
    root = out / "oracle_input"
    input_files = (root / "m0.pt", root / "first_request.json", root / "archived_wp")
    abi = isolate(out, cfg, input_files)

    import torch
    from peft import PeftModel
    from krmm.model import Omni
    from krmm.v3.damage import restore_weights, snapshot_weights, state_sha256
    from krmm.v3.knowledge_registry import question_prompt, train_retain_prompts
    from krmm.v8.constrained_writer import active_layers, collect_basis, delta, norm, relative_residual
    from krmm.v8.dev_eval import knowledge
    from krmm.v8.merge_precision import promote_edited_layers

    anchor_path, request_path, adapter_path = input_files
    old, _, repair = configs()
    request = read(request_path)
    catalog = read(data_paths(cfg)[0])
    row, = [entry for entry in catalog if entry["unit_id"] == request["unit_id"]]
    if request["unit_id"] != precision["first_unit_id"]:
        raise RuntimeError("Unexpected Oracle request")
    anchor = torch.load(anchor_path, map_location="cpu", weights_only=True)
    if state_sha256(anchor) != precision["m0_sha256"]:
        raise RuntimeError("Archived M0 checksum differs")
    torch.set_float32_matmul_precision("highest")
    model = Omni(old["model_path"])
    restore_weights(model, anchor)
    if state_sha256(snapshot_weights(model, repair)) != precision["m0_sha256"]:
        raise RuntimeError("Pilot initial edited weights differ from M0")
    retains = train_retain_prompts(include_multi_template=True)
    peers = [item["public_protection"] for item in catalog if item["domain"] == row["domain"] and item["unit_id"] != row["unit_id"]][:2]
    protection = [*retains[:4], row["public_protection"], *peers]
    if [hashlib.sha256(prompt.encode()).hexdigest() for prompt in protection] != precision["public_prompt_sha256"]:
        raise RuntimeError("First-step protection prompts differ from archive")
    basis = collect_basis(model, anchor, protection, cfg["rank_cap"])
    if set(basis) != set(anchor):
        raise RuntimeError("First-step basis is incomplete")
    unit = knowledge(row)
    prompts = [question_prompt(unit, variant=4, permutation=tuple(order))[0] for order in ("ABCD", "BCDA", "CDAB", "DABC")]
    prompts += [question_prompt(unit, variant=5)[0], *protection]

    def attach():
        model.model = PeftModel.from_pretrained(model.model, str(adapter_path), is_trainable=False)
        model.model.eval()

    def merge():
        model.model = model.model.merge_and_unload(safe_merge=True)
        model.model.eval()
        return snapshot_weights(model, repair)

    with torch.inference_mode():
        base_bf16 = probe_probabilities(model, prompts)
        attach()
        updates, pre = {}, {}
        for key, layer in active_layers(model.model, anchor):
            matrix = delta(layer)
            base = basis[key]
            adapter_a, adapter_b = layer.lora_A["default"], layer.lora_B["default"]
            direct = adapter_b(adapter_a(base.T.to(adapter_a.weight.dtype))) * layer.scaling["default"]
            value = norm(matrix)
            if not torch.isfinite(matrix).all() or value <= cfg["min_layer_delta"]:
                raise RuntimeError(f"Expected a nonzero finite first-step update in {key}")
            pre[key] = {"rho": relative_residual(matrix, base), "adapter_forward_rho": norm(direct) / value,
                        "norm": value, "basis_rank": base.shape[1]}
            updates[key] = matrix.cpu()
        if set(updates) != set(anchor):
            raise RuntimeError("Archived adapter does not cover all edited layers")
        unmerged = probe_probabilities(model, prompts)
        bf16 = merge()
        bf16_probes = probe_probabilities(model, prompts)
        bf16_sha = state_sha256(bf16)
        if bf16_sha != precision["archived_bf16_candidate_sha256"]:
            raise RuntimeError("BF16 merge did not reproduce the archived tentative state")

        restore_weights(model, anchor)
        real_anchor = promote_edited_layers(model.model, anchor)
        baseline_fp32 = snapshot_weights(model, repair)
        if state_sha256(baseline_fp32) != state_sha256(real_anchor):
            raise RuntimeError("Promoted weights differ numerically from original M0")
        base_fp32 = probe_probabilities(model, prompts)
        attach()
        fp32 = merge()
        fp32_probes = probe_probabilities(model, prompts)

        result = {
            "m0_sha256": precision["m0_sha256"], "target_unit_id": request["unit_id"],
            "landlock_abi": abi, "promoted_m0_sha256": state_sha256(baseline_fp32),
            "unmerged": {"layers": pre, "probes": unmerged},
            "base_fp32_vs_bf16": compare_probes(base_bf16, base_fp32),
            "original_m0_probes": base_bf16, "promoted_m0_probes": base_fp32,
        }
        for label, state, predictions in (("FP32", fp32, fp32_probes), ("BF16", bf16, bf16_probes)):
            layers = {}
            for key, actual in state.items():
                if actual.dtype != getattr(torch, "float32" if label == "FP32" else "bfloat16"):
                    raise RuntimeError(f"Actual storage dtype differs in {key}")
                measured = actual.to(basis[key].device).float() - anchor[key].to(basis[key].device).float()
                ideal = updates[key].to(measured.device)
                measured_norm = norm(measured)
                if not torch.isfinite(measured).all() or measured_norm <= cfg["min_layer_delta"]:
                    raise RuntimeError(f"Expected nonzero finite merged update in {key}")
                layers[key] = {
                    "rho": relative_residual(measured, basis[key]), "norm": measured_norm,
                    "relative_delta_error": norm(measured - ideal) / pre[key]["norm"],
                }
            result[label] = {
                "state_sha256": state_sha256(state), "layers": layers, "probes": predictions,
                "versus_unmerged": compare_probes(unmerged, predictions),
            }
    name = "pilot_fresh" if independent else "pilot_compare"
    write(out / "writer" / f"{name}.json", result)
    print(name, {mode: max(v["rho"] for v in result[mode]["layers"].values()) for mode in ("unmerged", "FP32", "BF16")}, flush=True)


def gate(primary, fresh, eps, tolerance):
    unmerged_rho = max(layer["rho"] for layer in primary["unmerged"]["layers"].values())
    operator_rho = max(layer["adapter_forward_rho"] for layer in primary["unmerged"]["layers"].values())
    checks = {}
    for name in ("BF16", "FP32"):
        left, right = primary[name], fresh[name]
        verification = compare_probes(left["probes"], right["probes"])
        maximum = max(max(layer["rho"] for layer in state["layers"].values()) for state in (left, right))
        valid = (
            maximum <= eps and left["state_sha256"] == right["state_sha256"]
            and verification["max_probability_difference"] <= tolerance and verification["label_changes"] == 0
            and unmerged_rho <= eps and operator_rho <= eps
        )
        checks[name] = {"passed": valid, "max_residual": maximum, "independent_probe_comparison": verification,
                        "independent_state_equal": left["state_sha256"] == right["state_sha256"]}
    eligible = next((name for name in ("BF16", "FP32") if checks[name]["passed"]), None)
    return {"eligible_merge_dtype": eligible, "routes": checks, "unmerged_rho": unmerged_rho,
            "unmerged_operator_rho": operator_rho, "threshold": eps,
            "status": "PASS" if eligible else "FAIL"}
