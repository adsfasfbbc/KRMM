"""Independent replay observer and frozen DEV-to-TEST decision."""
import math

from krmm.v8.dev_eval import accuracy, control_specs, controls, semantic_metrics, targets
from krmm.v7.transition_audit import compute_step_transition_metrics
from scripts.v8.dev_workers import LEGACY, PRIVATE, PUBLIC, configs
from scripts.v8.rebuild_replay import read, write


def observe(out, cfg):
    import torch
    from peft import PeftModel
    from krmm.model import Omni
    from krmm.v3.damage import restore_weights, snapshot_weights, state_sha256

    old, _, repair = configs()
    hidden = out / "private"
    workspace = out / "writer"
    oracle = read(out / "oracle_input" / "requests.json")
    public = read(PUBLIC)
    private = {row["unit_id"]: row for row in read(PRIVATE)}
    truth = read(hidden / "truth.json")
    degraded = {row["unit_id"] for row in truth if row["is_degraded_target"]}
    m0_targets = read(hidden / "damaged_targets.json")
    m0_controls = read(hidden / "m0_controls.json")
    healthy = {key for key, value in m0_targets.items() if key not in degraded and value["accuracy"] >= 0.999}
    specs = control_specs(public, private)
    model = Omni(old["model_path"])
    anchor = torch.load(out / "oracle_input" / "m0.pt", map_location="cpu", weights_only=True)
    m0_sha = oracle["state_sha256"]
    if state_sha256(anchor) != m0_sha:
        raise RuntimeError("Observer M0 digest mismatch")
    scores = {}

    for name in cfg["branches"]:
        restore_weights(model, anchor)
        if state_sha256(snapshot_weights(model, repair)) != m0_sha:
            raise RuntimeError("Observer could not reset M0")
        branch = read(workspace / name / "trajectory.json")
        committed = branch["committed"]
        history = [set()]
        trajectory_targets = [m0_targets]
        trajectory_controls = [m0_controls]
        for index, row in enumerate(committed, 1):
            if state_sha256(snapshot_weights(model, repair)) != row["pre_sha256"]:
                raise RuntimeError("Replay pre-step state differs from writer")
            model.model = PeftModel.from_pretrained(model.model, str(workspace / row["adapter"]), is_trainable=False)
            model.model = model.model.merge_and_unload(safe_merge=True)
            model.model.eval()
            if state_sha256(snapshot_weights(model, repair)) != row["candidate_sha256"]:
                raise RuntimeError("Replay post-step state differs from writer")
            history.append(history[-1] | {row["unit_id"]})
            trajectory_targets.append(targets(model, public, private))
            trajectory_controls.append(controls(model, specs, index == len(committed)))

        error = 0.0
        for probe in branch["public_probes"]:
            predicted, _ = model.probabilities(probe["prompt"])
            error = max(error, max(abs(a - b) for a, b in zip(predicted, probe["probabilities"], strict=True)))
        if not math.isfinite(error) or error > cfg["reload_tolerance"] or len(branch["public_probes"]) < 4:
            raise RuntimeError("Public probe fresh replay tolerance failed")
        if state_sha256(snapshot_weights(model, repair)) != branch["final_sha256"]:
            raise RuntimeError("Final state SHA mismatch")
        metrics = compute_step_transition_metrics(
            degraded, history, [accuracy(rows) for rows in trajectory_targets], trajectory_controls, fresh_reload_passed=True,
        )
        final = metrics["trajectory"][-1]
        same_entity = len(final["same_entity_regressions"])
        healthy_harm = sum(trajectory_targets[-1][unit]["accuracy"] < 0.999 for unit in healthy)
        semantics = semantic_metrics(m0_controls, trajectory_controls[-1])
        scores[name] = {
            "attempted": len(branch["attempts"]), "committed": len(committed),
            "rejected_at": [attempt["step"] for attempt in branch["attempts"] if not attempt["committed"]],
            "active_restored": final["active_count"], "incidental_restored": final["incidental_count"],
            "recovered": final["restored_count"], "R_final": final["R_t"], "same_entity_harm": same_entity,
            "H_new": metrics["H_new_final"], "H_ever": metrics["H_ever_total"], "max_forget": metrics["max_Forget_t"],
            "healthy_harm": healthy_harm, "strict_yield": metrics["Y_strict"],
            "fresh_pass": True, "max_public_probe_difference": error, "semantics": semantics,
            "max_attempted_analytic_residual": max((row["analytic_residual"] for row in branch["attempts"]), default=0),
            "max_attempted_bf16_residual": max((row["merged_residual"] for row in branch["attempts"]), default=0),
            "max_attempted_norm_error": max((row["max_norm_error"] for row in branch["attempts"]), default=0),
            "forward_tokens": sum(row["train_tokens"] for row in branch["attempts"]),
        }
        write(hidden / f"{name}_replay.json", {"targets": trajectory_targets, "controls": trajectory_controls, "metrics": metrics})
        print("DEV observer", name, scores[name], flush=True)
    result = decide(scores, cfg)
    write(out / "dev_summary.json", {"g0": read(hidden / "g0.json"), "branches": scores, **result})


def decide(scores, cfg):
    full = cfg["oracle_targets"]
    simple = [key for key in ("W0", "W1", "WN") if scores[key]["committed"] == full]
    baseline = min(simple, key=lambda key: (
        scores[key]["same_entity_harm"] + scores[key]["healthy_harm"],
        scores[key]["H_ever"], -scores[key]["active_restored"], -scores[key]["R_final"], key,
    ))
    wp = scores["WP"]
    base = scores[baseline]
    checks = {
        "full_nonzero_projected_sequence": wp["committed"] == full,
        "active_recovery_minimum": wp["active_restored"] >= cfg["min_active"],
        "recovery_within_baseline_tolerance": wp["R_final"] >= base["R_final"] - cfg["max_recovery_drop"],
        "same_entity_and_healthy_harm_zero": wp["same_entity_harm"] == wp["healthy_harm"] == 0,
        "fresh_strict_yield": wp["fresh_pass"] and wp["strict_yield"] > 0,
        "semantic_coverage": wp["semantics"]["coverage"] >= cfg["min_semantic_coverage"],
        "norm_and_random_complete": scores["WN"]["committed"] == scores["WR"]["committed"] == full,
        "meaningful_directional_advantage": all(
            wp["same_entity_harm"] < scores[key]["same_entity_harm"]
            and wp["active_restored"] >= scores[key]["active_restored"] - 1 for key in (baseline, "WN", "WR")
        ),
    }
    approved = all(checks.values())
    return {"strongest_simple": baseline, "pilot_checks": checks,
            "decision": "PREPARE_FROZEN_TEST_QUEUE" if approved else "DO_NOT_ENTER_TEST",
            "ow_pilot": "PASS" if approved else "FAIL", "ow_confirm": "NOT_RUN"}
