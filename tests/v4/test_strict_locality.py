"""Verify 5-gate strict locality & persistence report catches same-entity drops and reload failures."""
from __future__ import annotations

from krmm.v4.locality_audit import audit_side_effects


def test_strict_locality_requires_all_five_gates():
    good = audit_side_effects("u1", 1.0, 1.0, 1.0, 0.95, 0.94, 0.85, 0.84, fresh_process_reload_passed=True)
    assert good.strict_parametric_recovery is True

    # Same-entity invariant dropped (like timpani / crystal wine glass) -> MUST FAIL strict recovery
    bad_same_ent = audit_side_effects("u2", 1.0, 1.0, 0.0, 0.95, 0.95, 0.85, 0.85, fresh_process_reload_passed=True)
    assert bad_same_ent.same_entity_preserved is False
    assert bad_same_ent.strict_parametric_recovery is False

    # Reload failed -> MUST FAIL strict recovery
    bad_reload = audit_side_effects("u3", 1.0, 1.0, 1.0, 0.95, 0.95, 0.85, 0.85, fresh_process_reload_passed=False)
    assert bad_reload.fresh_process_reload_passed is False
    assert bad_reload.strict_parametric_recovery is False
