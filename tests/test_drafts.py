import pytest

from manto.drafts import AnalysisDraft, DialogueState


def test_defaults_are_not_consent_and_partial_state_roundtrips():
    state = DialogueState()
    state.draft = state.draft.patch({"target_id": "sales", "pinned_ids": []}, "turn1")
    restored = DialogueState.model_validate_json(state.model_dump_json())
    assert restored.draft.values["target_id"] == "sales"
    assert "pinned_ids" not in restored.draft.unresolved
    assert "model_size" in restored.draft.unresolved
    with pytest.raises(ValueError, match="Undiscussed"):
        restored.draft.request()


def test_atomic_patch_confirmation_and_target_dependency_invalidation():
    draft = AnalysisDraft.new().patch(
        {"target_id": "sales", "candidate_ids": ["inflation"], "model_size": 1}, "1"
    )
    proposal = draft.patch({"lag_menu": [0, 1]}, "2", proposed=True, rationale="Hypothesis")
    assert proposal.settings["lag_menu"].status == "proposed"
    ready = proposal.confirm(proposal.unresolved, "3")
    assert ready.request().lag_menu == [0, 1]
    changed = ready.patch({"target_id": "inflation"}, "4")
    assert "candidate_ids" in changed.unresolved
    assert "lag_menu" in changed.unresolved
    assert changed.settings["model_size"].status == "confirmed"
    with pytest.raises(ValueError):
        ready.patch({"model_size": 9}, "5")
    assert ready.request().model_size == 1


def test_unknown_fields_and_invalid_modes_rejected():
    for change in ({"arbitrary_tool": 1}, {"run_mode": "magic"}, {"share_vectors": "yes"}):
        with pytest.raises(ValueError):
            AnalysisDraft.new().patch(change, "1")
