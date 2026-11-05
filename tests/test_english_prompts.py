"""Check English prompt contracts and spatial text/geometry consistency."""

from omniecho_audio.director import (
    _build_user_prompt,
    _detect_contradictions,
    _detect_text_direction,
    _simple_swap_direction_consistency,
    _swap_left_right,
)
from omniecho_audio.production_audit import prompt_has_required_contract
from scripts.filter_samples import is_reasonable_mover


def test_english_direction_words_match_without_substring_collisions():
    assert _detect_text_direction("A bird is on the LEFT, overhead and far away.") == {
        "horizontal": "left", "vertical": "up", "distance": "far"
    }
    assert _detect_text_direction("A bright upright clock has a linear pattern.") == {
        "horizontal": None, "vertical": None, "distance": None
    }
    assert _swap_left_right("A bright object is on the left; LEFT-1/3.") == (
        "A bright object is on the right; RIGHT-1/3."
    )


def test_english_contradictions_cover_position_height_and_distance():
    scene = {
        "objects": [{
            "id": "bell", "label": "bell", "type": "bell",
            "trajectory": [
                {"t": 0.0, "azimuth": -30, "elevation": -20, "distance": 8},
                {"t": 6.0, "azimuth": -30, "elevation": -20, "distance": 8},
            ],
        }],
        "sub_prompts": ["The bell is on the left, above the listener, and nearby."],
        "video_prompt": "",
    }
    found = _detect_contradictions(scene)
    assert {item["dimension"] for item in found} == {
        "horizontal (azimuth)", "vertical (elevation)", "distance"
    }
    fixed, changes = _simple_swap_direction_consistency(scene)
    assert "on the right" in fixed["sub_prompts"][0]
    assert len(changes) == 1


def test_english_prompt_contract_and_static_camera_constraint():
    prompt = (
        "Spatial layout: a bell is 2 meters to the left. "
        "0.0-2.0 seconds: it rings. Audio design: clear attacks. "
        "No background music or subtitles."
    )
    assert all(prompt_has_required_contract(prompt).values())
    weak = prompt_has_required_contract("A bright meeting room with people.")
    assert not any(weak.values())
    request = _build_user_prompt(2, True, 6.0, False)
    assert "must remain constant" in request
    assert "all text in English" in request


def test_mover_labels_match_english_words_without_false_vehicle_or_animal_hits():
    assert is_reasonable_mover("cat_footsteps", "A walking cat")
    assert is_reasonable_mover("bird_flapping", "Flying bird")
    assert is_reasonable_mover("", "A person jogging")
    assert not is_reasonable_mover("cardboard_rustling", "A stationary cardboard box")
    assert not is_reasonable_mover("notification", "A notification tone")
