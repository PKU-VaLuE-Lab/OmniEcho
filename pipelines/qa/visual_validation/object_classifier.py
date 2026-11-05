"""OmniEcho processing component."""

from __future__ import annotations

from typing import Any, Dict


# ---------------------------------------------------------------------------
# Non-visual type vocabulary
# ---------------------------------------------------------------------------

# Set of *base* type tokens.  Matching is performed against the lowercased
# raw type AND against any underscore-delimited token of it, so that
# ``footsteps_on_stone`` matches ``footsteps`` and
# ``canal_water_ambience`` matches ``water`` / ``ambience``.
NON_VISUAL_TYPES: set[str] = {
    
    "footsteps", "footstep", "walking", "walk", "running",
    "running_footsteps", "wooden_footsteps",
    
    "wind", "breeze", "gust", "windy", "wind_gust",
    
    "rain", "rainfall", "drizzle", "raindrops",
    "thunder", "lightning",
    
    "ambient", "ambience", "atmosphere", "background",
    "echo", "reverb", "rustle", "rustling",
    
    "canal_water", "river_flow", "stream", "water_drip",
    "water", "waves", "ocean_waves", "wave",
    "river", "creek", "current",
    
    "crackling", "crackling_bonfire", "fire_crackle",
    
    "music", "melody", "song", "tune",
    "speech", "conversation", "talking", "chatter",
    "announcement", "station_announcement",
    "noise", "static", "hum", "buzz", "drone",
    
    "wind_chime", "windchime", "chime",
}


# ---------------------------------------------------------------------------
# Label mapping (scene.json type → SAM3 text prompt)
# ---------------------------------------------------------------------------

# Curated mapping.  Keys are the *full* scene.json ``type`` value (lower
# case), values are the natural-language phrase that produces the best
# detections from SAM3's open-vocabulary text head.  Entries with a
# trailing action descriptor (``_flapping``/``_singing``/``_barking``...)
# are also handled by the action-stripping fallback in
# :func:`label_for_segmentor`, so adding both is safe but not required.
LABEL_MAPPING: Dict[str, str] = {
    # ----- bicycles / motorbikes -----
    "bicycle_bell": "bicycle",
    "bike_bell": "bicycle",
    "bicycle": "bicycle",
    "bike": "bicycle",
    "motorcycle": "motorcycle",
    "motorbike": "motorcycle",
    "scooter": "scooter",
    # ----- birds -----
    "pigeon_flapping": "pigeon",
    "pigeon_cooing": "pigeon",
    "pigeon": "pigeon",
    "bird_singing": "bird",
    "bird_chirping": "bird",
    "bird_call": "bird",
    "seagull_call": "seagull",
    "seagull_calling": "seagull",
    "seagull": "seagull",
    "rooster_crowing": "rooster",
    "rooster": "rooster",
    "crow_cawing": "crow",
    "crow": "crow",
    # ----- mammals -----
    "dog_barking": "dog",
    "dog": "dog",
    "cat_meowing": "cat",
    "cat": "cat",
    "horse_neighing": "horse",
    "horse": "horse",
    "cow_mooing": "cow",
    "cow": "cow",
    "sheep_bleating": "sheep",
    "sheep": "sheep",
    # ----- vehicles -----
    "car_horn": "car",
    "car_passing": "car",
    "car_engine": "car",
    "car": "car",
    "truck_engine": "truck",
    "truck_horn": "truck",
    "truck": "truck",
    "bus_engine": "bus",
    "bus": "bus",
    "train_horn": "train",
    "train_passing": "train",
    "train": "train",
    "boat_horn": "boat",
    "boat": "boat",
    "ship": "ship",
    "airplane": "airplane",
    "helicopter": "helicopter",
    "siren": "emergency vehicle",
    # ----- bells / clocks / phones -----
    "church_bell": "church",
    "church_bells": "church",
    "alarm_clock": "alarm clock",
    "alarm": "alarm clock",
    "clock_ticking": "clock",
    "phone_ringing": "phone",
    "phone": "phone",
    # ----- doors / household -----
    "door_knock": "door",
    "door_creak": "door",
    "door": "door",
    "doorbell": "door",
    "kettle_whistle": "kettle",
    "kettle": "kettle",
    # ----- water with visible source -----
    "fountain_water": "fountain",
    "fountain": "fountain",
    # ----- tools / construction -----
    "drill": "drill",
    "hammer": "hammer",
    "blacksmith_hammer": "anvil",
    "saw": "saw",
    # ----- musical instruments / tech -----
    "keyboard_typing": "keyboard",
    "keyboard": "keyboard",
    "guitar_playing": "guitar",
    "guitar": "guitar",
    "piano_playing": "piano",
    "piano": "piano",
    "drum_beating": "drum",
    "drums": "drum",
    "violin": "violin",
    "trumpet": "trumpet",
    # ----- people (occasionally visible) -----
    "children_playing": "children",
    "children": "children",
    "baby_crying": "baby",
    "baby": "baby",
    "crowd_cheering": "crowd",
    "crowd": "crowd",
    # ----- carts / wheels -----
    "wooden_cart_wheels": "cart",
    "cart": "cart",
    "wheelbarrow": "wheelbarrow",
}


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def _normalise_type(raw: Any) -> str:
    """Return ``str(raw)`` lower-cased and stripped (or ``""``)."""

    if raw is None:
        return ""
    return str(raw).strip().lower()


def is_visual_object(obj: Dict[str, Any]) -> bool:
    """Return ``True`` when ``obj`` should participate in visual checks.

    Decision rule:

    1. If ``obj`` carries an explicit ``is_visual`` field (either bool or
       ``"true"`` / ``"false"``) it is honoured verbatim – this gives
       upstream renderers / human reviewers a manual override.
    2. Otherwise the ``type`` is normalised and compared against
       :data:`NON_VISUAL_TYPES` both as a whole string and token-by-token
       on the underscore-separated parts.  A match in either form classifies
       the object as *non-visual*.

    Examples
    --------
    >>> is_visual_object({"type": "bicycle_bell"})
    True
    >>> is_visual_object({"type": "footsteps_on_stone"})
    False
    >>> is_visual_object({"type": "canal_water_ambience"})
    False
    """

    # Manual override.
    explicit = obj.get("is_visual")
    if isinstance(explicit, bool):
        return explicit
    if isinstance(explicit, str):
        if explicit.strip().lower() in {"true", "yes", "1"}:
            return True
        if explicit.strip().lower() in {"false", "no", "0"}:
            return False

    obj_type = _normalise_type(obj.get("type"))
    if not obj_type:
        # No type hint at all → safest to treat as visual; the segmentor
        # will simply produce no detections if it cannot find anything.
        return True

    if obj_type in NON_VISUAL_TYPES:
        return False

    # Token-wise match: ``footsteps_on_stone`` matches ``footsteps``,
    # ``canal_water_ambience`` matches both ``water`` and ``ambience``.
    tokens = [t for t in obj_type.split("_") if t]
    if any(tok in NON_VISUAL_TYPES for tok in tokens):
        return False

    return True


def label_for_segmentor(obj: Dict[str, Any]) -> str:
    """Return the SAM3 text prompt for ``obj``.

    Resolution order:

    1. Exact match in :data:`LABEL_MAPPING` for the object's ``type``.
    2. Drop the *last* underscore-delimited token (the action suffix –
       ``flapping`` / ``barking`` / ``singing`` / ...) and look up again
       (``pigeon_cooing`` → ``pigeon``).
    3. Drop the *first* underscore-delimited token to handle modifiers
       like ``loud_drill`` → ``drill`` (only when the result is in the
       mapping).
    4. Fallback: replace ``_`` with spaces, lowercase, strip.
    5. As a last resort use the display ``label`` or ``id`` so the report
       still has *some* identifier to display – these keys will not match
       anything in SAM3 but downstream reporting still works.
    """

    raw_type = _normalise_type(obj.get("type"))
    if raw_type:
        if raw_type in LABEL_MAPPING:
            return LABEL_MAPPING[raw_type]

        if "_" in raw_type:
            base_left = raw_type.rsplit("_", 1)[0]
            if base_left in LABEL_MAPPING:
                return LABEL_MAPPING[base_left]
            base_right = raw_type.split("_", 1)[1]
            if base_right in LABEL_MAPPING:
                return LABEL_MAPPING[base_right]

        # Token-wise: if any underscore token by itself maps cleanly use it.
        for tok in raw_type.split("_"):
            if tok and tok in LABEL_MAPPING:
                return LABEL_MAPPING[tok]

        return raw_type.replace("_", " ").strip()

    label = obj.get("label")
    if label:
        return str(label).strip()
    return str(obj.get("id", ""))


__all__ = [
    "NON_VISUAL_TYPES",
    "LABEL_MAPPING",
    "is_visual_object",
    "label_for_segmentor",
]
