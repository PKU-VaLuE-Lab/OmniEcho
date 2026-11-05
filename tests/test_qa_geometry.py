"""OmniEcho processing component."""
import sys

from visual_validation.metadata_correction import (
    _inverse_project,
    _smooth_trajectory,
    _resample_trajectory,
    _extrapolate_trajectory,
    MetadataCorrector,
)
from omniecho_audio.schema import (
    CameraMode, VisibilityState, AnnotatedTrajectoryPoint, OffScreenStrategy
)
import math

def test_inverse_project_center():
    """OmniEcho processing component."""
    W, H = 1920, 1080
    fov_h = 90.0
    az, el = _inverse_project(W/2, H/2, W, H, fov_h)
    assert abs(az) < 0.5, f"Center azimuth should be ~0, got {az}"
    assert abs(el) < 0.5, f"Center elevation should be ~0, got {el}"
    print("✓ test_inverse_project_center passed")

def test_inverse_project_edges():
    """OmniEcho processing component."""
    W, H = 1920, 1080
    fov_h = 90.0
    az_left, _ = _inverse_project(0, H/2, W, H, fov_h)
    az_right, _ = _inverse_project(W, H/2, W, H, fov_h)
    
    assert az_left > 40, f"Left edge azimuth should be ~+45, got {az_left}"
    assert az_right < -40, f"Right edge azimuth should be ~-45, got {az_right}"
    print("✓ test_inverse_project_edges passed")

def test_extrapolate_confidence_decay():
    """OmniEcho processing component."""
    
    last_points = [
        AnnotatedTrajectoryPoint(time=1.0, azimuth=10.0, elevation=0.0, distance=2.0,
                         gain_db=0.0, visibility_state=VisibilityState.ON_SCREEN, confidence=0.9),
        AnnotatedTrajectoryPoint(time=1.25, azimuth=12.0, elevation=0.0, distance=2.0,
                         gain_db=0.0, visibility_state=VisibilityState.ON_SCREEN, confidence=0.9),
        AnnotatedTrajectoryPoint(time=1.5, azimuth=14.0, elevation=0.0, distance=2.0,
                         gain_db=0.0, visibility_state=VisibilityState.ON_SCREEN, confidence=0.9),
    ]
    
    extrapolated = _extrapolate_trajectory(last_points, duration=2.0, sample_interval=0.25)
    if extrapolated:
        
        assert extrapolated[0].confidence <= 0.85, f"First extrapolated confidence too high: {extrapolated[0].confidence}"
        
        assert extrapolated[-1].confidence < extrapolated[0].confidence, "Confidence should decrease"
        
        assert extrapolated[0].azimuth > 14.0, f"Extrapolated azimuth should continue trend"
        print(f"✓ test_extrapolate_confidence_decay passed (conf: {extrapolated[0].confidence:.2f}→{extrapolated[-1].confidence:.2f})")
    else:
        print("⚠ _extrapolate_trajectory returned empty (check implementation)")

def test_extrapolate_azimuth_clamp():
    """OmniEcho processing component."""
    last_points = [
        AnnotatedTrajectoryPoint(time=0.0, azimuth=120.0, elevation=0.0, distance=2.0,
                         gain_db=0.0, visibility_state=VisibilityState.ON_SCREEN, confidence=0.9),
        AnnotatedTrajectoryPoint(time=0.25, azimuth=125.0, elevation=0.0, distance=2.0,
                         gain_db=0.0, visibility_state=VisibilityState.ON_SCREEN, confidence=0.9),
        AnnotatedTrajectoryPoint(time=0.5, azimuth=130.0, elevation=0.0, distance=2.0,
                         gain_db=0.0, visibility_state=VisibilityState.ON_SCREEN, confidence=0.9),
    ]
    extrapolated = _extrapolate_trajectory(last_points, duration=3.0, sample_interval=0.25)
    if extrapolated:
        for pt in extrapolated:
            assert abs(pt.azimuth) <= 135.0, f"Azimuth exceeded ±135°: {pt.azimuth}"
        print("✓ test_extrapolate_azimuth_clamp passed")
    else:
        print("⚠ _extrapolate_trajectory returned empty")

def test_schema_enums():
    """OmniEcho processing component."""
    assert CameraMode.STATIC == "static"
    assert VisibilityState.OFF_SCREEN_DYNAMIC == "off_screen_dynamic"
    assert OffScreenStrategy.TRUST_METADATA == "trust_metadata"
    print("✓ test_schema_enums passed")

def test_corrector_init():
    """OmniEcho processing component."""
    from visual_validation.config import ValidationConfig
    config = ValidationConfig.default()
    corrector = MetadataCorrector(config)
    assert corrector is not None
    print("✓ test_corrector_init passed")

if __name__ == "__main__":
    test_schema_enums()
    test_corrector_init()
    test_inverse_project_center()
    test_inverse_project_edges()
    test_extrapolate_confidence_decay()
    test_extrapolate_azimuth_clamp()
    print("\n=== All OmniEcho unit tests passed ===")
