from contour_tracing.core.metrics import validate_performance_metrics


def test_performance_validation_passes():
    metrics = {
        "max_pos_err_mm": 5.0,
        "mean_force_n": 10.0,
    }

    result = validate_performance_metrics(metrics)

    assert result["position_pass"]
    assert result["force_pass"]
    assert result["overall_pass"]


def test_position_validation_fails_when_error_is_too_large():
    metrics = {
        "max_pos_err_mm": 20.0,
        "mean_force_n": 10.0,
    }

    result = validate_performance_metrics(metrics)

    assert not result["position_pass"]
    assert result["force_pass"]
    assert not result["overall_pass"]


def test_force_validation_fails_when_force_error_is_too_large():
    metrics = {
        "max_pos_err_mm": 5.0,
        "mean_force_n": 15.0,
    }

    result = validate_performance_metrics(metrics)

    assert result["position_pass"]
    assert not result["force_pass"]
    assert not result["overall_pass"]
