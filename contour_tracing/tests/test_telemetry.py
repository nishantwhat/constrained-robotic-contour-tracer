import numpy as np

from contour_tracing.core.telemetry import SimulationTelemetry


def test_telemetry_records_simulation_state():
    telemetry = SimulationTelemetry()

    telemetry.record(
        time=0.0,
        desired_position=np.array([0.25, 0.10]),
        actual_position=np.array([0.249, 0.101]),
        desired_force=10.0,
        actual_force=np.array([0.0, 9.8]),
        joint_position=np.array([0.6, 0.8]),
        joint_velocity=np.array([0.1, 0.2]),
        control_torque=np.array([1.5, 2.0]),
        tracking_error=0.001,
    )

    data = telemetry.to_dict()

    assert data["time"].shape == (1,)
    assert data["desired_position"].shape == (1, 2)
    assert data["actual_position"].shape == (1, 2)
    assert data["desired_force"].shape == (1,)
    assert data["actual_force"].shape == (1, 2)
    assert data["joint_position"].shape == (1, 2)
    assert data["joint_velocity"].shape == (1, 2)
    assert data["control_torque"].shape == (1, 2)
    assert data["tracking_error"].shape == (1,)


def test_telemetry_preserves_recorded_values():
    telemetry = SimulationTelemetry()

    torque = np.array([1.5, 2.0])

    telemetry.record(
        time=0.002,
        desired_position=np.array([0.3, 0.1]),
        actual_position=np.array([0.299, 0.101]),
        desired_force=10.0,
        actual_force=np.array([0.2, 9.7]),
        joint_position=np.array([0.7, 0.8]),
        joint_velocity=np.array([0.1, 0.15]),
        control_torque=torque,
        tracking_error=0.001,
    )

    data = telemetry.to_dict()

    assert np.allclose(data["control_torque"][0], torque)
    assert np.isclose(data["desired_force"][0], 10.0)
    assert np.isclose(data["tracking_error"][0], 0.001)
