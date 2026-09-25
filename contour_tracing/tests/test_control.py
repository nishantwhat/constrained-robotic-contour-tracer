import numpy as np

from contour_tracing.core.control import (
    DecoupledHybridController,
    RobotState,
    ControlReference,
    ControllerCommand,
)


def test_controller_interface_returns_command():
    """Controller should return a valid ControllerCommand."""

    controller = DecoupledHybridController()

    state = RobotState(
        position=np.array([0.25, 0.10]),
        velocity=np.array([0.01, 0.0]),
        force=np.array([0.0, 9.5]),
        joint_position=np.array([0.6, 0.8]),
        joint_velocity=np.array([0.01, 0.02]),
        jacobian=np.eye(2),
        gravity=np.zeros(2),
        coriolis=np.zeros(2),
    )

    reference = ControlReference(
        position=np.array([0.26, 0.10]),
        velocity=np.array([0.02, 0.0]),
        force=10.0,
    )

    command = controller.compute(
        state=state,
        reference=reference,
        dt=0.002,
    )

    assert isinstance(command, ControllerCommand)
    assert command.task_wrench.shape == (2,)
    assert command.torque.shape == (2,)
    assert np.all(np.isfinite(command.torque))


def test_controller_interface_accepts_external_model_quantities():
    """Controller should work with externally supplied model quantities."""

    controller = DecoupledHybridController()

    state = RobotState(
        position=np.array([0.25, 0.10]),
        velocity=np.array([0.0, 0.0]),
        force=np.array([0.0, 10.0]),
        joint_position=np.array([0.6, 0.8]),
        joint_velocity=np.array([0.0, 0.0]),
        jacobian=np.array([
            [0.5, 0.2],
            [0.3, 0.4],
        ]),
        gravity=np.array([1.0, 2.0]),
        coriolis=np.array([0.1, 0.2]),
    )

    reference = ControlReference(
        position=np.array([0.25, 0.10]),
        velocity=np.array([0.0, 0.0]),
        force=10.0,
    )

    command = controller.compute(
        state=state,
        reference=reference,
        dt=0.002,
    )

    assert isinstance(command, ControllerCommand)
    assert command.task_wrench.shape == (2,)
    assert command.torque.shape == (2,)
    assert np.all(np.isfinite(command.torque))
