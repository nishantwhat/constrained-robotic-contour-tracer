from dataclasses import dataclass, field
from typing import List

import numpy as np


@dataclass
class SimulationTelemetry:
    """Structured telemetry collected during closed-loop simulation."""

    time: List[float] = field(default_factory=list)

    desired_position: List[np.ndarray] = field(default_factory=list)
    actual_position: List[np.ndarray] = field(default_factory=list)

    desired_force: List[float] = field(default_factory=list)
    actual_force: List[np.ndarray] = field(default_factory=list)

    joint_position: List[np.ndarray] = field(default_factory=list)
    joint_velocity: List[np.ndarray] = field(default_factory=list)

    control_torque: List[np.ndarray] = field(default_factory=list)

    tracking_error: List[float] = field(default_factory=list)

    def record(
        self,
        time: float,
        desired_position: np.ndarray,
        actual_position: np.ndarray,
        desired_force: float,
        actual_force: np.ndarray,
        joint_position: np.ndarray,
        joint_velocity: np.ndarray,
        control_torque: np.ndarray,
        tracking_error: float,
    ) -> None:
        """Record one simulation timestep."""

        self.time.append(time)

        self.desired_position.append(desired_position.copy())
        self.actual_position.append(actual_position.copy())

        self.desired_force.append(float(desired_force))
        self.actual_force.append(actual_force.copy())

        self.joint_position.append(joint_position.copy())
        self.joint_velocity.append(joint_velocity.copy())

        self.control_torque.append(control_torque.copy())

        self.tracking_error.append(float(tracking_error))

    def to_dict(self) -> dict:
        """Return telemetry as NumPy arrays for analysis and visualization."""

        return {
            "time": np.asarray(self.time),
            "desired_position": np.asarray(self.desired_position),
            "actual_position": np.asarray(self.actual_position),
            "desired_force": np.asarray(self.desired_force),
            "actual_force": np.asarray(self.actual_force),
            "joint_position": np.asarray(self.joint_position),
            "joint_velocity": np.asarray(self.joint_velocity),
            "control_torque": np.asarray(self.control_torque),
            "tracking_error": np.asarray(self.tracking_error),
        }
