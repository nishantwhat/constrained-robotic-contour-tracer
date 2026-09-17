import numpy as np
from typing import Tuple


class Planar2LinkKinematics:
    """Kinematic engine for a 2-DOF planar serial manipulator."""

    def __init__(self, l1: float = 0.25, l2: float = 0.25):
        """
        Args:
            l1: Length of Link 1 in meters.
            l2: Length of Link 2 in meters.
        """
        self.l1 = l1
        self.l2 = l2

    def forward_kinematics(self, q: np.ndarray) -> np.ndarray:
        """Computes 2D Cartesian end-effector position (x, y) from joint angles q [q1, q2]."""
        q1, q2 = q[0], q[1]
        x = self.l1 * np.cos(q1) + self.l2 * np.cos(q1 + q2)
        y = self.l1 * np.sin(q1) + self.l2 * np.sin(q1 + q2)
        return np.array([x, y])

    def get_joint_positions(self, q: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns base, elbow, and end-effector coordinates for visualization."""
        p0 = np.array([0.0, 0.0])
        p1 = np.array([self.l1 * np.cos(q[0]), self.l1 * np.sin(q[0])])
        p2 = self.forward_kinematics(q)
        return p0, p1, p2

    def jacobian(self, q: np.ndarray) -> np.ndarray:
        """Computes the 2x2 analytical Jacobian matrix at configuration q."""
        q1, q2 = q[0], q[1]
        s1 = np.sin(q1)
        c1 = np.cos(q1)
        s12 = np.sin(q1 + q2)
        c12 = np.cos(q1 + q2)

        j11 = -self.l1 * s1 - self.l2 * s12
        j12 = -self.l2 * s12
        j21 = self.l1 * c1 + self.l2 * c12
        j22 = self.l2 * c12

        return np.array([[j11, j12], [j21, j22]])

    def inverse_kinematics_dls(
        self,
        target_pos: np.ndarray,
        q_init: np.ndarray,
        damping: float = 0.01,
        max_iters: int = 100,
        tol: float = 1e-5,
    ) -> Tuple[np.ndarray, bool]:
        """Solves IK using Damped Least-Squares (Levenberg-Marquardt singularity robust IK).

        Args:
            target_pos: Desired [x, y] position in meters.
            q_init: Initial joint angles guess [q1, q2] in radians.
            damping: Damping factor lambda for singularity handling.
            max_iters: Maximum convergence iterations.
            tol: Position error tolerance in meters.

        Returns:
            q_sol: Converged joint configuration.
            success: Boolean flag indicating convergence.
        """
        q = np.array(q_init, dtype=np.float64)

        for _ in range(max_iters):
            current_pos = self.forward_kinematics(q)
            error = target_pos - current_pos

            if np.linalg.norm(error) < tol:
                return q, True

            J = self.jacobian(q)
            # DLS Inverse: J_dls = J^T * (J * J^T + lambda^2 * I)^(-1)
            J_JT = J @ J.T
            damped_inv = np.linalg.inv(J_JT + (damping**2) * np.eye(2))
            J_dls = J.T @ damped_inv

            # Update joint configuration step
            dq = J_dls @ error
            q += dq

            # Wrap joint angles between [-pi, pi]
            q = (q + np.pi) % (2 * np.pi) - np.pi

        return q, False