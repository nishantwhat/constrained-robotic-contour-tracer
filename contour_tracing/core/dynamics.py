import numpy as np
from typing import Tuple


class TwoLinkDynamics:
    """Rigid body dynamics and contact interaction model for a 2-DOF planar robot."""

    def __init__(
        self,
        m1: float = 1.2,
        m2: float = 0.8,
        l1: float = 0.25,
        l2: float = 0.25,
        lc1: float = 0.125,
        lc2: float = 0.125,
        I1: float = 0.025,
        I2: float = 0.015,
        k_env: float = 5000.0,
        b_env: float = 50.0,
        mu_friction: float = 0.15,
    ):
        # Physical Parameters
        self.m1, self.m2 = m1, m2
        self.l1, self.l2 = l1, l2
        self.lc1, self.lc2 = lc1, lc2
        self.I1, self.I2 = I1, I2
        self.g = 9.81

        # Surface Contact Model Parameters
        self.k_env = k_env          # Environment stiffness (N/m)
        self.b_env = b_env          # Environment damping (N*s/m)
        self.mu = mu_friction       # Coulomb friction coefficient

    def inertia_matrix(self, q: np.ndarray) -> np.ndarray:
        """Computes the 2x2 joint space mass matrix M(q)."""
        _, q2 = q[0], q[1]
        c2 = np.cos(q2)

        m11 = (
            self.I1
            + self.I2
            + self.m1 * self.lc1**2
            + self.m2 * (self.l1**2 + self.lc2**2 + 2 * self.l1 * self.lc2 * c2)
        )
        m12 = self.I2 + self.m2 * (self.lc2**2 + self.l1 * self.lc2 * c2)
        m21 = m12
        m22 = self.I2 + self.m2 * self.lc2**2

        return np.array([[m11, m12], [m21, m22]])

    def coriolis_vector(self, q: np.ndarray, dq: np.ndarray) -> np.ndarray:
        """Computes the 2x1 Coriolis and centrifugal forces vector C(q, dq)*dq."""
        _, q2 = q[0], q[1]
        dq1, dq2 = dq[0], dq[1]
        s2 = np.sin(q2)

        h = -self.m2 * self.l1 * self.lc2 * s2

        c1 = h * dq2 * dq1 + h * (dq1 + dq2) * dq2
        c2 = -h * (dq1**2)

        return np.array([c1, c2])

    def gravity_vector(self, q: np.ndarray) -> np.ndarray:
        """Computes the 2x1 gravity vector G(q)."""
        q1, q2 = q[0], q[1]
        g1 = (self.m1 * self.lc1 + self.m2 * self.l1) * self.g * np.cos(q1) + self.m2 * self.lc2 * self.g * np.cos(q1 + q2)
        g2 = self.m2 * self.lc2 * self.g * np.cos(q1 + q2)
        return np.array([g1, g2])

    def compute_contact_force(
        self, pos: np.ndarray, vel: np.ndarray, surface_y: float = 0.0
    ) -> np.ndarray:
        """Calculates normal contact force and tangential friction at surface interaction boundary.

        Args:
            pos: End-effector 2D position [x, y].
            vel: End-effector 2D velocity [vx, vy].
            surface_y: Task-space Y position of rigid surface.

        Returns:
            F_ext: External contact force vector [Fx, Fy] applied on the end-effector.
        """
        penetration = surface_y - pos[1]

        if penetration > 0:
            # Contact engaged: Normal force via Kelvin-Voigt spring-damper model
            v_normal = -vel[1]
            f_normal = max(0.0, self.k_env * penetration + self.b_env * v_normal)

            # Tangential Coulomb friction opposing motion
            v_tangent = vel[0]
            f_tangent = -self.mu * f_normal * np.sign(v_tangent) if abs(v_tangent) > 1e-4 else 0.0

            return np.array([f_tangent, f_normal])

        # No contact
        return np.array([0.0, 0.0])

    def forward_dynamics(
        self, q: np.ndarray, dq: np.ndarray, tau: np.ndarray, f_ext: np.ndarray, J: np.ndarray
    ) -> np.ndarray:
        """Computes joint acceleration ddq = M^-1 * (tau + J^T * F_ext - C - G)."""
        M = self.inertia_matrix(q)
        C = self.coriolis_vector(q, dq)
        G = self.gravity_vector(q)

        tau_ext = J.T @ f_ext
        ddq = np.linalg.solve(M, tau + tau_ext - C - G)
        return ddq