import numpy as np
from contour_tracing.core.kinematics import Planar2LinkKinematics


def test_kinematics_engine():
    print("Testing Phase 2 Kinematics Engine...")
    robot = Planar2LinkKinematics(l1=0.25, l2=0.25)

    # 1. Forward Kinematics Verification (Fully extended horizontal: q = [0, 0])
    q_test = np.array([0.0, 0.0])
    pos_fk = robot.forward_kinematics(q_test)
    assert np.allclose(pos_fk, [0.5, 0.0]), f"FK Failed! Expected [0.5, 0.0], got {pos_fk}"
    print("-> Forward Kinematics check: PASSED")

    # 2. DLS Inverse Kinematics Verification
    target_pos = np.array([0.2, 0.25])
    q_sol, success = robot.inverse_kinematics_dls(target_pos, q_init=np.array([0.1, 0.1]))

    assert success, "DLS IK failed to converge within maximum iterations!"
    pos_achieved = robot.forward_kinematics(q_sol)
    ik_error = np.linalg.norm(target_pos - pos_achieved)
    assert ik_error < 1e-4, f"IK positional accuracy failure! Error: {ik_error}"
    print(f"-> DLS Inverse Kinematics check: PASSED (Positional Error: {ik_error:.6f} m)")

    # 3. Jacobian Consistency Check (Finite Difference vs Analytical)
    eps = 1e-6
    q_rand = np.array([0.4, -0.6])
    J_analytical = robot.jacobian(q_rand)

    # Numerical Jacobian via central differences
    J_numerical = np.zeros((2, 2))
    for i in range(2):
        q_plus = q_rand.copy()
        q_minus = q_rand.copy()
        q_plus[i] += eps
        q_minus[i] -= eps

        pos_plus = robot.forward_kinematics(q_plus)
        pos_minus = robot.forward_kinematics(q_minus)
        J_numerical[:, i] = (pos_plus - pos_minus) / (2 * eps)

    jacobian_diff = np.max(np.abs(J_analytical - J_numerical))
    assert jacobian_diff < 1e-5, f"Jacobian mismatch! Max Difference: {jacobian_diff}"
    print(f"-> Analytical vs. Finite-Difference Jacobian check: PASSED (Diff: {jacobian_diff:.8f})")


if __name__ == "__main__":
    test_kinematics_engine()