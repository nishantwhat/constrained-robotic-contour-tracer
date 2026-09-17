import numpy as np
from contour_tracing.core.dynamics import TwoLinkDynamics
from contour_tracing.core.kinematics import Planar2LinkKinematics


def test_dynamics_engine():
    print("Testing Phase 3 Dynamics Engine...")
    dyn = TwoLinkDynamics()
    kin = Planar2LinkKinematics()

    q = np.array([0.2, 0.5])
    dq = np.array([0.1, -0.2])

    # 1. Inertia Matrix Positive-Definiteness Check
    M = dyn.inertia_matrix(q)
    eigenvalues = np.linalg.eigvals(M)
    assert np.all(eigenvalues > 0), "Inertia Matrix M(q) is not positive-definite!"
    print(f"-> Inertia Matrix Check: PASSED (Eigenvalues: {eigenvalues})")

    # 2. Contact Force Interaction Verification
    pos_above = np.array([0.2, 0.05])
    vel = np.array([0.05, -0.01])
    f_no_contact = dyn.compute_contact_force(pos_above, vel, surface_y=0.0)
    assert np.allclose(f_no_contact, [0.0, 0.0]), "Force detected without surface contact!"

    pos_penetrated = np.array([0.2, -0.002])  # 2 mm penetration
    f_contact = dyn.compute_contact_force(pos_penetrated, vel, surface_y=0.0)
    assert f_contact[1] > 0, "Normal force failed to engage upon penetration!"
    print(f"-> Contact Physics Check: PASSED (Normal Force: {f_contact[1]:.2f} N, Tangential Force: {f_contact[0]:.2f} N)")

    # 3. Forward Dynamics Integration Check
    J = kin.jacobian(q)
    tau = dyn.gravity_vector(q)  # Gravity compensation torque
    ddq = dyn.forward_dynamics(q, dq, tau, f_contact, J)

    assert not np.isnan(ddq).any(), "Forward dynamics returned NaN acceleration!"
    print(f"-> Forward Dynamics Check: PASSED (Joint Accelerations: {ddq})")


if __name__ == "__main__":
    test_dynamics_engine()