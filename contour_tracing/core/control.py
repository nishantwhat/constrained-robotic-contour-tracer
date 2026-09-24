import numpy as np
from contour_tracing.core.kinematics import Planar2LinkKinematics
from contour_tracing.core.dynamics import TwoLinkDynamics
from contour_tracing.core.telemetry import SimulationTelemetry


class DecoupledHybridController:
    """True Hybrid Force/Position Controller using orthogonal subspace projection.

    Projects position control onto the tangent plane and force control strictly
    along the surface normal.
    """

    def __init__(
        self,
        kp_pos: float = 800.0,
        kd_pos: float = 60.0,
        kp_force: float = 0.05,
        ki_force: float = 1.2,
        target_force: float = 10.0,
    ):
        self.kp_pos = kp_pos
        self.kd_pos = kd_pos
        self.kp_force = kp_force
        self.ki_force = ki_force
        self.f_des = target_force
        self.f_integral = 0.0

    def compute_control_torque(
        self,
        q: np.ndarray,
        dq: np.ndarray,
        x_des: np.ndarray,
        v_des: np.ndarray,
        f_meas: np.ndarray,
        kin: Planar2LinkKinematics,
        dyn: TwoLinkDynamics,
        dt: float,
    ) -> np.ndarray:
        x_curr = kin.forward_kinematics(q)
        J = kin.jacobian(q)
        v_curr = J @ dq

        # 1. Pure Tangential Position Control (X-Axis)
        e_x = x_des[0] - x_curr[0]
        e_vx = v_des[0] - v_curr[0]
        Fx = self.kp_pos * e_x + self.kd_pos * e_vx

        # 2. Pure Normal Force Control (Y-Axis)
        # We do NOT apply position PD along Y to prevent axis competition
        fn_meas = f_meas[1]
        fn_err = self.f_des - fn_meas

        if fn_meas > 0.01:
            self.f_integral = np.clip(
                self.f_integral + fn_err * dt,
                -15.0,
                15.0,
            )
        else:
            # Pre-load integral slightly to feedforward surface entry smoothly
            self.f_integral = 2.0

        Fy = -(
            self.kp_force * fn_err
            + self.ki_force * self.f_integral
        )

        # 3. Decoupled Task-Space Wrench
        F_task = np.array([Fx, Fy])

        # 4. Model Inversion / Dynamic Compensation
        G = dyn.gravity_vector(q)
        C = dyn.coriolis_vector(q, dq)

        return J.T @ F_task + G + C


def run_closed_loop_simulation(
    spiral_points: np.ndarray,
    time_profile: dict,
    surface_y: float = 0.0,
) -> dict:
    kin = Planar2LinkKinematics()
    dyn = TwoLinkDynamics()
    controller = DecoupledHybridController()

    t_array = time_profile["t"]
    dt = t_array[1] - t_array[0]
    num_steps = len(t_array)

    q_curr, _ = kin.inverse_kinematics_dls(
        target_pos=spiral_points[0],
        q_init=np.array([0.6, 0.8]),
    )
    dq_curr = np.zeros(2)

    telemetry = SimulationTelemetry()

    for i in range(num_steps):
        s_target = time_profile["s"][i]
        v_target = time_profile["v"][i]

        idx = min(
            int(s_target / 0.002),
            len(spiral_points) - 1,
        )

        x_des = spiral_points[idx]
        v_des = np.array([v_target, 0.0])

        x_curr = kin.forward_kinematics(q_curr)
        J = kin.jacobian(q_curr)
        v_curr = J @ dq_curr

        f_ext = dyn.compute_contact_force(
            x_curr,
            v_curr,
            surface_y,
        )

        tau = controller.compute_control_torque(
            q_curr,
            dq_curr,
            x_des,
            v_des,
            f_ext,
            kin,
            dyn,
            dt,
        )

        ddq = dyn.forward_dynamics(
            q_curr,
            dq_curr,
            tau,
            f_ext,
            J,
        )

        dq_curr += ddq * dt
        q_curr += dq_curr * dt

        # Evaluate tangential error along tracked X motion
        tracking_error = abs(
            x_des[0] - x_curr[0]
        )

        telemetry.record(
            time=t_array[i],
            desired_position=x_des,
            actual_position=x_curr,
            desired_force=controller.f_des,
            actual_force=f_ext,
            joint_position=q_curr,
            joint_velocity=dq_curr,
            control_torque=tau,
            tracking_error=tracking_error,
        )

    # Convert collected telemetry to NumPy arrays once,
    # after the simulation has completed.
    telemetry_data = telemetry.to_dict()

    return {
        # Backward-compatible simulation outputs
        "t": telemetry_data["time"],
        "q": telemetry_data["joint_position"],
        "pos": telemetry_data["actual_position"],
        "force": telemetry_data["actual_force"],
        "error": telemetry_data["tracking_error"],

        # Structured telemetry for analysis and simulator integration
        "telemetry": telemetry_data,
    }
