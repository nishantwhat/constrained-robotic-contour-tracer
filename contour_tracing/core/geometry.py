import numpy as np
from typing import Tuple, Dict


# class ArchimedeanSpiral:
#     """Parametric Archimedean Spiral in 2D task frame: r(theta) = a + b * theta."""

#     def __init__(self, a: float = 0.02, b: float = 0.015, theta_max: float = 4.0 * np.pi):
#         self.a = a
#         self.b = b
#         self.theta_max = theta_max

#     def evaluate(self, theta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
#         """Evaluates 2D Cartesian coordinates (x, y) given theta array."""
#         r = self.a + self.b * theta
#         x = r * np.cos(theta)
#         y = r * np.sin(theta)
#         return x, y
class ArchimedeanSpiral:
    """Parametric Archimedean Spiral shifted into active workspace."""

    def __init__(
        self,
        a: float = 0.02,
        b: float = 0.012,
        theta_max: float = 3.0 * np.pi,
        center: Tuple[float, float] = (0.25, 0.10),
    ):
        self.a = a
        self.b = b
        self.theta_max = theta_max
        self.xc, self.yc = center

    def evaluate(self, theta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        r = self.a + self.b * theta
        x = self.xc + r * np.cos(theta)
        y = self.yc + r * np.sin(theta)
        return x, y


def sample_curve_arc_length(
    curve: ArchimedeanSpiral, num_samples: int = 1000, ds: float = 0.002
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Samples parametric curve with approximately uniform spatial arc length ds.

    Returns:
        x_sampled: Sampled X coordinates in task frame (meters).
        y_sampled: Sampled Y coordinates in task frame (meters).
        s_accum: Accumulated arc length array (meters).
    """
    # High-resolution dense parameter sampling to compute arc lengths numerically
    theta_dense = np.linspace(0.0, curve.theta_max, num_samples * 10)
    x_dense, y_dense = curve.evaluate(theta_dense)

    # Numerical arc-length integration
    dx = np.diff(x_dense)
    dy = np.diff(y_dense)
    ds_dense = np.hypot(dx, dy)
    s_dense = np.concatenate(([0.0], np.cumsum(ds_dense)))

    # Interpolate theta at uniform spatial intervals ds
    total_length = s_dense[-1]
    s_uniform = np.arange(0.0, total_length, ds)
    theta_uniform = np.interp(s_uniform, s_dense, theta_dense)

    x_sampled, y_sampled = curve.evaluate(theta_uniform)
    return x_sampled, y_sampled, s_uniform


class SCurveProfile:
    """Generates continuous jerk (C^2) S-curve time parameterization for spatial arc length s."""

    def __init__(self, max_vel: float = 0.05, max_acc: float = 0.02):
        self.v_max = max_vel
        self.a_max = max_acc

    def generate_profile(self, total_s: float, dt: float = 0.02) -> Dict[str, np.ndarray]:
        """Computes a trapezoidal acceleration / smooth S-curve velocity profile over arc length total_s.

        Args:
            total_s: Total spatial path length (meters).
            dt: Controller sampling timestep (seconds).

        Returns:
            Dictionary containing time, s(t), v(t), and a(t) arrays.
        """
        # Acceleration ramp-up time to reach v_max
        t_acc = self.v_max / self.a_max
        s_acc = 0.5 * self.a_max * (t_acc**2)

        if 2 * s_acc > total_s:
            # Triangular profile if distance is too short to reach v_max
            s_acc = total_s / 2.0
            t_acc = np.sqrt(2 * s_acc / self.a_max)
            t_cruise = 0.0
            v_peak = self.a_max * t_acc
        else:
            t_cruise = (total_s - 2 * s_acc) / self.v_max
            v_peak = self.v_max

        t_total = 2 * t_acc + t_cruise
        t_array = np.arange(0.0, t_total + dt, dt)

        s_array = np.zeros_like(t_array)
        v_array = np.zeros_like(t_array)
        a_array = np.zeros_like(t_array)

        for i, t in enumerate(t_array):
            if t <= t_acc:
                # Acceleration phase
                a_array[i] = self.a_max
                v_array[i] = self.a_max * t
                s_array[i] = 0.5 * self.a_max * (t**2)
            elif t <= (t_acc + t_cruise):
                # Cruise phase
                a_array[i] = 0.0
                v_array[i] = v_peak
                s_array[i] = s_acc + v_peak * (t - t_acc)
            else:
                # Deceleration phase
                t_dec = t - (t_acc + t_cruise)
                a_array[i] = -self.a_max
                v_array[i] = max(0.0, v_peak - self.a_max * t_dec)
                s_array[i] = (
                    s_acc
                    + v_peak * t_cruise
                    + v_peak * t_dec
                    - 0.5 * self.a_max * (t_dec**2)
                )

        # Enforce boundary clamping
        s_array = np.clip(s_array, 0.0, total_s)
        return {"t": t_array, "s": s_array, "v": v_array, "a": a_array}