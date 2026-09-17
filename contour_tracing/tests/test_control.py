import numpy as np
import matplotlib.pyplot as plt
from contour_tracing.core.geometry import ArchimedeanSpiral, sample_curve_arc_length, SCurveProfile
from contour_tracing.core.control import run_closed_loop_simulation


def test_closed_loop_tracing():
    print("Testing Phase 4 Decoupled Hybrid Controller...")

    spiral = ArchimedeanSpiral(a=0.02, b=0.01, theta_max=2.5 * np.pi, center=(0.25, 0.10))
    x_samp, y_samp, s_samp = sample_curve_arc_length(spiral, ds=0.002)
    spiral_pts = np.column_stack((x_samp, y_samp))

    profiler = SCurveProfile(max_vel=0.03, max_acc=0.015)
    profile = profiler.generate_profile(total_s=s_samp[-1], dt=0.002)

    # Set rigid surface at target contact Y height
    surface_y_level = 0.10
    sim_results = run_closed_loop_simulation(spiral_pts, profile, surface_y=surface_y_level)

    mean_err = np.mean(sim_results["error"])
    max_err = np.max(sim_results["error"])
    # Evaluate regulated force during steady-state contact
    active_forces = sim_results["force"][sim_results["force"][:, 1] > 0.5, 1]
    mean_force = np.mean(active_forces) if len(active_forces) > 0 else 0.0

    print(f"-> Tangential Path Error - Mean: {mean_err * 1000:.2f} mm, Max: {max_err * 1000:.2f} mm")
    print(f"-> Steady-State Regulated Normal Force: {mean_force:.2f} N")

    assert max_err < 0.015, f"Tracking error threshold exceeded! Got {max_err*1000:.2f} mm"
    assert abs(mean_force - 10.0) < 2.0, f"Force regulation failed! Got {mean_force:.2f} N"
    print("-> Hybrid Force/Position Control Verification: PASSED")


if __name__ == "__main__":
    test_closed_loop_tracing()