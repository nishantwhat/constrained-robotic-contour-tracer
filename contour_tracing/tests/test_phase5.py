import numpy as np
from contour_tracing.core.geometry import ArchimedeanSpiral, sample_curve_arc_length, SCurveProfile
from contour_tracing.core.control import run_closed_loop_simulation
from contour_tracing.core.metrics import (
    compute_performance_metrics,
    validate_performance_metrics,
)
from contour_tracing.visualization.dashboard import plot_tracing_dashboard


def run_phase5_evaluation():
    print("==========================================================")
    print(" Running Phase 5 Final Pipeline & Dashboard Verification  ")
    print("==========================================================")

    # 1. Geometry & Profile Setup
    spiral = ArchimedeanSpiral(a=0.02, b=0.01, theta_max=2.5 * np.pi, center=(0.25, 0.10))
    x_samp, y_samp, s_samp = sample_curve_arc_length(spiral, ds=0.002)
    spiral_pts = np.column_stack((x_samp, y_samp))

    profiler = SCurveProfile(max_vel=0.03, max_acc=0.015)
    profile = profiler.generate_profile(total_s=s_samp[-1], dt=0.002)

    # 2. Closed-Loop Simulation
    surface_y_level = 0.10
    sim_results = run_closed_loop_simulation(spiral_pts, profile, surface_y=surface_y_level)

    # 3. Compute Quantitative Metrics
    metrics = compute_performance_metrics(
        t=sim_results["t"],
        pos_actual=sim_results["pos"],
        spiral_pts=spiral_pts,
        time_profile=profile,
        force_actual=sim_results["force"],
        target_force=10.0,
    )

    print("\n--- PERFORMANCE EVALUATION SUMMARY ---")
    print(f"-> Tangential Position RMSE : {metrics['rmse_pos_mm']:.2f} mm")
    print(f"-> Peak Tracking Error     : {metrics['max_pos_err_mm']:.2f} mm")
    print(f"-> Mean Contact Force      : {metrics['mean_force_n']:.2f} N")
    print(f"-> Normal Force Error RMSE : {metrics['rmse_force_n']:.2f} N")
    print("--------------------------------------\n")

    
    # Validate against system-level specifications
    validation = validate_performance_metrics(
        metrics=metrics,
        max_position_error_mm=15.0,
        target_force=10.0,
        force_tolerance_n=2.0,
    )

    print("\n--- SYSTEM VALIDATION ---")
    print(
        f"-> Position Tracking : "
        f"{'PASS' if validation['position_pass'] else 'FAIL'}"
    )
    print(
        f"-> Force Regulation  : "
        f"{'PASS' if validation['force_pass'] else 'FAIL'}"
    )
    print(
        f"-> Overall Status    : "
        f"{'PASS' if validation['overall_pass'] else 'FAIL'}"
    )
    print("-------------------------\n")

    assert validation["position_pass"], (
        "Position accuracy specification failed!"
    )

    assert validation["force_pass"], (
        "Force regulation specification failed!"
    )

    assert validation["overall_pass"], (
        "Controller performance validation failed!"
    )

    print("-> ALL PHASE 5 SYSTEM SPECIFICATIONS PASSED!")

    # 4. Render Dashboard
    plot_tracing_dashboard(
        t=sim_results["t"],
        spiral_pts=spiral_pts,
        pos_actual=sim_results["pos"],
        force_actual=sim_results["force"],
        errors=sim_results["error"],
        target_force=10.0,
        surface_y=surface_y_level,
    )


if __name__ == "__main__":
    run_phase5_evaluation()
