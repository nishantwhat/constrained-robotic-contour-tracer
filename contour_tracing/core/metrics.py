import numpy as np


def compute_performance_metrics(
    t: np.ndarray,
    pos_actual: np.ndarray,
    spiral_pts: np.ndarray,
    time_profile: dict,
    force_actual: np.ndarray,
    target_force: float = 10.0,
) -> dict:
    """Computes quantitative metrics for hybrid force/position contour tracing."""

    s_profile = time_profile["s"]
    ds = 0.002  # Arc length sampling resolution

    # Reconstruct exact desired trajectory mapped over simulation time
    indices = np.clip(
        (s_profile / ds).astype(int),
        0,
        len(spiral_pts) - 1,
    )
    pos_desired = spiral_pts[indices]

    # Compute position tracking error
    e_x = pos_desired[:, 0] - pos_actual[:, 0]
    e_y = pos_desired[:, 1] - pos_actual[:, 1]

    rmse_pos = np.sqrt(np.mean(e_x**2))
    max_pos_err = np.max(np.abs(e_x))

    # Force regulation metrics
    f_normal = force_actual[:, 1]
    contact_mask = f_normal > 0.1

    if np.any(contact_mask):
        f_active = f_normal[contact_mask]
        mean_force = np.mean(f_active)
        f_err_rmse = np.sqrt(
            np.mean((f_active - target_force) ** 2)
        )
    else:
        mean_force = 0.0
        f_err_rmse = target_force

    return {
        "rmse_pos_mm": rmse_pos * 1000.0,
        "max_pos_err_mm": max_pos_err * 1000.0,
        "mean_force_n": mean_force,
        "rmse_force_n": f_err_rmse,
        "pos_desired": pos_desired,
    }


def validate_performance_metrics(
    metrics: dict,
    max_position_error_mm: float = 15.0,
    target_force: float = 10.0,
    force_tolerance_n: float = 2.0,
) -> dict:
    """
    Evaluates controller performance against system-level specifications.

    Returns a structured validation report containing individual
    pass/fail results and an overall validation status.
    """

    position_pass = (
        metrics["max_pos_err_mm"] < max_position_error_mm
    )

    force_pass = (
        abs(metrics["mean_force_n"] - target_force)
        < force_tolerance_n
    )

    overall_pass = position_pass and force_pass

    return {
        "position_pass": position_pass,
        "force_pass": force_pass,
        "overall_pass": overall_pass,
        "position_limit_mm": max_position_error_mm,
        "force_target_n": target_force,
        "force_tolerance_n": force_tolerance_n,
    }
