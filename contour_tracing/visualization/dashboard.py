import matplotlib.pyplot as plt
import numpy as np


def plot_tracing_dashboard(
    t: np.ndarray,
    spiral_pts: np.ndarray,
    pos_actual: np.ndarray,
    force_actual: np.ndarray,
    errors: np.ndarray,
    target_force: float = 10.0,
    surface_y: float = 0.10,
):
    """Renders 4-panel telemetry for closed-loop contour tracing."""
    fig, axs = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle("Phase 5: Hybrid Contour Tracing Performance Dashboard", fontsize=14, fontweight="bold")

    # 1. Spatial Trajectory Plot
    axs[0, 0].plot(spiral_pts[:, 0], spiral_pts[:, 1], "k--", label="Target Path")
    axs[0, 0].plot(pos_actual[:, 0], pos_actual[:, 1], "b-", label="Actual Path")
    axs[0, 0].axhline(y=surface_y, color="r", linestyle=":", label="Surface Plane")
    axs[0, 0].set_title("Task Space Trajectory")
    axs[0, 0].set_xlabel("X [m]")
    axs[0, 0].set_ylabel("Y [m]")
    axs[0, 0].legend()
    axs[0, 0].grid(True)

    # 2. Tangential Error Over Time
    axs[0, 1].plot(t, errors * 1000.0, "g-")
    axs[0, 1].axhline(y=15.0, color="r", linestyle="--", label="Threshold (15mm)")
    axs[0, 1].set_title("Tangential Tracking Error")
    axs[0, 1].set_xlabel("Time [s]")
    axs[0, 1].set_ylabel("Error [mm]")
    axs[0, 1].legend()
    axs[0, 1].grid(True)

    # 3. Normal Force Regulation
    axs[1, 0].plot(t, force_actual[:, 1], "m-", label="Measured Normal Force")
    axs[1, 0].axhline(y=target_force, color="k", linestyle="--", label=f"Target ({target_force}N)")
    axs[1, 0].set_title("Contact Normal Force")
    axs[1, 0].set_xlabel("Time [s]")
    axs[1, 0].set_ylabel("Force [N]")
    axs[1, 0].legend()
    axs[1, 0].grid(True)

    # 4. Normal vs. Tangential Force Component Comparison
    axs[1, 1].plot(t, force_actual[:, 0], "c-", label="Tangential Friction Force")
    axs[1, 1].plot(t, force_actual[:, 1], "m-", label="Normal Contact Force")
    axs[1, 1].set_title("Environmental Interaction Forces")
    axs[1, 1].set_xlabel("Time [s]")
    axs[1, 1].set_ylabel("Force [N]")
    axs[1, 1].legend()
    axs[1, 1].grid(True)

    plt.tight_layout()
    plt.show()