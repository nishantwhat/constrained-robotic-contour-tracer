import matplotlib.pyplot as plt
import numpy as np

from contour_tracing.core.geometry import ArchimedeanSpiral, sample_curve_arc_length, SCurveProfile


def test_geometry_and_profile():
    print("Testing Phase 1 Geometry Engine...")

    # 1. Instantiate and sample curve
    spiral = ArchimedeanSpiral(a=0.02, b=0.015, theta_max=4.0 * 3.14159)
    x, y, s = sample_curve_arc_length(spiral, ds=0.002)

    total_length = s[-1]
    print(f"-> Spiral sampled successfully. Points: {len(x)}, Total Arc Length: {total_length:.4f} meters")

    # Assert spatial sampling uniformity
    ds_measured = np.hypot(np.diff(x), np.diff(y))
    assert np.allclose(ds_measured, 0.002, atol=1e-4), "Spatial sampling ds is non-uniform!"
    print("-> Spatial sampling uniformity check: PASSED")

    # 2. Time-parameterize curve
    profiler = SCurveProfile(max_vel=0.05, max_acc=0.02)
    profile = profiler.generate_profile(total_s=total_length, dt=0.02)

    print(f"-> Trajectory profile generated. Duration: {profile['t'][-1]:.2f} seconds")
    assert profile["s"][-1] == total_length, "Final parameterized arc length does not match path length!"
    print("-> Boundary conditions check: PASSED")

    # 3. Plotting results for visual verification
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # Spatial Spiral Plot
    axs[0].plot(x, y, "b.-", label="Sampled Curve")
    axs[0].set_title("Arc-Length Sampled Archimedean Spiral")
    axs[0].set_xlabel("X (m)")
    axs[0].set_ylabel("Y (m)")
    axs[0].axis("equal")
    axs[0].grid(True)

    # Motion Profile Plot
    axs[1].plot(profile["t"], profile["s"], label="s(t) [m]")
    axs[1].plot(profile["t"], profile["v"], label="v(t) [m/s]")
    axs[1].plot(profile["t"], profile["a"], label="a(t) [m/s²]")
    axs[1].set_title("Time-Parameterization Profile")
    axs[1].set_xlabel("Time (s)")
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()
    print("-> Displaying Phase 1 verification plots (close plot window to finish test)...")
    plt.show()


if __name__ == "__main__":
    import numpy as np
    test_geometry_and_profile()
