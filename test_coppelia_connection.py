from coppeliasim_zmqremoteapi_client import RemoteAPIClient


def main():
    print("Connecting to CoppeliaSim...")

    client = RemoteAPIClient()
    sim = client.require("sim")

    print("Connected successfully!")
    print("CoppeliaSim version:",
          sim.getInt32Param(sim.intparam_program_version))

    sim.startSimulation()

    print("Simulation started.")

    while sim.getSimulationTime() < 2.0:
        print(f"Simulation time: {sim.getSimulationTime():.2f}s")
        sim.step()

    sim.stopSimulation()

    print("Simulation stopped.")
    print("Connection test PASSED.")


if __name__ == "__main__":
    main()
