import gymnasium as gym
import shimmy
import pprint # This will help print dictionary outputs nicely

try:
    # Attempt to make the environment.
    # The exact ID might vary slightly, but this is the most common for Melting Pot environments via shimmy.
    env = gym.make("dm_control/AllelopathicHarvest-v0") 
    print("Environment created successfully!")

    # Get the action space specification
    action_space = env.action_space
    print("\n--- Action Space Details ---")
    print(f"Action Space Type: {type(action_space)}")
    print(f"Action Space Description: {action_space}")

    # For discrete action spaces (most common for control signals like "beam type")
    if isinstance(action_space, gym.spaces.Discrete):
        print("\nThis is a Discrete action space.")
        print("You'll need to consult the Melting Pot documentation or source code for mapping integer IDs to actions.")
        print("Typically, actions are represented by integers from 0 up to num_actions - 1.")
        print(f"Number of discrete actions: {action_space.n}")

    # For dictionary action spaces (common in multi-component actions)
    elif isinstance(action_space, gym.spaces.Dict):
        print("\nThis is a Dictionary action space. Actions are usually named.")
        print("Inspecting each component:")
        for key, spec in action_space.spaces.items():
            print(f"\n  Key: '{key}'")
            print(f"    Type: {type(spec)}")
            print(f"    Description: {spec}")
            if hasattr(spec, 'high') and hasattr(spec, 'low'):
                print(f"    Range: [{spec.low}, {spec.high}]")
            if hasattr(spec, 'n'): # For Discrete spaces within a Dict
                print(f"    Number of discrete values: {spec.n}")

            # This is where you'll look for something like 'beam_type' or 'action_type'
            # If you see a discrete action here, the values (e.g., 0, 1, 2) will correspond to different beams.
            # You'll still need to map the *meaning* of those numbers to "punishment beam" from documentation.
            if key == "global_interaction": # This is a common name for such an action in Melting Pot
                print("\n  *** Potential punishment beam action found here! ***")
                print("  You will likely need to check the Melting Pot source or docs for what specific integer value corresponds to punishment.")


    # For Box action spaces (continuous, often for movement or strength)
    elif isinstance(action_space, gym.spaces.Box):
        print("\nThis is a Box (continuous) action space.")
        print(f"Shape: {action_space.shape}")
        print(f"Low values: {action_space.low}")
        print(f"High values: {action_space.high}")
        print("If the punishment action is continuous, it might be represented by a specific range within one of these dimensions.")

    # Always close the environment when done
    env.close()
    print("\nEnvironment closed.")

except Exception as e:
    print(f"An error occurred: {e}")
    print("Please ensure you have 'shimmy' and 'dm_control' (which Melting Pot depends on) installed correctly.")
    print("You might need: pip install gymnasium shimmy dm_control[mujoco] meltingpot")
    print("Note: dm_control[mujoco] requires a MuJoCo license and setup if you don't have it already, but for just inspecting action space, it might not be strictly necessary, depending on the environment's full initialization.")