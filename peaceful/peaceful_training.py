import gymnasium as gym
from stable_baselines3 import PPO # We'll import this, though not used in the env setup directly yet
from shimmy.meltingpot_compatibility import MeltingPotCompatibilityV0 # Corrected import for shimmy API
from shimmy.utils.meltingpot import load_meltingpot # Utility for loading Melting Pot environments

def make_allelopathic_harvest_env(
    num_agents: int = 2,
    max_episode_length: int = 100,
    render_mode="human"
) -> gym.Env:
    """
    Creates and returns a Gymnasium-compatible Melting Pot allelopathic_harvest environment
    using shimmy.MeltingPotCompatibilityV0.

    Args:
        num_agents: The number of agents in the environment (note: MeltingPotCompatibilityV0
                    handles agent roles internally based on the loaded substrate).
        max_episode_length: The maximum number of steps an episode can run.
        render_mode: The render mode for the environment.
    Returns:
        A Gymnasium environment instance of allelopathic_harvest.
    """
    # Load the Melting Pot environment using the utility function
    # The utility function handles the underlying dm_env and substrate configuration.
    mp_env = load_meltingpot(substrate_name="allelopathic_harvest__open")

    # Wrap the Melting Pot environment with shimmy.MeltingPotCompatibilityV0 for Gymnasium compatibility.
    # num_agents and add_individual_players are handled by MeltingPotCompatibilityV0
    # based on the loaded substrate and its default roles.
    env = MeltingPotCompatibilityV0(
        env=mp_env,
        max_cycles=max_episode_length,
        render_mode=render_mode
    )

    return env

if __name__ == "__main__":
    print("Setting up the allelopathic_harvest environment with MeltingPotCompatibilityV0...")

    # Create an instance of the environment
    # num_agents is less critical here as MeltingPotCompatibilityV0 determines agents
    # from the loaded substrate. max_episode_length is passed.
    env = make_allelopathic_harvest_env(num_agents=2, max_episode_length=1000)

    print(f"Environment created: {env}")
    # With MeltingPotCompatibilityV0, observation_space and action_space are methods
    # that take an agent ID. We can't print a single unified space like before
    # for all agents directly from env.observation_space.
    # The environment will expose `env.agents` which is a list of current agents.
    print(f"Available agents at reset: {env.agents}")

    # Example of interacting with the environment
    print("\nStarting an example episode...")
    observations, info = env.reset()
    done = False
    total_rewards = {agent_id: 0.0 for agent_id in env.agents} # Initialize rewards for current agents
    step_count = 0

    print(f"Initial observations keys: {observations.keys()}")

    while not done and step_count < env.spec.max_episode_steps:
        actions = {}
        for agent_id in env.agents: # Iterate through current agents for sampling actions
            actions[agent_id] = env.action_space(agent_id).sample()

        observations, rewards, terminations, truncations, infos = env.step(actions)

        # Check if all agents have terminated or truncated.
        # MeltingPotCompatibilityV0 returns dicts for terminations and truncations.
        done = all(terminations.values()) or all(truncations.values())
        step_count += 1

        for agent_id, reward in rewards.items():
            total_rewards[agent_id] += reward

        if step_count % 100 == 0:
            print(f"Step {step_count}: Rewards = {rewards}")

    print(f"\nEpisode finished after {step_count} steps.")
    print(f"Total rewards per agent: {total_rewards}")

    env.close()
    print("Environment closed.")

    print("\nReady for Stable Baselines3 training!")
    print("When using SB3, you would typically adapt the environment further or use a multi-agent RL framework.")
    print("MeltingPotCompatibilityV0 provides a Gymnasium API where each agent has its own observation_space(agent_id) and action_space(agent_id).")
    print("For SB3, you might need to flatten or concatenate observations/actions, or use a custom policy that handles multi-agent inputs/outputs.")
    # Example placeholder for SB3 training (conceptual)
    # from stable_baselines3.common.env_util import make_vec_env
    # Note: make_vec_env might not directly support the per-agent spaces of MeltingPotCompatibilityV0
    # without a custom wrapper or a multi-agent RL library.
    # vec_env = make_vec_env(make_allelopathic_harvest_env, n_envs=4,
    #                        env_kwargs={'max_episode_length': 1000, 'render_mode': None}) # render_mode=None for vectorized envs
    # model = PPO("MultiInputPolicy", vec_env, verbose=1)
    # model.learn(total_timesteps=10000)