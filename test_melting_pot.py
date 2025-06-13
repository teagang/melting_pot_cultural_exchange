from shimmy import MeltingPotCompatibilityV0

env = MeltingPotCompatibilityV0(substrate_name="allelopathic_harvest__open", render_mode="human")

from shimmy import MeltingPotCompatibilityV0
from shimmy.utils.meltingpot import load_meltingpot

env = load_meltingpot("allelopathic_harvest__open")
env = MeltingPotCompatibilityV0(env, render_mode="human")

observations = env.reset()
while env.agents:
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}
    observations, rewards, terminations, truncations, infos = env.step(actions)
    print(rewards) # note from calude they are doing random actions not learning
env.close()