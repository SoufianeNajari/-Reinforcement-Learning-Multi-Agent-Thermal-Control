import numpy as np
from core.environment import BuildingEnv

config = {
    "adj_matrix": [[0, 1, 1], [1, 0, 0], [1, 0, 0]],
    "expo_ext": [0.0, 1.0, 1.0],
    "t_ext_offset": [0, -2, 2],
    "start_temp": 19.0,
    "R_val": 0.1,
    "C_val": 1e6,
    "R_inter": 0.5,
    "max_power": 2000,
    "dt": 60
}

env = BuildingEnv(config)
obs, info = env.reset()

print(f"Agents détectés : {env.possible_agents}")
print(f"Première observation : {obs['zone_0']}")

actions = {agent: np.array([0.5], dtype=np.float32) for agent in env.possible_agents}
obs, rewards, term, trunc, infos = env.step(actions)

print(f"Récompenses après 1 step : {rewards}")
print("Test réussi !")