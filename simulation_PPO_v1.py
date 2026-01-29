import os
import numpy as np
import pandas as pd
import supersuit as ss
from stable_baselines3 import PPO
from core.environment import BuildingEnv

config = {
    "model_name": "PPO_v1",
    "nb_zones": 3,
    "total_timesteps": 1e5, # Nombre de minutes d'entraînement
    "t_ext_scenario": 5.0,
    "time_steps_eval": 240    # Durée de la simulation finale
}

# 1. Création et préparation de l'environnement pour SB3
env = BuildingEnv(nb_zones=config["nb_zones"])
# On transforme l'env PettingZoo en un format compréhensible par l'IA (Vectorized Env)
env_train = ss.pettingzoo_env_to_vec_env_v1(env)
env_train = ss.concat_vec_envs_v1(env_train, num_vec_envs=1, num_cpus=1, base_class="stable_baselines3")

# 2. Configuration du "Cerveau" (PPO)
model = PPO(
    "MlpPolicy",           # Réseau de neurones classique
    env_train, 
    verbose=1,             # Affiche les progrès dans la console
    learning_rate=0.0003,  # Vitesse à laquelle l'IA ajuste ses poids
    n_steps=2048,
    device="auto"      # Nombre de pas avant de mettre à jour la stratégie
)

# 3. Entraînement
print(f"--- Début de l'entraînement de {config['model_name']} ---")
model.learn(total_timesteps=config["total_timesteps"], progress_bar=True)
model.save(f"models/{config['model_name']}")

# 4. Évaluation et Sauvegarde des résultats
print("--- Génération du CSV de performance ---")
obs, _ = env.reset()
data = []

for step in range(config["time_steps_eval"]):
    actions_dict = {}
    row = {"step": step, "target": env.target_temp}
    
    for agent in env.agents:
        # L'IA prédit maintenant l'action optimale au lieu d'utiliser Kp
        action, _ = model.predict(obs[agent], deterministic=True)
        actions_dict[agent] = action
        
        row[f"temp_{agent}"] = obs[agent][0]
        row[f"act_{agent}"] = action[0]

    obs, rewards, _, _, _ = env.step(actions_dict, t_ext=config["t_ext_scenario"])
    data.append(row)

os.makedirs("results", exist_ok=True)
filename = f"results/data_{config['model_name']}.csv"
pd.DataFrame(data).to_csv(filename, index=False)
print(f"Terminé ! Fichier créé : {filename}")