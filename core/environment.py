import functools
import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv
from .building_model import ThermalModel


class BuildingEnv(ParallelEnv):
    # metadata indispensable pour Stable-Baselines3 et SuperSuit
    metadata = {
        "name": "building_thermal_v0",
        "render_modes": [None]
    }

    def __init__(self, nb_zones=3, render_mode=None):
        self.render_mode = render_mode
        self.possible_agents = [f"zone_{i}" for i in range(nb_zones)]
        
        # Consigne de température cible
        self.target_temp = 21.0
        
        # Modèle physique RC
        self.model = ThermalModel(
            nb_zones=nb_zones,
            start_temp=19.0,
            R_val=0.1,         # Isolation
            C_val=1e6,         # Inertie (1 000 000 J/K)
            R_inter=0.5,       # Échanges entre zones
            max_power=2000,    # Puissance PAC
            dt=60              # 1 minute par pas
        )
        
        self.t_ext = self.model.t_ext  # Température extérieure par défaut

        # Compteur interne pour limiter la durée des épisodes
        self.current_step = 0

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        # [Température zone, Température extérieure]
        return spaces.Box(low=0, high=50, shape=(2,), dtype=np.float32)

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        # Action continue entre -1 (froid) et 1 (chaud)
        return spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        # Réinitialisation PettingZoo
        self.agents = self.possible_agents[:]
        self.current_step = 0
        
        # Reset de la physique
        temps = self.model.reset()
        
        # Gestion de t_ext au démarrage
        t_ext = self.t_ext
        if options and "t_ext" in options:
            t_ext = options["t_ext"]
            
        observations = {
            agent: np.array([temps[i], t_ext], dtype=np.float32) for i, agent in enumerate(self.agents)
        }
        return observations, {}

    def step(self, actions, t_ext=None):
        # Sécurité si t_ext n'est pas fourni par l'IA (Training)
        if t_ext is None:
            t_ext = self.t_ext
            
        self.current_step += 1
        
        # 1. Calcul de la physique
        act_array = np.array([actions[agent][0] for agent in self.agents])
        new_temps = self.model.step(act_array, t_ext)
        
        # 2. Préparation des observations
        observations = {
            agent: np.array([new_temps[i], t_ext], dtype=np.float32) 
            for i, agent in enumerate(self.agents)
        }
        
        # 3. Calcul de la récompense (Reward)
        rewards = {}
        for i, agent in enumerate(self.agents):
            # Plus on est loin de 21°C, plus le score est mauvais
            error = abs(new_temps[i] - self.target_temp)
            rewards[agent] = -(error **2)

        # 4. Conditions d'arrêt (Truncation après 24h)
        # Indispensable pour que SB3 affiche 'ep_rew_mean'
        duree_max_atteinte = self.current_step >= 1440
        
        # Indispensable : On définit l'état pour TOUS les agents possibles
        terminations = {agent: False for agent in self.possible_agents}
        truncations = {agent: duree_max_atteinte for agent in self.possible_agents}
        infos = {agent: {} for agent in self.possible_agents}
        
        # Logique PettingZoo : si c'est fini, on vide la liste self.agents
        if duree_max_atteinte:
            self.agents = []

        return observations, rewards, terminations, truncations, infos