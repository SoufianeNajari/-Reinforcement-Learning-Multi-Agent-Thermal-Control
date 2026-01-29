import numpy as np
T_EXT = 5.0  # Température extérieure constante pour la simulation


class ThermalModel:
    def __init__(self, nb_zones, start_temp, R_val, C_val, R_inter, max_power, dt):
        self.nb_zones = nb_zones
        self.R = R_val
        self.C = C_val
        self.R_inter = R_inter
        self.max_power = max_power
        self.dt = dt
        self.start_temp = start_temp
        self.temp_interne = np.full(nb_zones, self.start_temp)
        self.t_ext = T_EXT


    def step(self, actions, t_ext):
        # actions est un array numpy avec des valeurs entre -1 (froid max) et 1 (chaud max)
        q_hvac = actions * self.max_power
        
        # 1. Flux avec l'extérieur (Loi de Fourier)
        flux_ext = (t_ext - self.temp_interne) / self.R
        
        # 2. Flux entre les zones (Inter-zone)
        flux_inter = np.zeros(self.nb_zones)
        for i in range(self.nb_zones):
            # Somme des échanges avec toutes les autres zones
            # On considère que chaque zone échange avec toutes les autres avec la même résistance R_inter
            diff_temp = self.temp_interne - self.temp_interne[i]
            flux_inter[i] = np.sum(diff_temp) / self.R_inter

        # 3. Application de la méthode d'Euler
        # dT/dt = (Flux_total) / C
        dT = (flux_ext + q_hvac + flux_inter) / self.C * self.dt
        self.temp_interne += dT
        
        return self.temp_interne.copy()

    def reset(self):
        self.temp_interne = np.full(self.nb_zones, self.start_temp)
        return self.temp_interne.copy()