# Reinforcement Learning Multi-Agent Thermal Control

Projet de contrôle thermique multi-zone d’un bâtiment avec :
- un environnement multi-agent (PettingZoo),
- un modèle physique simplifié de transferts thermiques,
- des approches de contrôle (Proportionnel, PI, PPO).

## Structure du projet

- `core/` : environnement et modèle thermique (`environment.py`, `building_model.py`)
- `simulations/` : scripts d’entraînement et de simulation
- `config.py` : paramètres globaux (consigne, météo, physique, récompense)
- `model_benchmark.py` : évaluation d’un modèle entraîné
- `model_visualisation.py` : visualisation des résultats
- `graphs/`, `logs/`, `Figures/` : résultats et figures

## Prérequis

Python 3.10+ recommandé.

Installer les dépendances principales :

```bash
pip install numpy pandas matplotlib gymnasium pettingzoo supersuit stable-baselines3
```

## Utilisation

### 1) Entraîner un agent PPO

```bash
python simulations/simulation_PPO_v5.py
```

Le modèle est sauvegardé dans `models/`.

### 2) Lancer une simulation de référence

Contrôleur proportionnel :

```bash
python simulations/simulation_Proportionnel.py
```

Contrôleur PI :

```bash
python simulations/simulation_PI.py
```

### 3) Évaluer un modèle

```bash
python model_benchmark.py
```

### 4) Visualiser les résultats

```bash
python model_visualisation.py
```

## Configuration

Les paramètres principaux sont dans `config.py` :
- température cible,
- durée d’épisode,
- température extérieure d’entraînement (hiver/été),
- paramètres thermiques du bâtiment,
- poids de la récompense (confort vs coût énergétique).
