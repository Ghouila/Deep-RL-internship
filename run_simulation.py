import warnings
warnings.filterwarnings("ignore")

import gymnasium as gym
from Env2DCylinder import Env2DCylinder
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
import numpy as np
from mpi4py import MPI
import matplotlib.pyplot as plt
from fenics import set_log_level, LogLevel
from torch import *
from stable_baselines3.common.callbacks import BaseCallback
from fenics import *
from petsc4py import PETSc
from mpi4py import MPI
import sys
import os

# Désactiver les messages de log de FEniCS
set_log_level(LogLevel.ERROR)

# Désactiver les messages de log de PETSc
PETSc.Sys.popErrorHandler()

# Désactiver les messages MPI pour les processus autres que le maître
if MPI.COMM_WORLD.rank != 0:
    sys.stdout = open(os.devnull, 'w')

total_timesteps = 10

def run_simulation():
    geometry_params = {
        'mesh': 'mesh_with_cylinder_and_jets.xml',  # Le fichier XML du maillage
        'facets': 'facets_with_cylinder_and_jets.xml',  # Le fichier des facettes
        'length': 2.2,
        'front_distance': 0.2,
        'bottom_distance': 0.2,
        'jet_radius': 0.05,
        'width': 0.41,
        'cylinder_size': 0.01,
        'jet_positions': [90, 270],
        'jet_width': 10
    }

    flow_params = {
        'mu': 1E-3,
        'rho': 1,
        'inflow_profile': None
    }

    solver_params = {
        'dt': 0.0005
    }

    output_params = {
        'locations': np.array([[0.1, 0.0], [0.2, 0.0]]),
        'probe_type': 'pressure'
    }

    optimization_params = {
        'num_steps_in_pressure_history': 1,
        'min_value_jet_MFR': 1e-3,
        'max_value_jet_MFR': 1e-2,
        'random_start': False
    }

    # Fonction pour créer l'environnement
    def create_env():
        env = Env2DCylinder(
            geometry_params=geometry_params,
            flow_params=flow_params,
            solver_params=solver_params,
            output_params=output_params,
            optimization_params=optimization_params,
            reward_function='plain_drag',
            number_steps_execution=1,
            path_root='geometry'
        )
        return Monitor(env)
    
    env = DummyVecEnv([create_env])

    # Créer un modèle PPO avec GPU (cuda)
    #model = PPO("MlpPolicy", env, verbose=1, device='cuda')
    model = PPO.load("ppo_model_episode.zip", device='cuda')
    actions_taken=[] 
    # Manuellement entraîner le modèle sans utiliser model.learn()
    num_episodes = 7  # Nombre d'épisodes d'entraînement
    for episode in __builtins__.range(num_episodes):
        print(f"Début de l'épisode {episode}")
        obs = env.reset()
        done = False
        episode_reward = 0

        # Effectuer des étapes d'apprentissage jusqu'à la fin de l'épisode
        while not done:
            # Prédire l'action
            action, _ = model.predict(obs)
            obs, reward, done, _ = env.step(action)

            # Accumuler la récompense de l'épisode
            episode_reward += reward

            print(f"Action: {action}, Observation: {obs}, Récompense: {reward}, Done: {done}")
            actions_taken.append(action[0][0])
            
        print(f"Fin de l'épisode {episode} - Récompense totale: {episode_reward}")

        # Sauvegarder le modèle après chaque épisode
        model.save(f"ppo_model_episode.zip")
        print(f"Modèle sauvegardé pour l'épisode {episode}")
        print(actions_taken)

    # Sauvegarder les résultats VTK pour visualisation
    env.unwrapped.envs[0].export_to_vtk(num_episodes)

    # Visualiser la distribution des actions
    plt.hist(actions_taken, bins=20)
    plt.xlabel("Action Value")
    plt.ylabel("Frequency")
    plt.title("Distribution of Actions Taken by PPO Agent")
    plt.savefig('Distribution of Actions.png')
    plt.show()

    #
    plt.plot(actions_taken)
    plt.ylabel("Action Value")
    plt.xlabel("temps")
    plt.title("Actions Taken by PPO Agent")
    plt.savefig('Actions Taken by PPO Agent.png')
    plt.show()
    # Fermer l'environnement
    env.close()

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    run_simulation()
