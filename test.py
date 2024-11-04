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
from stable_baselines3.common.callbacks import BaseCallback
from fenics import *
from petsc4py import PETSc
from mpi4py import MPI
import sys
import os
import torch
import matplotlib.pyplot as plt
from dolfin import plot
from stable_baselines3.common.logger import configure
from stable_baselines3.common.env_util import make_vec_env

# Désactiver les messages de log de FEniCS
set_log_level(LogLevel.ERROR)

# Désactiver les messages de log de PETSc
PETSc.Sys.popErrorHandler()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using ",device)

def generate_probes():
    L = 1; H = 0.41; r = 0.075; c_x = c_y = 0.2 
    #Fonction pour calculer la distance d'un point par rapport au centre du cylindre
    def distance_to_cylinder(x, y, c_x, c_y):
        return np.sqrt((x - c_x) ** 2 + (y - c_y) ** 2)

    # Générer des points de sonde en grille dans la zone rectangulaire (premier tiers du tube)
    probe_points = [];  tol=2e-2
    x_values = np.linspace(tol, 0.4 * L, 10)  # Repartir les points en X sur le premier tiers
    y_values = np.linspace(tol, H-tol, 10)          # Repartir les points en Y sur toute la hauteur
    tolerance = r + 0.02                      # Tolérance autour du cylindre (distance minimale)

    # Placer les points tout en évitant ceux proches du cylindre
    for x in x_values:
        for y in y_values:
            if distance_to_cylinder(x, y, c_x, c_y) > tolerance:
                probe_points.append([x, y])

    # Ajouter des points de sonde autour du cylindre
    theta_values = np.linspace(0, 2 * np.pi, 20)
    for theta in theta_values:
        x_c = c_x + (r + 0.02) * np.cos(theta)
        y_c = c_y + (r + 0.02) * np.sin(theta)
        probe_points.append([x_c, y_c])

    # Convertir les points de sonde en tableau numpy
    probe_points = np.array(probe_points)
    return probe_points

def run(steps, env, model):
    actions_taken=[] 
    # Manuellement entraîner le modèle sans utiliser model.learn()
    num_episodes = 2  # Nombre d'épisodes d'entraînement
    for episode in __builtins__.range(1, num_episodes):
        print(f"Début de l'épisode {episode}")
        obs = env.reset()
        done = False
        episode_reward = 0

        # Effectuer des étapes d'apprentissage jusqu'à la fin de l'épisode
        for i in __builtins__.range(steps):
            # Prédire l'action
            action, _ = model.predict(obs, deterministic = True)
            obs, reward, done, _ = env.step(action)
            #if not(i % 50):
                #env.unwrapped.envs[0].render()
            # Accumuler la récompense de l'épisode
            episode_reward += reward

            print(f"Action: {action}, Observation moy: {obs.mean()}, Récompense: {reward}, Done: {done}")
            actions_taken.append(action[0][0])
            
        print(f"Fin de l'épisode {episode} - Récompense totale: {episode_reward}")

        
        print(actions_taken)


    # Sauvegarder les résultats VTK pour visualisation
    #env.unwrapped.envs[0].export_to_vtk(num_episodes)
    env.unwrapped.envs[0].render()

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
    plt.grid(True)
    plt.title("Actions Taken by PPO Agent")
    plt.savefig('Actions Taken by PPO Agent.png')
    plt.show()
    # Fermer l'environnement
    env.close()

class StopTrainingOnEpisodeCountCallback(BaseCallback):
    def __init__(self, max_episodes, verbose=0):
        super(StopTrainingOnEpisodeCountCallback, self).__init__(verbose)
        self.max_episodes = max_episodes
        self.episode_count = 0

    def _on_step(self) -> bool:
        # Check if a new episode is done
        if self.locals["dones"][0]:
            self.episode_count += 1
        # Stop training if the max number of episodes is reached
        if self.episode_count >= self.max_episodes:
            print(f"Stopping training after {self.episode_count} episodes.")
            return False  # This will stop training
        return True  # Continue training

class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Custom callback that saves the model when the reward improves
    and logs training details to a CSV file.
    """

    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        if self.log_dir is not None:
            os.makedirs(self.log_dir, exist_ok=True)

        # File to save logs
        self.csv_file = os.path.join(self.log_dir, 'training_log.csv')

        # Write the header of the CSV file
        with open(self.csv_file, 'w') as f:
            f.write('Step,Mean Reward\n')

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            # Retrieve training reward
            mean_reward = np.mean(self.locals['rollout_buffer'].rewards)
            step = self.num_timesteps

            # Log to CSV
            with open(self.csv_file, 'a') as f:
                f.write(f'{step},{mean_reward}\n')

            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                # Optionally, save the model
                self.model.save(f"{self.log_dir}/best_model.zip")

        return True

class CustomTensorboardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(CustomTensorboardCallback, self).__init__(verbose)

    def _on_step(self):
        # Access some internal state data
        current_rewards = np.mean(self.model.rollout_buffer.rewards)
        current_actions = np.mean(self.model.rollout_buffer.actions)

        # Use the logger to record custom metrics
        self.logger.record('custom/current_rewards', current_rewards)
        self.logger.record('custom/current_actions', current_actions)

        return True

# Désactiver les messages MPI pour les processus autres que le maître
if MPI.COMM_WORLD.rank != 0:
    sys.stdout = open(os.devnull, 'w')


probe_points = generate_probes()

def test(n):
    geometry_params = {
        'mesh': 'mesh_with_cylinder_and_jets.xml',  # Le fichier XML du maillage
        'facets': 'facets_with_cylinder_and_jets.xml',  # Le fichier des facettes
        'length': 1,
        'front_distance': 0.2,
        'bottom_distance': 0.2,
        'jet_radius': 0.05,
        'width': 0.41,
        'cylinder_size': 0.01,
        'jet_positions': [90, 270],
        'jet_width': 10
    }

    flow_params = {
        'mu': 2E-4,
        'rho': 0.135,
        'inflow_profile': None
    }

    output_params = {
        'locations': probe_points,
        'probe_type': 'pressure'
    }

    optimization_params = {
        'num_steps_in_pressure_history': 1,
        'min_value_jet_MFR': 1e-3,
        'max_value_jet_MFR': 1e-2,
        'random_start': False
    }
    
    action_interval = 1000
    solver_params = {
        'dt': 0.05,
        'action_interval' : action_interval
    }
    # Fonction pour créer l'environnement
    env2d = Env2DCylinder(
            geometry_params=geometry_params,
            flow_params=flow_params,
            solver_params=solver_params,
            output_params=output_params,
            optimization_params=optimization_params,
            num_steps=5000,
            path_root='geometry'
        )
    env = make_vec_env(lambda: env2d, n_envs=1)

    ###log
    log_dir = "./ppo_training_logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    #callback = SaveOnBestTrainingRewardCallback(check_freq=1200, log_dir=log_dir)

    ###Tensorboard
    # Create a logger for TensorBoard
    log_dir_board = "./ppo_tensorboard/"
    if not os.path.exists(log_dir_board):
        os.makedirs(log_dir_board)
    new_logger = configure(log_dir_board, ["tensorboard"])
    # Set the logger
    #model.set_logger(new_logger)
    callback=CustomTensorboardCallback()

    model_dir = "./ppo_model"

    model = PPO.load(f"{model_dir}/ppo_model.zip", env=env, device='cuda')

    run(n, env, model)

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    test(100)