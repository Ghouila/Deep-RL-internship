import warnings
warnings.filterwarnings("ignore")


import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
from flow_solver import FlowSolver
import os
from fenics import *
from mshr import *
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

class Env2DCylinder(gym.Env):
    metadata = {'render.modes': ['human', 'plot']}

    def __init__(self, path_root, geometry_params, flow_params, solver_params, output_params, optimization_params, reward_function='plain_drag', number_steps_execution=1):
        print("Appel de init")
        super().__init__()
        self.output_dir = "./output"
        self.path_root = path_root
        self.geometry_params = geometry_params
        self.flow_params = flow_params
        self.solver_params = solver_params
        self.output_params = output_params
        self.optimization_params = optimization_params
        self.reward_function = reward_function
        self.number_steps_execution = number_steps_execution
        self.flow_solver = FlowSolver(flow_params, geometry_params, solver_params) 
        #self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(len(geometry_params["jet_positions"]),), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(len(geometry_params["jet_positions"]),), dtype=np.float32)
        self.points = output_params['locations']
        
        # Stockage des résultats physiques
        self.num_steps = 10  
        self.C_D = np.zeros(self.num_steps)  
        self.C_L = np.zeros(self.num_steps)
        self.reward_list = np.zeros(self.num_steps)
        self.p_diff = np.zeros(self.num_steps)
        self.t_u = np.zeros(self.num_steps)
        self.t_p = np.zeros(self.num_steps)
        self.step_count = 0
        """
    def set_num_steps(self, num_steps):
    
        print(f"Setting num_steps to {num_steps}")
        self.num_steps = num_steps
        """
    def reset(self, *, seed: int = None, options: dict = None):
        """
        Réinitialise l'environnement pour un nouvel épisode de simulation.
        Accepte un paramètre seed pour la reproductibilité.
        """
        print("Appel de reset")
        # Si un seed est fourni, initialisez le générateur de nombres aléatoires avec.
        super().reset(seed=seed)

        # Si seed est fourni, initialisez le générateur de nombres aléatoires avec
        if seed is not None:
            self._np_random, seed = gym.utils.seeding.np_random(seed)

        # Réinitialiser le flow_solver avec des jets à 0.0
        #self.flow_solver.evolve([0.0 for _ in range(len(self.action_space.shape))])
        self.flow_solver.evolve([0.0])

        # Obtenir les nouvelles observations
        observation = self._get_observation()

        # Réinitialiser le compteur d'exécution des étapes
        self.number_steps_execution = 0
        self.step_count = 0

        # Retourner l'observation et les informations optionnelles
        return observation, {}



    def step(self, action):
        print("Appel de step")
        # Effectuer l'action (contrôle des jets)
        self.flow_solver.evolve(action)
        
        # Calculer la traînée et la portance en tant que scalaires
        drag_coeff = self.flow_solver.compute_drag()  # Ceci doit retourner un scalaire
        lift_coeff = self.flow_solver.compute_lift()  # Ceci doit retourner un scalaire

        # Calcul de la pression aux points spécifiques (avant et arrière de l'obstacle)
        p_front = self.flow_solver.sample_pressure_at_location(self.output_params["locations"][0])
        p_back = self.flow_solver.sample_pressure_at_location(self.output_params["locations"][1])

        # Mise à jour des résultats physiques
        self.C_D[self.step_count-1] = drag_coeff
        self.C_L[self.step_count-1] = lift_coeff
        self.p_diff[self.step_count-1] = p_front - p_back
        self.t_u[self.step_count-1] = self.step_count  # Remplacer par une gestion du temps adaptée
        self.t_p[self.step_count-1] = self.step_count - self.solver_params['dt'] / 2

        print("step count / toatal time steps: ", self.step_count, "/",self.num_steps)

        # Avancement du compteur de pas de simulation
        self.step_count += 1

        # Renvoie l'observation et la récompense (si applicable)
        observation = self._get_observation()
        reward = self._compute_reward()
        done = self.step_count >= self.num_steps
        print("done = ", done)
        
        info = {}
        truncated =0
        return observation, reward, done,truncated, info

    def _get_observation(self):
        """
        Récupère les observations depuis les sondes (pression).
        """
        observations = []
        for location in self.output_params["locations"]:
            pressure_value = self.flow_solver.sample_pressure_at_location(location)
            observations.append(pressure_value)
        
        return np.array(observations, dtype=np.float32)


    

    def _compute_reward(self):
            """
            Calcule la récompense en fonction de la traînée et de la portance.
            """
            drag = self.flow_solver.compute_drag()
            lift = self.flow_solver.compute_lift()
            reward = -drag + 0.2 * lift  # Minimiser la traînée tout en prenant en compte la portance
            self.reward_list[self.step_count - 1] = reward
            print("Time step: ",self.step_count -1, "Drag calculated:", drag, "Lift calculated:", lift, "Reward calculated:", reward)
            return reward

    def export_to_vtk(self, episode):
        """
        Exporte la vitesse et la pression pour la visualisation avec Paraview.
        """
        # Définir les noms des fichiers de sortie au format .xdmf
        xdmf_velocity_file = os.path.join(self.output_dir, f"velocity_{episode}.xdmf")
        xdmf_pressure_file = os.path.join(self.output_dir, f"pressure_{episode}.xdmf")

        # Affichage et sauvegarde du champ de pression
        fig = plot(self.flow_solver.p_, "Champs de pression p")
        fig.set_cmap("rainbow")
        plt.colorbar(fig)
        plt.savefig('pression_p.png')
        plt.show()  # Affiche la figure avant de la fermer
        plt.close()  # Ferme la figure après l'affichage

        # Affichage et sauvegarde du champ de vitesse
        fig = plot(self.flow_solver.u_, "Champs de vitesse u")
        fig.set_cmap("rainbow")
        plt.colorbar(fig)
        plt.savefig('champs_de_vitesse_u.png')
        plt.show()  # Affiche la figure avant de la fermer
        plt.close()  # Ferme la figure après l'affichage

        # Créer le répertoire de sortie s'il n'existe pas
        self.output_dir = os.path.join(os.getcwd(), 'output')
        os.makedirs(self.output_dir, exist_ok=True)

        # Sauvegarder les fichiers XDMF pour la vitesse et la pression
        with XDMFFile(MPI.COMM_WORLD, xdmf_velocity_file) as xdmf_velocity:
            xdmf_velocity.write(self.flow_solver.u_, episode)  # Enregistrer le champ de vitesse

        with XDMFFile(MPI.COMM_WORLD, xdmf_pressure_file) as xdmf_pressure:
            xdmf_pressure.write(self.flow_solver.p_, episode)  # Enregistrer le champ de pression

        print(f"Velocity and Pressure for episode {episode} written to XDMF.")
        pass


    def close(self):
        """
        Nettoie et ferme l'environnement.
        """
        # Si le solveur est actif, le fermer
        if hasattr(self, 'flow_solver'):
            del self.flow_solver  # Libérer la mémoire associée au solveur

        # Fermer tous les graphiques ou processus associés
        plt.close('all')  # Fermer les figures ouvertes de Matplotlib

        print("Environment successfully closed.")

        
