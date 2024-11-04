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

    def __init__(self, path_root, geometry_params, flow_params, solver_params, output_params, optimization_params,num_steps):
        print("Appel de init")
        super().__init__()
        self.output_dir = "./output"
        self.path_root = path_root
        self.geometry_params = geometry_params
        self.flow_params = flow_params
        self.solver_params = solver_params
        self.output_params = output_params
        self.optimization_params = optimization_params
        self.action_interval = solver_params['action_interval']
        
        self.flow_solver = FlowSolver(flow_params, geometry_params, solver_params, num_steps) 

        #self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(len(geometry_params["jet_positions"]),), dtype=np.float32)
        mesh = Mesh(geometry_params['mesh'])
        V = VectorFunctionSpace(mesh, 'P', 2)
        dof = V.dim()  # Number of degrees of freedom (nodes)
        print(dof)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        """
        self.action_space = spaces.Dict({
            "Q_debit": spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
            "source": spaces.Box(low=-1.0, high=1.0, shape=dof, dtype=np.float32)  # As a flattened array
        })
        """
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(len(output_params["locations"]),), dtype=np.float32)
        self.points = output_params['locations']
        
        # Stockage des param physiques
        self.num_steps = num_steps
        self.num_eps=1  
        self.time_step = 1
        self.C_D = np.zeros(self.num_steps)  
        self.C_L = np.zeros(self.num_steps)
        self.reward_list = np.zeros(self.num_steps)
        self.p_diff = np.zeros(self.num_steps)
        self.t_u = np.zeros(self.num_steps)
        self.t_p = np.zeros(self.num_steps)
        self.step_count = 1
        self.action_nbr = 0
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

        # Réinitialiser le flow_solver avec des jets à 0.0 (remettre toutes les conditions initiales)
        self.flow_solver = FlowSolver(self.flow_params, self.geometry_params, self.solver_params, self.num_steps)
        
        # Réinitialiser le solveur pour revenir aux conditions initiales
        self.flow_solver.evolve([0.0],int(self.num_steps/self.action_interval), self.action_nbr, action_done = False )  # Vous pourriez définir des conditions initiales spécifiques ici

        # Réinitialiser les listes de stockage des résultats
        self.C_D = np.zeros(self.num_steps)  
        self.C_L = np.zeros(self.num_steps)
        self.C_D_cumul = np.zeros(self.num_steps)  
        self.C_L_cumul = np.zeros(self.num_steps)  
        self.reward_list =[] 
        self.p_diff = np.zeros(self.num_steps)
        self.t_u = np.zeros(self.num_steps)
        self.t_p = np.zeros(self.num_steps)
        print("Episode:",self.num_eps)
        # Réinitialiser le compteur d'exécution des étapes
        self.step_count = 1
        observation = self._get_observation()

        # Retourner l'observation et les informations optionnelles
        return observation, {}


    def step(self, action):
        print("Appel de step")
        # Effectuer l'action (contrôle des jets)
        action_done = False
        if self.step_count % self.action_interval == 0:
            print("PPO action taken")
             action_done = True;  self.flow_solver.evolve(action,int(self.num_steps/self.action_interval), self.action_nbr, action_done);   self.action_nbr +=1
        else:
            self.flow_solver.evolve([0.0],int(self.num_steps/self.action_interval), self.action_nbr, action_done)

        # Calcul de la pression aux points spécifiques (avant et arrière de l'obstacle)
        p_front = self.flow_solver.sample_pressure_at_location(self.output_params["locations"][0])
        p_back = self.flow_solver.sample_pressure_at_location(self.output_params["locations"][1])

        # Mise à jour des résultats physiques
        
        self.p_diff[self.step_count-1] = p_front - p_back
        self.t_u[self.step_count-1] = self.step_count  # Remplacer par une gestion du temps adaptée
        self.t_p[self.step_count-1] = self.step_count - self.solver_params['dt'] / 2
        print("Episode:",self.num_eps)
        print("step count / toatal time steps: ", self.step_count, "/",self.num_steps)

        # Renvoie l'observation et la récompense (si applicable)
        observation = self._get_observation()
        reward = self._compute_reward(action_done)

        # Avancement du compteur de pas de simulation
        self.step_count += 1
        self.time_step += 1
        
        done = False
        # Low varaince in env state 
        velocity_variance = np.var([self.flow_solver.sample_pressure_at_location(loc) for loc in self.output_params['locations']])
        pressure_variance = np.var([self.flow_solver.sample_pressure_at_location(loc) for loc in self.output_params['locations']])
        print("velocity_variance: ", velocity_variance, " pressure_variance: ",pressure_variance)
        if velocity_variance < 1e-3 and pressure_variance < 1e-3:
            done = True
        # No change in Cd and Cl
        if self.step_count > 10:
            drag_avg_change = np.mean(np.abs(np.diff(self.C_D[self.step_count-10:self.step_count])))
            lift_avg_change = np.mean(np.abs(np.diff(self.C_L[self.step_count-10:self.step_count])))
            if drag_avg_change < 1e-2 and lift_avg_change < 1e-2:
                done = True
        # Surpassed step count
        if self.step_count == (self.num_steps):
            done = True

        self.num_eps+=1 * done
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

    # Updated reward calculation based on time-averaged drag and lift coefficients
    def _compute_reward(self, action_done):
        """
        Calcule la récompense en fonction de la traînée et de la portance moyennes temporelles.
        """
        # Compute the cumulative sum of drag and lift coefficients up to the current step
        drag = self.flow_solver.compute_drag()
        lift = self.flow_solver.compute_lift()
        cumulative_C_D = np.sum(self.C_D[:self.step_count]) / self.step_count
        cumulative_C_L = np.sum(self.C_L[:self.step_count]) / self.step_count
        if action_done:
            self.C_D[self.step_count-1] = drag
            self.C_L[self.step_count-1] = lift
            self.C_D_cumul[self.step_count-1] = cumulative_C_D
            self.C_L_cumul[self.step_count-1] = cumulative_C_L

        # Compute the reward based on these averages (you can adjust this as needed)
        reward = -cumulative_C_D + 0.2 * cumulative_C_L  # Minimiser la traînée tout en prenant en compte la portance

        # Store the reward in the reward list
        self.reward_list.append(reward)
        #print(f"Time step: {self.step_count - 1},drag: {drag}, lift: {lift} Cumulative Drag: {cumulative_C_D}, Cumulative Lift: {cumulative_C_L}, Reward: {reward}")
        return reward

    
    def render(self, mode='human'):
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

        u_x, u_y = self.flow_solver.u_.split(deepcopy=True)

        # Create a figure with two subplots, one for each component
        fig, axs = plt.subplots(2, 1, figsize=(10, 8))

        # Define the levels to ensure consistency
        levels = np.linspace(-2, 2, 10)  # Adjusted to show possible full value range

        # Plot the u_x component
        plt.sca(axs[0])  # Set the active subplot to the first one
        p1 = plot(u_x, title="Composante u_x (Vitesse en X)")
        
        plt.colorbar(p1, ax=axs[0])
        axs[0].set_title("Composante u_x (Vitesse en X)")
        axs[0].set_xlabel("X")
        axs[0].set_ylabel("Y")

        # Plot the u_y component
        plt.sca(axs[1])  # Set the active subplot to the second one
        p2 = plot(u_y, title="Composante u_y (Vitesse en Y)")
        
        plt.colorbar(p2, ax=axs[1])
        axs[1].set_title("Composante u_y (Vitesse en Y)")
        axs[1].set_xlabel("X")
        axs[1].set_ylabel("Y")

        # Save and display the figure
        plt.tight_layout()
        plt.savefig('composantes_champs_de_vitesse_u.png')
        plt.show()


        fig, axs = plt.subplots(2, 2, figsize=(12, 10))

        # Première ligne : C_D et C_L
        # Plot C_D sur le premier subplot (ligne 1, colonne 1)
        color = 'tab:red'
        axs[0, 0].set_xlabel('Time Steps')
        axs[0, 0].set_ylabel('C_D (Drag Coefficient)', color=color)
        axs[0, 0].plot(self.C_D[:self.step_count-1], color=color, label='C_D (Drag Coefficient)')
        axs[0, 0].tick_params(axis='y', labelcolor=color)
        axs[0, 0].set_title('C_D over Time')

        # Plot C_L sur le deuxième subplot (ligne 1, colonne 2)
        color = 'tab:blue'
        axs[0, 1].set_xlabel('Time Steps')
        axs[0, 1].set_ylabel('C_L (Lift Coefficient)', color=color)
        axs[0, 1].plot(self.C_L[:self.step_count-1], color=color, label='C_L (Lift Coefficient)')
        axs[0, 1].tick_params(axis='y', labelcolor=color)
        axs[0, 1].set_title('C_L over Time')

        # Deuxième ligne : Cumulative C_D et Cumulative C_L
        # Plot cumulative C_D sur le troisième subplot (ligne 2, colonne 1)
        color = 'tab:green'
        axs[1, 0].set_xlabel('Time Steps')
        axs[1, 0].set_ylabel('Cumulative C_D', color=color)
        axs[1, 0].plot(self.C_D_cumul[:self.step_count-1], color=color, label='Cumulative C_D')
        axs[1, 0].tick_params(axis='y', labelcolor=color)
        axs[1, 0].set_title('Cumulative C_D over Time')

        # Plot cumulative C_L sur le quatrième subplot (ligne 2, colonne 2)
        color = 'tab:orange'
        axs[1, 1].set_xlabel('Time Steps')
        axs[1, 1].set_ylabel('Cumulative C_L', color=color)
        axs[1, 1].plot(self.C_L_cumul[:self.step_count-1], color=color, label='Cumulative C_L')
        axs[1, 1].tick_params(axis='y', labelcolor=color)
        axs[1, 1].set_title('Cumulative C_L over Time')

        # Ajuster l'espacement entre les sous-plots
        plt.tight_layout()

        # Sauvegarder et afficher
        plt.savefig('Cd_Cl_and_cumulative_over_time.png')
        plt.show()



    



    """
    def render(self, mode = 'human'):
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

        u_x, u_y = self.flow_solver.u_.split()

        # Créer des figures avec deux sous-plots, un pour chaque composante
        fig, axs = plt.subplots(2, 1, figsize=(10, 8))

        # Option pour les lignes de niveau
        levels = np.linspace(0, 2, 10)  # Adapter selon la plage de valeurs des composantes

        # Plot de la composante u_x
        p = plot(u_x, title="Composante u_x (Vitesse en X)")
        plt.colorbar(p, ax=axs[0])
        axs[0].set_title("Composante u_x (Vitesse en X)")
        axs[0].set_xlabel("X")
        axs[0].set_ylabel("Y")

        

        # Plot de la composante u_y
        p = plot(u_y, title="Composante u_y (Vitesse en Y)")
        plt.colorbar(p, ax=axs[1])
        axs[1].set_title("Composante u_y (Vitesse en Y)")
        axs[1].set_xlabel("X")
        axs[1].set_ylabel("Y")

        
        plt.savefig('composantes_champs_de_vitesse_u.png')
        # Ajuster l'espacement entre les sous-plots
        plt.tight_layout()

        # Afficher la figure
        plt.show()
    """
    def export_to_vtk(self, episode):
        """
        Exporte la vitesse et la pression pour la visualisation avec Paraview.
        """
        # Définir les noms des fichiers de sortie au format .xdmf
        xdmf_velocity_file = os.path.join(self.output_dir, f"velocity_{episode}.xdmf")
        xdmf_pressure_file = os.path.join(self.output_dir, f"pressure_{episode}.xdmf")

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

        
