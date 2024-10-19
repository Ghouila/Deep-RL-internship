import warnings
warnings.filterwarnings("ignore")


from fenics import *
import numpy as np
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




class JetVelocity(UserExpression):
    def __init__(self, jet_strength, **kwargs):
        self.jet_strength = jet_strength
        super().__init__(**kwargs)

    def eval(self, values, x):
        # Profil de jet parabolique
        values[0] = self.jet_strength * 4.0 * x[0] * (0.41 - x[0]) / 0.41**2  # Composante x de la vitesse
        values[1] = 0

    def value_shape(self):
        return (2,)


class FlowSolver:
    """Solveur IPCS pour FEniCS"""

    def __init__(self, flow_params, geometry_params, solver_params):
        self.geometry_params = geometry_params
        self.flow_params = flow_params
        self.solver_params = solver_params

        # Paramètres fluidiques
        self.mu = self.flow_params['mu']  # Viscosité dynamique
        self.rho = self.flow_params['rho']  # Densité
        self.dt = self.solver_params['dt']  # Pas de temps

        # Lire le maillage et les facettes depuis un fichier
        mesh = Mesh(geometry_params['mesh'])
        facets = MeshFunction('size_t', mesh, geometry_params['facets'])  # Import des facettes

        # Définir les espaces fonctionnels pour la vitesse et la pression
        V = VectorFunctionSpace(mesh, 'P', 2)  # Espace pour la vitesse
        Q = FunctionSpace(mesh, 'P', 1)  # Espace pour la pression

        # Conditions aux limites via les facettes
        #inflow_profile = ('4.0*1.5*x[0]*(0.41 - x[0]) / pow(0.41, 2)', '4.0*1.5*x[1]*(0.41 - x[1]) / pow(0.41, 2)' )
        inflow_profile = ('4.0*1.5*x[1]*(0.41 - x[1]) / pow(0.41, 2)', '0' )
        #inflow_profile = ('3', '2' )
        
        # Utiliser les facettes marquées pour appliquer les conditions aux limites
        bcu_inlet = DirichletBC(V, Expression(inflow_profile, degree=2), facets, 1)  # Facette 1 : Inlet
        bcu_walls = DirichletBC(V, Constant((0, 0)), facets, 3)  # Facette 3 : Walls
        bcu_cylinder = DirichletBC(V, Constant((0, 0)), facets, 4)  # Facette 4 : Obstacle (cylindre)
        bcu_jet1 = DirichletBC(V, Constant((0, 0)), facets, 5)  # Facette 5 : Jet 1
        bcu_jet2 = DirichletBC(V, Constant((0, 0)), facets, 6)  # Facette 6 : Jet 2
        bcp_outlet = DirichletBC(Q, Constant(0), facets, 2)  # Facette 2 : Outlet

        # Combiner les conditions aux limites
        bcu = [bcu_inlet, bcu_walls, bcu_cylinder, bcu_jet1, bcu_jet2]
        bcp = [bcp_outlet]

        # Fonctions pour stocker les solutions à chaque pas de temps
        u_n = Function(V)
        u_ = Function(V)
        p_n = Function(Q)
        p_ = Function(Q)

        # Définir les fonctions test et essai
        u = TrialFunction(V)
        v = TestFunction(V)
        p = TrialFunction(Q)
        q = TestFunction(Q)

        # Constantes et termes du problème
        U = 0.5*(u_n + u)
        n = FacetNormal(mesh)
        f = Constant((0, 0))  # Force externe (zéro)
        k = Constant(self.dt)
        mu = Constant(self.mu)
        rho = Constant(self.rho)

        # Gradient symétrique
        def epsilon(u):
            return sym(nabla_grad(u))

        # Tenseur de contrainte
        def sigma(u, p):
            return 2*mu*epsilon(u) - p*Identity(len(u))

        # Formulation variationnelle pour l'étape 1 (vitesse)
        F1 = rho*dot((u - u_n) / k, v)*dx + rho*dot(dot(u_n, nabla_grad(u_n)), v)*dx + inner(sigma(U, p_n), epsilon(v))*dx + dot(p_n*n, v)*ds - dot(mu*nabla_grad(U)*n, v)*ds - dot(f, v)*dx
        a1 = lhs(F1)
        L1 = rhs(F1)

        # Formulation variationnelle pour l'étape 2 (correction de pression)
        a2 = dot(nabla_grad(p), nabla_grad(q))*dx
        L2 = dot(nabla_grad(p_n), nabla_grad(q))*dx - (1/k)*div(u_)*q*dx

        # Formulation variationnelle pour l'étape 3 (correction de vitesse)
        a3 = dot(u, v)*dx
        L3 = dot(u_, v)*dx - k*dot(nabla_grad(p_ - p_n), v)*dx

        # Assembler les matrices
        A1 = assemble(a1)
        A2 = assemble(a2)
        A3 = assemble(a3)

        # Appliquer les conditions aux limites aux matrices
        [bc.apply(A1) for bc in bcu]
        [bc.apply(A2) for bc in bcp]

        # Sauvegarder les attributs pour le solveur
        self.A1, self.L1 = A1, L1
        self.A2, self.L2 = A2, L2
        self.A3, self.L3 = A3, L3
        self.bcu, self.bcp = bcu, bcp
        self.u_, self.u_n = u_, u_n
        self.p_, self.p_n = p_, p_n
        self.mesh = mesh
        self.normal = n
        self.dt = self.dt

    def evolve(self, jet_bc_values):
        # Appliquer les conditions de jets aux conditions limites
        
        Q1 = jet_bc_values; Q2 = [-x for x in Q1]
        print("Q1 = ",Q1);   print("Q2 = ", Q2)
        jet_profile = JetVelocity(Q1, degree=2)
        bcu_jet = DirichletBC(self.u_.function_space(), jet_profile, f"near(x[0], {self.geometry_params['jet_positions'][0]})")
        self.bcu.append(bcu_jet)
        jet_profile = JetVelocity(Q2, degree=2)
        bcu_jet = DirichletBC(self.u_.function_space(), jet_profile, f"near(x[0], {self.geometry_params['jet_positions'][1]})")
        self.bcu.append(bcu_jet)
        # Réduire les messages à WARNINGS uniquement
        set_log_level(LogLevel.WARNING)
        # Étape 1 : Résolution de la vitesse
        b1 = assemble(self.L1)
        [bc.apply(b1) for bc in self.bcu]
        solve(self.A1, self.u_.vector(), b1)
        
        # Étape 2 : Correction de la pression
        b2 = assemble(self.L2)
        [bc.apply(b2) for bc in self.bcp]
        solve(self.A2, self.p_.vector(), b2)

        # Étape 3 : Correction de la vitesse
        b3 = assemble(self.L3)
        solve(self.A3, self.u_.vector(), b3)

        # Mise à jour des solutions pour le prochain pas de temps
        self.u_n.assign(self.u_)
        self.p_n.assign(self.p_)

    def compute_drag(self):
        # Calcul de la traînée
        dObs = Measure('ds', domain=self.mesh)  # Remplacer par l'identifiant correct de l'obstacle
        n = self.normal
        u_t = inner(as_vector((n[1], -n[0])), self.u_)
        
        # Normaliser correctement les forces
        drag_form = (2 / 0.1) * (self.mu / self.rho * inner(grad(u_t), n) * n[0]) * dObs
        drag = assemble(drag_form)
        
        # Ajoutez des vérifications pour éviter les valeurs anormales
        if abs(drag) > 1e10:  # Vérification si le drag est trop grand
            drag = 0.0
        
        return drag

    def compute_lift(self):
        # Calcul de la portance
        dObs = Measure('ds', domain=self.mesh)  # Remplacer '1' par l'identifiant de l'obstacle
        n = self.normal
        u_t = inner(as_vector((n[1], -n[0])), self.u_)
        lift_form = (2 / 0.1) * (self.mu / self.rho * inner(grad(u_t), n) * n[1]) * dObs
        lift = assemble(lift_form)
        
        # Ajoutez des vérifications pour éviter les valeurs anormales
        if abs(lift) > 1e10:  # Vérification si le drag est trop grand
            lift = 0.0
        return lift

    def sample_pressure_at_location(self, location):
        """
        Extrait la pression à une position spécifique dans le domaine.
        Args:
            location (tuple): Les coordonnées (x, y) de l'emplacement où échantillonner la pression.
        Returns:
            float: Valeur de la pression à cet emplacement.
        """
        point = Point(location)
        p_value = self.p_(point)
        return p_value
