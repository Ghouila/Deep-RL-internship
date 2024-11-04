from fenics import *
from mshr import *
import matplotlib.pyplot as plt
import numpy as np
# Paramètres géométriques
L = 1
H = 0.41
r = 0.05  # Rayon du cylindre
c_x = c_y = 0.2  # Centre du cylindre
print(1)
# Paramètres des jets : ajustement pour les rendre plus petits et bien centrés
jet_width = 0.015  # Largeur des jets (plus petit)
jet_length = 0.015  # Longueur des jets (réduit)

# Créer le domaine du canal avec le cylindre
channel = Rectangle(Point(0, 0), Point(L, H))
cylinder = Circle(Point(c_x, c_y), r, 64)  # Plus de points pour un cercle plus lisse
print(1)
# Créer les jets autour du cylindre, ajustés pour être plus petits et collés
jet1 = Rectangle(Point(c_x - jet_length, c_y + r), Point(c_x, c_y + r + jet_width))  # Jet en haut
jet2 = Rectangle(Point(c_x - jet_length, c_y - r - jet_width), Point(c_x, c_y - r))  # Jet en bas
print(1)
# Soustraire le cylindre et les jets du domaine fluide
domain = channel - cylinder - jet1 - jet2
print(1)
# Générer le maillage
mesh = generate_mesh(domain, 36)
print(1)
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

# Visualisation des points de sonde
plt.figure(figsize=(8, 4))
plot(mesh, color="gray")
plt.scatter(probe_points[:, 0], probe_points[:, 1], c='k', label="Probe Points", zorder=5)
plt.title("Probe Points around Cylinder and in Rectangular Region")
plt.legend()
plt.show()

# Marquage des frontières
class InletBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], 0) and on_boundary

class OutletBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], L) and on_boundary

class WallsBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return (near(x[1], 0) or near(x[1], H)) and on_boundary

class CylinderBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return x[0] > 0.1 and x[0] < 0.3 and x[1] > 0.1 and x[1] < 0.3 and on_boundary

class Jet1Boundary(SubDomain):
    def inside(self, x, on_boundary):
        return x[0] >= (c_x - jet_length) and x[0] <= c_x and near(x[1], c_y + r + jet_width/2) and on_boundary

class Jet2Boundary(SubDomain):
    def inside(self, x, on_boundary):
        return x[0] >= (c_x - jet_length) and x[0] <= c_x and near(x[1], c_y - r - jet_width/2) and on_boundary

# Créer des sous-domaines pour chaque type de frontière
inlet_boundary = InletBoundary()
outlet_boundary = OutletBoundary()
walls_boundary = WallsBoundary()
cylinder_boundary = CylinderBoundary()
jet1_boundary = Jet1Boundary()
jet2_boundary = Jet2Boundary()

# Marquer les facettes
facets = MeshFunction("size_t", mesh, mesh.geometry().dim()-1)
facets.set_all(0)

# Attribuer des étiquettes aux différentes frontières
inlet_boundary.mark(facets, 1)
outlet_boundary.mark(facets, 2)
walls_boundary.mark(facets, 3)
cylinder_boundary.mark(facets, 4)
jet1_boundary.mark(facets, 5)
jet2_boundary.mark(facets, 6)

# Sauvegarder le maillage et les facettes dans des fichiers pour visualisation future
File('mesh_with_cylinder_and_jets.xml') << mesh
File('facets_with_cylinder_and_jets.xml') << facets

# Pour visualiser les facettes dans Paraview
File('mesh_with_cylinder_and_jets.pvd') << mesh
File('facets_with_cylinder_and_jets.pvd') << facets
