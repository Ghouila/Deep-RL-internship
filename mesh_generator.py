from fenics import *
from mshr import *
import matplotlib.pyplot as plt

# Paramètres géométriques
L = 2.2
H = 0.41
r = 0.05  # Rayon du cylindre
c_x = c_y = 0.2  # Centre du cylindre

# Paramètres des jets : ajustement pour les rendre plus petits et bien centrés
jet_width = 0.015  # Largeur des jets (plus petit)
jet_length = 0.015  # Longueur des jets (réduit)

# Créer le domaine du canal avec le cylindre
channel = Rectangle(Point(0, 0), Point(L, H))
cylinder = Circle(Point(c_x, c_y), r, 64)  # Plus de points pour un cercle plus lisse

# Créer les jets autour du cylindre, ajustés pour être plus petits et collés
jet1 = Rectangle(Point(c_x - jet_length, c_y + r), Point(c_x, c_y + r + jet_width))  # Jet en haut
jet2 = Rectangle(Point(c_x - jet_length, c_y - r - jet_width), Point(c_x, c_y - r))  # Jet en bas

# Soustraire le cylindre et les jets du domaine fluide
domain = channel - cylinder - jet1 - jet2

# Générer le maillage
mesh = generate_mesh(domain, 64)

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

# Visualiser le maillage avec les facettes marquées
plot(mesh)
plt.show()

# Pour visualiser les facettes dans Paraview
File('mesh_with_cylinder_and_jets.pvd') << mesh
File('facets_with_cylinder_and_jets.pvd') << facets
