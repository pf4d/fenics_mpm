# evaluate basis function at a point (x,y) :


from dolfin import *
#import fenicstools as ft
import numpy as np

# Have the compiler generate code for evaluating derivatives
parameters['form_compiler']['no-evaluate_basis_derivatives'] = False

#mesh = UnitCubeMesh(2, 2, 2)
mesh = UnitSquareMesh(1, 1)
#mesh = UnitIntervalMesh(2)
V    = FunctionSpace(mesh, 'CG', 1)
Ve   = V.ufl_element()
el   = V.element()

# Where to evaluate
x = np.array([0.4, 0.6])
#x = np.array([0.6])

# Find the cell with point
x_point  = Point(*x) 
cell_id  = mesh.bounding_box_tree().compute_first_entity_collision(x_point)
cell     = Cell(mesh, cell_id)
vertices = cell.entities(0)                      # vertex indicies 
coordinate_dofs = cell.get_vertex_coordinates()  # local vertex coordinates

# Array for values. Here it is simply a single scalar
values = np.zeros(1, dtype=float)
for i in range(el.space_dimension()): 
  el.evaluate_basis(i, values, x, coordinate_dofs, cell.orientation()) 
  print i, values

# you can also evaluate all basis functions of the 
# cell at once by evaluate_basis_all.
values = np.zeros(el.space_dimension(), dtype=float)
el.evaluate_basis_all(values, x, coordinate_dofs, cell.orientation()) 
print values

#===============================================================================
# evaluate basis function derivatives at a point (x,y) :

# Array for values with derivatives of all basis functions. 4 * element dim
deriv_values = np.zeros(2*el.space_dimension(), dtype=float)
# Compute all 1st order derivatives
el.evaluate_basis_derivatives_all(1, deriv_values, x, 
                                  coordinate_dofs, cell.orientation())
# Reshape such that columns are [d/dxx, d/dxy, d/dyx, d/dyy]
deriv_values = deriv_values.reshape((-1, 2))
print deriv_values

import sys
sys.exit(0)


#===============================================================================
# evaluation of a function at a set of points (x_i, y_i) :

# Initialize some functions in V
u = interpolate(Expression('x[0]',degree = 1), V)
v = interpolate(Expression('1-x[0]',degree = 1), V)
w = interpolate(Expression('x[0]+1',degree =1 ), V)
flist = [u, v, w]

# Define some points
xp = np.array([[0.1, 0.1],[0.2, 0.2],[0.3, 0.3]])

# Initialize Probes class from fenicstools and evaluate
p = ft.Probes(xp.flatten(), V)
for f in flist:
  p(f)

# Print the result as np.array
print p.array()

# Print results of first probings on rank 0
pa = p.array()
if MPI.rank(mpi_comm_world()) == 0:
  print "Probe = ", pa[0]



