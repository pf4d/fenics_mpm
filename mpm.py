from dolfin import *
#import fenicstools       as ft
import numpy             as np
import matplotlib.pyplot as plt
from colored           import fg, attr

def print_min_max(u, title, color='97'):
  """
  Print the minimum and maximum values of ``u``, a Vector, Function, or array.

  :param u: the variable to print the min and max of
  :param title: the name of the function to print
  :param color: the color of printed text
  :type u: :class:`~fenics.GenericVector`, :class:`~numpy.ndarray`, :class:`~fenics.Function`, int, float, :class:`~fenics.Constant`
  :type title: string
  :type color: string
  """
  if isinstance(u, GenericVector):
    uMin = MPI.min(mpi_comm_world(), u.min())
    uMax = MPI.max(mpi_comm_world(), u.max())
    s    = title + ' <min, max> : <%.3e, %.3e>' % (uMin, uMax)
    print_text(s, color)
  elif isinstance(u, np.ndarray):
    if u.dtype != np.float64:
      u = u.astype(float64)
    uMin = MPI.min(mpi_comm_world(), u.min())
    uMax = MPI.max(mpi_comm_world(), u.max())
    s    = title + ' <min, max> : <%.3e, %.3e>' % (uMin, uMax)
    print_text(s, color)
  elif isinstance(u, Function):# \
    #   or isinstance(u, dolfin.functions.function.Function):
    uMin = MPI.min(mpi_comm_world(), u.vector().min())
    uMax = MPI.max(mpi_comm_world(), u.vector().max())
    s    = title + ' <min, max> : <%.3e, %.3e>' % (uMin, uMax)
    print_text(s, color)
  elif isinstance(u, int) or isinstance(u, float):
    s    = title + ' : %.3e' % u
    print_text(s, color)
  elif isinstance(u, Constant):
    s    = title + ' : %.3e' % u(0)
    print_text(s, color)
  else:
    er = title + ": print_min_max function requires a Vector, Function" \
         + ", array, int or float, not %s." % type(u)
    print_text(er, 'red', 1)


def get_text(text, color='white', atrb=0, cls=None):
  """
  Returns text ``text`` from calling class ``cls`` for printing at a later time.

  :param text: the text to print
  :param color: the color of the text to print
  :param atrb: attributes to send use by ``colored`` package
  :param cls: the calling class
  :type text: string
  :type color: string
  :type atrb: int
  :type cls: object
  """
  if cls is not None:
    color = cls.color()
  if MPI.rank(mpi_comm_world())==0:
    if atrb != 0:
      text = ('%s%s' + text + '%s') % (fg(color), attr(atrb), attr(0))
    else:
      text = ('%s' + text + '%s') % (fg(color), attr(0))
    return text


def print_text(text, color='white', atrb=0, cls=None):
  """
  Print text ``text`` from calling class ``cls`` to the screen.

  :param text: the text to print
  :param color: the color of the text to print
  :param atrb: attributes to send use by ``colored`` package
  :param cls: the calling class
  :type text: string
  :type color: string
  :type atrb: int
  :type cls: object
  """
  if cls is not None:
    color = cls.color()
  if MPI.rank(mpi_comm_world())==0:
    if atrb != 0:
      text = ('%s%s' + text + '%s') % (fg(color), attr(atrb), attr(0))
    else:
      text = ('%s' + text + '%s') % (fg(color), attr(0))
    print text

# sunflower seed arrangement :
# "A better way to construct the sunflower head"
# https://doi.org/10.1016/0025-5564(79)90080-4

def radius(k,n,b,r_max):
  # put on the boundary :
  if k > n-b:  r = r_max
  # apply square root :
  else:        r = r_max*np.sqrt(k - 0.5) / np.sqrt( n - (b+1) / 2.0)
  return r

#  example: n=500, alpha=2
def sunflower(n, alpha, x0, y0, r_max):
    b       = np.round(alpha*np.sqrt(n))  # number of boundary points
    phi     = (np.sqrt(5)+1) / 2.0        # golden ratio
    r_v     = []
    theta_v = []
    for k in range(1, n+1):
      r_v.append( radius(k,n,b,r_max) )
      theta_v.append( 2*pi*k / phi**2 )
    x_v     = x0 + r_v*np.cos(theta_v)
    y_v     = y0 + r_v*np.sin(theta_v)
    X       = np.ascontiguousarray(np.array([x_v, y_v]).T)
    return X

class Material(object):
  """
  Representation material consisting of n particles with mass m, 
  position x, and velocity u.
  """
  def __init__(self, n, m, x, u):
    """
    """
    self.n = n
    self.m = m
    self.x = x
    self.u = u
  
  def plot(self):
    """
    """
    plt.subplot(111)
    plt.plot(self.X[:,0], self.X[:,1], 'r*')
    plt.axis('equal')
    plt.show()

class Model(object):

  def __init__(self, order, n):
    """
    """
    # have the compiler generate code for evaluating basis derivatives :
    parameters['form_compiler']['no-evaluate_basis_derivatives'] = False

    self.order   = order
    self.mesh    = UnitSquareMesh(n, n)
    self.Q       = FunctionSpace(self.mesh, 'CG', order)
    self.V       = VectorFunctionSpace(self.mesh, 'CG', order)
    self.element = self.Q.element()
    self.dofmap  = self.Q.dofmap()
    
    # grid functions :
    self.U_mag         = Function(self.Q, name='U_mag')
    self.U3            = Function(self.V, name='U3')
    u,v                = self.U3.split()
    u.rename('u', '')
    v.rename('v', '')
    self.u             = u
    self.v             = v
    self.m             = Function(self.Q, name='m')
    #self.grad_u        = Function(

    # function assigners speed assigning up :
    self.assx          = FunctionAssigner(self.u.function_space(), self.Q)
    #                                      self.V.sub(0))
    self.assy          = FunctionAssigner(self.v.function_space(), self.Q)
    #                                      self.V.sub(1))
    self.assm          = FunctionAssigner(self.m.function_space(), self.Q)

  def interpolate_material_to_grid(self, M):
    """
    """
    element = self.element
    mesh    = self.mesh
    n       = mesh.num_vertices()           # number of mesh vertices
    m       = self.m

    # zero the vector :
    #model.assign_variable(self.U3, DOLFIN_EPS)
    #u,v = self.U3.split(True)
    u = Function(self.Q)
    v = Function(self.Q)

    phi = []
    vrt = []
  
    # iterate through all the particle positions :
    for x_p in M.x: 
      # find the cell with point :
      x_pt       = Point(x_p) 
      cell_id    = mesh.bounding_box_tree().compute_first_entity_collision(x_pt)
      cell       = Cell(mesh, cell_id)
      coord_dofs = cell.get_vertex_coordinates()       # local coordinates
      
      # get all basis functions of the cell :
      phi_i = np.zeros(element.space_dimension(), dtype=float)
      element.evaluate_basis_all(phi_i, x_p, coord_dofs, cell.orientation())

      # get corresponding vertex indices, in dof indicies : 
      vrt_i = self.dofmap.cell_dofs(cell.index())

      # append these to a list corresponding with particles : 
      phi.append(phi_i)
      vrt.append(vrt_i)

    # interpolation of mass to the grid :
    for p, phi_p, m_p in zip(vrt, phi, M.m):
      m.vector()[p] += phi_p * m_p

    # interpolation of mass-conserving velocity to the grid :
    for p, phi_p, m_p, u_p in zip(vrt, phi, M.m, M.u):
      m_i = m.vector()[p]
      u.vector()[p] += u_p[0] * phi_p * m_p / m_i
      v.vector()[p] += u_p[1] * phi_p * m_p / m_i
    
    # assign the variables to the functions
    self.assm.assign(self.m, m)
    self.assx.assign(self.u, u)
    self.assy.assign(self.v, v)

  def calculate_particle_velocity_gradients(self):
    """
    """
    # array for values with derivatives of all 
    # basis functions, 4 * element dim
    deriv_values = np.zeros(4*self.element.space_dimension(), dtype=float)
    
    # compute all 2nd order derivatives
    el.evaluate_basis_derivatives_all(1, deriv_values, x, 
                                      coordinate_dofs, cell.orientation())

    # reshape such that rows are [d/dx, d/dy] :
    deriv_values = deriv_values.reshape((-1, 2))
    print deriv_values

  def assign_variable(self, u, var):
    """
    Manually assign the values from ``var`` to ``u``.  The parameter ``var``
    may be a string pointing to the location of an :class:`~fenics.XDMFFile`, 
    :class:`~fenics.HDF5File`, or an xml file.

    :param u:        FEniCS :class:`~fenics.Function` assigning to
    :param var:      value assigning from
    :param annotate: allow Dolfin-Adjoint annotation
    :type var:       float, int, :class:`~fenics.Expression`,
                     :class:`~fenics.Constant`, :class:`~fenics.GenericVector`,
                     string, :class:`~fenics.HDF5File`
    :type u:         :class:`~fenics.Function`, :class:`~fenics.GenericVector`,
                     :class:`~fenics.Constant`, float, int
    """
    if isinstance(var, float) or isinstance(var, int):
      if    isinstance(u, GenericVector) or isinstance(u, Function) \
         or isinstance(u, dolfin.functions.function.Function):
        u.vector()[:] = var
      elif  isinstance(u, Constant):
        u.assign(var)
      elif  isinstance(u, float) or isinstance(u, int):
        u = var
    
    elif isinstance(var, np.ndarray):
      if var.dtype != np.float64:
        var = var.astype(np.float64)
      u.vector().set_local(var)
      u.vector().apply('insert')
    
    elif isinstance(var, Expression) \
      or isinstance(var, Constant)  \
      or isinstance(var, dolfin.functions.constant.Constant) \
      or isinstance(var, Function) \
      or isinstance(var, dolfin.functions.function.Function) \
      or isinstance(var, GenericVector):
      u.assign(var)
      #u.interpolate(var, annotate=annotate)

    #elif isinstance(var, GenericVector):
    #  self.assign_variable(u, var.array(), annotate=annotate)

    elif isinstance(var, str):
      File(var) >> u

    elif isinstance(var, HDF5File):
      var.read(u, u.name())

    else:
      s =  "*************************************************************\n" + \
           "assign_variable() function requires a Function, array, float,\n" + \
           " int, Vector, Expression, Constant, or string path to .xml,\n"   + \
           "not %s.  Replacing object entirely\n" + \
           "*************************************************************"
      print_text(s % type(var) , 'red', 1)
      u = var
    print_min_max(u, u.name())


#===============================================================================
# model properties :
order    = 1
n_x      = 100

# create a material :
n        = 500
x0, y0   = 0.6,0.6
r_max    = 0.1

X        = sunflower(n, 2, x0, y0, r_max)
#X        = np.ascontiguousarray(np.array([[0.4,0.1],[0.9,0.1]]))
M        = 1 * np.ones(n)
U        = 1 * np.ones([n,2])

M1       = Material(n,M,X,U)

# initialize the model :
model    = Model(order, n_x)

# interpolate the material to the grid :
model.interpolate_material_to_grid(M1)

# save the result :
File('m.pvd') << model.m
File('u.pvd') << model.U3

#mu    = E / (2.0*(1.0 + nu))
#lmbda = E*nu / ((1.0 + nu)*(1.0 - 2.0*nu))

# Stress tensor
def sigma(r):
  return 2.0*mu*sym(grad(r)) + lmbda*tr(sym(grad(r)))*Identity(len(r))

#===============================================================================
# evaluate basis function derivatives at a point (x,y) :

## Array for values with derivatives of all basis functions. 4 * element dim
#deriv_values = np.zeros(4*el.space_dimension(), dtype=float)
## Compute all 2nd order derivatives
#el.evaluate_basis_derivatives_all(1, deriv_values, x, 
#                                  coordinate_dofs, cell.orientation())
## Reshape such that columns are [d/dxx, d/dxy, d/dyx, d/dyy]
#deriv_values = deriv_values.reshape((-1, 4))
#print deriv_values
#
#
##===============================================================================
## evaluation of a function at a set of points (x_i, y_i) :
#
## Initialize some functions in V
#u = interpolate(Expression('x[0]',degree = 1), V)
#v = interpolate(Expression('1-x[0]',degree = 1), V)
#w = interpolate(Expression('x[0]+1',degree =1 ), V)
#flist = [u, v, w]
#
## Define some points
#xp = np.array([[0.1, 0.1],[0.2, 0.2],[0.3, 0.3]])
#
## Initialize Probes class from fenicstools and evaluate
#p = ft.Probes(xp.flatten(), V)
#for f in flist:
#  p(f)
#
## Print the result as np.array
#print p.array()
#
## Print results of first probings on rank 0
#pa = p.array()
#if MPI.rank(mpi_comm_world()) == 0:
#  print "Probe = ", pa[0]
