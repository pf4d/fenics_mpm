from dolfin import *
#import fenicstools       as ft
import numpy             as np
import pandas            as pd
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

  def __init__(self, out_dir, order, n):
    """
    """
    # have the compiler generate code for evaluating basis derivatives :
    parameters['form_compiler']['no-evaluate_basis_derivatives'] = False

    self.order   = order
    self.out_dir = out_dir
    self.mesh    = UnitSquareMesh(n, n)
    self.Q       = FunctionSpace(self.mesh, 'CG', order)
    self.V       = VectorFunctionSpace(self.mesh, 'CG', order)
    self.T       = TensorFunctionSpace(self.mesh, 'CG', order)
    self.element = self.Q.element()
    self.top_dim = self.element.topological_dimension()
    self.dofmap  = self.Q.dofmap()
    
    # grid velocity :
    self.U_mag          = Function(self.Q, name='U_mag')
    self.U3             = Function(self.V, name='U3')
    self.u, self.v      = self.U3.split()
    self.u.rename('u', '')
    self.v.rename('v', '')

    # particle velocity gradient :
    self.grad_U         = Function(self.T, name='grad_U')
    self.dudx, self.dudy, self.dvdx, self.dvdy = self.grad_U.split()
    self.dudx.rename('dudx', '')
    self.dudy.rename('dudy', '')
    self.dvdx.rename('dvdx', '')
    self.dvdy.rename('dvdy', '')

    # grid mass :
    self.m              = Function(self.Q, name='m')

    # function assigners speed assigning up :
    self.assdudx       = FunctionAssigner(self.dudx.function_space(), self.Q)
    self.assdudy       = FunctionAssigner(self.dudy.function_space(), self.Q)
    self.assdvdx       = FunctionAssigner(self.dvdx.function_space(), self.Q)
    self.assdvdy       = FunctionAssigner(self.dvdy.function_space(), self.Q)
    self.assx          = FunctionAssigner(self.u.function_space(),    self.Q)
    #                                      self.V.sub(0))
    self.assy          = FunctionAssigner(self.v.function_space(),    self.Q)
    #                                      self.V.sub(1))
    self.assm          = FunctionAssigner(self.m.function_space(),    self.Q)

  def formulate_material_basis_functions(self, M):
    """
    """
    element  = self.element
    mesh     = self.mesh

    phi      = []
    vrt      = []
    grad_phi = []
    
    # iterate through all the particle positions :
    for x_p in M.x: 
      # find the cell with point :
      x_pt       = Point(x_p) 
      cell_id    = mesh.bounding_box_tree().compute_first_entity_collision(x_pt)
      cell       = Cell(mesh, cell_id)
      coord_dofs = cell.get_vertex_coordinates()       # local coordinates
      
      # array for all basis functions of the cell :
      phi_i = np.zeros(element.space_dimension(), dtype=float)
    
      # array for values with derivatives of all 
      # basis functions, 2 * element dim :
      grad_phi_i = np.zeros(2*element.space_dimension(), dtype=float)
     
      # compute basis function values :
      element.evaluate_basis_all(phi_i, x_p, coord_dofs, cell.orientation())
      
      # compute 1st order derivatives :
      element.evaluate_basis_derivatives_all(1, grad_phi_i, x_p, 
                                             coord_dofs, cell.orientation())

      # reshape such that rows are [d/dx, d/dy] :
      grad_phi_i = grad_phi_i.reshape((-1, 2))

      # get corresponding vertex indices, in dof indicies : 
      vrt_i = self.dofmap.cell_dofs(cell.index())

      # append these to a list corresponding with particles : 
      phi.append(phi_i)
      grad_phi.append(grad_phi_i)
      vrt.append(vrt_i)

    # save as arrays :
    self.phi      = np.array(phi, dtype=float)
    self.vrt      = np.array(vrt, dtype=float)
    self.grad_phi = np.array(grad_phi, dtype=float)

    # save the unique vertex values for interpolating back to material points :
    vrt_i, idx = np.unique(self.vrt, return_index=True)
    self.vrt_i = Vector(mpi_comm_world(), idx.size)
    self.vrt_i.set_local(vrt_i)

    # the corresponding unique basis values :
    phi_i      = self.phi.flatten()[idx]
    self.phi_i = Vector(mpi_comm_world(), idx.size)
    self.phi_i.set_local(phi_i)

    # basis derivatives :
    grad_phi_i        = self.grad_phi.reshape(self.vrt.size, self.top_dim)[idx]
    self.grad_phi_i_x = Vector(mpi_comm_world(), idx.size)
    self.grad_phi_i_y = Vector(mpi_comm_world(), idx.size)
    self.grad_phi_i_x.set_local(grad_phi_i[:,0])
    self.grad_phi_i_y.set_local(grad_phi_i[:,1])

    # and save the non-zero vertex values for later :
    self.nonzero_nodes = idx

  def advect_particles(self):
    """
    """
    self.vrt_i

    # the corresponding unique basis values :
    self.phi_i

    # and basis derivatives :
    self.grad_phi_i_x
    self.grad_phi_i_y

  def interpolate_particle_mass_to_grid(self, M):
    """
    """
    # new mass must start at zero :
    m    = Function(self.Q)

    # interpolation of mass to the grid :
    for p, phi_p, m_p in zip(self.vrt, self.phi, M.m):
      m.vector()[p] += phi_p * m_p
    
    # assign the mass to the model variable :
    self.assm.assign(self.m, m)

  def interpolate_particle_velocity_to_grid(self, M):
    """
    """
    # new velocity must start at zero :
    #model.assign_variable(self.U3, DOLFIN_EPS)
    #u,v = self.U3.split(True)
    u    = Function(self.Q)
    v    = Function(self.Q)

    # interpolation of mass-conserving velocity to the grid :
    u_i = []
    for p, phi_p, m_p, u_p in zip(self.vrt, self.phi, M.m, M.u):
      m_i = m.vector()[p]
      u.vector()[p] += u_p[0] * phi_p * m_p / m_i
      v.vector()[p] += u_p[1] * phi_p * m_p / m_i
      u_i.append([u.vector()[p], v.vector()[p]])

    # assign the variables to the functions
    self.assx.assign(self.u, u)
    self.assy.assign(self.v, v)

  def calculate_particle_velocity_gradient(self, M):
    """
    """
    # basis derivatives :
    grad_phi_i_x = self.grad_phi_i_x
    grad_phi_i_y = self.grad_phi_i_y

    # calculate particle velocity gradients :
    for p, grad_phi_p, u_p in zip(self.vrt, self.grad_phi, M.u):
      u_i = u.vector()[p]
      v_i = v.vector()[p]
      dudx.vector()[p] += grad_phi_p[0] * u_i
      dudy.vector()[p] += grad_phi_p[1] * u_i
      dvdx.vector()[p] += grad_phi_p[2] * v_i
      dvdy.vector()[p] += grad_phi_p[3] * v_i

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
    u    = Function(self.Q)
    v    = Function(self.Q)
    dudx = Function(self.Q)
    dudy = Function(self.Q)
    dvdx = Function(self.Q)
    dvdy = Function(self.Q)

    # interpolation of mass to the grid :
    for p, phi_p, m_p in zip(self.vrt, self.phi, M.m):
      m.vector()[p] += phi_p * m_p

    # interpolation of mass-conserving velocity to the grid :
    u_i = []
    for p, phi_p, m_p, u_p in zip(self.vrt, self.phi, M.m, M.u):
      m_i = m.vector()[p]
      u.vector()[p] += u_p[0] * phi_p * m_p / m_i
      v.vector()[p] += u_p[1] * phi_p * m_p / m_i
      u_i.append([u.vector()[p], v.vector()[p]])

    ## calculate particle velocity gradients :
    #for p, grad_phi_p, u_p in zip(self.vrt, self.grad_phi, M.u):
    #  u_i = u.vector()[p]
    #  v_i = v.vector()[p]
    #  dudx.vector()[p] += grad_phi_p[0] * u_i
    #  dudy.vector()[p] += grad_phi_p[1] * u_i
    #  dvdx.vector()[p] += grad_phi_p[2] * v_i
    #  dvdy.vector()[p] += grad_phi_p[3] * v_i
    
    # assign the variables to the functions
    self.assm.assign(self.m, m)
    self.assx.assign(self.u, u)
    self.assy.assign(self.v, v)
    self.assdudx.assign(self.dudx, dudx)
    self.assdudy.assign(self.dudy, dudy)
    self.assdvdx.assign(self.dvdx, dvdx)
    self.assdvdy.assign(self.dvdy, dvdy)

  def calculate_particle_velocity_gradient(self):
    """
    """
    # array for values with derivatives of all 
    # basis functions, 4 * element dim
    grad_phi = np.zeros(4*self.element.space_dimension(), dtype=float)
    
    # compute all 2nd order derivatives
    el.evaluate_basis_derivatives_all(1, grad_phi, x, 
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

  def save_pvd(self, v):
    """
    """
    File(self.out_dir + '/' + v.name() + '.pvd') << v


#===============================================================================
# model properties :
out_dir  = 'output/'
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
model    = Model(out_dir, order, n_x)

# calculate the particle basis :
model.formulate_material_basis_functions(M1)

# interpolate the material to the grid :
model.interpolate_material_to_grid(M1)

# save the result :
model.save_pvd(model.m)
model.save_pvd(model.U3)

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
