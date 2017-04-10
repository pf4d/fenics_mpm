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
    self.n = n                       # number of particles
    self.m = m                       # mass vector
    self.x = x                       # position vector
    self.u = u                       # velocity vector
    self.a = np.zeros(n)             # acceleration vector
    self.grad_u = np.zeros((n,2,2))  # velocity gradient
  
  def plot(self):
    """
    """
    plt.subplot(111)
    plt.plot(self.X[:,0], self.X[:,1], 'r*')
    plt.axis('equal')
    plt.show()


class Model(object):

  def __init__(self, out_dir, order, n, dt):
    """
    """
    # have the compiler generate code for evaluating basis derivatives :
    parameters['form_compiler']['no-evaluate_basis_derivatives'] = False

    self.out_dir = out_dir
    self.order   = order
    self.dt      = dt
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
    
    # grid acceleration :
    self.a_mag          = Function(self.Q, name='a_mag')
    self.a3             = Function(self.V, name='a3')
    self.a_x, self.a_y  = self.a3.split()
    self.a_x.rename('a_x', '')
    self.a_y.rename('a_y', '')

    # particle velocity gradient :
    self.grad_U         = Function(self.T, name='grad_U')
    self.dudx, self.dudy, self.dvdx, self.dvdy = self.grad_U.split()
    self.dudx.rename('dudx', '')
    self.dudy.rename('dudy', '')
    self.dvdx.rename('dvdx', '')
    self.dvdy.rename('dvdy', '')

    # grid mass :
    self.m             = Function(self.Q, name='m')

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
    self.vrt      = np.array(vrt)
    self.phi      = np.array(phi, dtype=float)
    self.grad_phi = np.array(grad_phi, dtype=float)

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
      m_i = self.m.vector()[p]
      u.vector()[p] += u_p[0] * phi_p * m_p / m_i
      v.vector()[p] += u_p[1] * phi_p * m_p / m_i
      u_i.append([u.vector()[p], v.vector()[p]])

    # assign the variables to the functions
    self.assx.assign(self.u, u)
    self.assy.assign(self.v, v)

  def calculate_particle_velocity_gradient(self, M):
    """
    """
    u, v       = self.U3.split(True)
    grad_U_p_v = []

    # calculate particle velocity gradients :
    for i, grad_phi_i in zip(self.vrt, self.grad_phi):
      u_i = u.vector()[i]
      v_i = v.vector()[i]
      dudx_p = np.sum(grad_phi_i[:,0] * u.vector()[i])
      dudy_p = np.sum(grad_phi_i[:,1] * u.vector()[i])
      dvdx_p = np.sum(grad_phi_i[:,0] * v.vector()[i])
      dvdy_p = np.sum(grad_phi_i[:,1] * v.vector()[i])
      grad_U_p_v.append(np.array( [[dudx_p, dudy_p], [dvdx_p, dvdy_p]] ))
    
    # update the particle velocity gradients :
    M.grad_u = np.array(grad_U_p_v, dtype=float)

  def calculate_particle_velocity(self, M):
    """
    """
    u, v  = self.U3.split(True)
    v_p_v = []

    for i, phi_i in zip(self.vrt, self.phi):
      u_p = np.sum(phi_i * u.vector()[i])
      v_p = np.sum(phi_i * v.vector()[i])
      v_p_v.append(np.array([u_p, v_p]))

    # update particle velocity :
    M.u = np.array(v_p_v, dtype=float)

  def calculate_particle_acceleration(self, M):
    """
    """
    a_x, a_y = self.a3.split(True)
    a_p_v    = []

    for i, phi_i in zip(self.vrt, self.phi):
      a_x_p = np.sum(phi_i * a_x.vector()[i])
      a_y_p = np.sum(phi_i * a_y.vector()[i])
      a_p_v.append(np.array([a_x_p, a_y_p]))

    # update particle acceleration :
    M.a = np.array(a_p_v, dtype=float)

  def update_grid_velocity(self):
    """
    """
    v_i   = self.U3.vector().array()
    a_i   = self.a3.vector().array()
    dt    = self.dt
    v_i_n = v_i + a_i * dt

    # assign the new velocity vector :
    self.assign_variable(self.U3, v_i_n)

  def advect_material_particles(self, M):
    """
    """
    self.calculate_particle_acceleration(M)
    u_p_n = M.u + M.a * self.dt
    self.update_grid_velocity()
    self.calculate_particle_velocity(M)
    x_p_n = M.x + M.u * self.dt
    M.u = u_p_n
    M.x = x_p_n


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

  def save_pvd(self, u, name, f=None, t=0.0):
    """
    Save a :class:`~fenics.XDMFFile` with name ``name`` of the 
    :class:`~fenics.Function` ``u`` to the ``xdmf`` directory specified by 
    ``self.out_dir``.
    
    If ``f`` is a :class:`~fenics.XDMFFile` object, save to this instead.

    If ``t`` is a float or an int, mark the file with the timestep ``t``.

    :param u:    the function to save
    :param name: the name of the .xdmf file to save
    :param f:    the file to save to
    :param t:    the timestep to mark the file with
    :type f:     :class:`~fenics.XDMFFile`
    :type u:     :class:`~fenics.Function` or :class:`~fenics.GenericVector`
    :type t:     int or float
    """
    if f != None:
      s       = "::: saving %s.pdf file :::" % name
      print_text(s, 'green')#cls=self.this)
      f << (u, float(t))
    else :
      s       = "::: saving %spvd/%s.pvd file :::" % (self.out_dir, name)
      print_text(s, 'green')#cls=self.this)
      f = File(self.out_dir + 'pvd/' +  name + '.pvd')
      f << (u, float(t))

  def save_xdmf(self, u, name, f=None, t=0.0):
    """
    Save a :class:`~fenics.XDMFFile` with name ``name`` of the 
    :class:`~fenics.Function` ``u`` to the ``xdmf`` directory specified by 
    ``self.out_dir``.
    
    If ``f`` is a :class:`~fenics.XDMFFile` object, save to this instead.

    If ``t`` is a float or an int, mark the file with the timestep ``t``.

    :param u:    the function to save
    :param name: the name of the .xdmf file to save
    :param f:    the file to save to
    :param t:    the timestep to mark the file with
    :type f:     :class:`~fenics.XDMFFile`
    :type u:     :class:`~fenics.Function` or :class:`~fenics.GenericVector`
    :type t:     int or float
    """
    if f != None:
      s       = "::: saving %s.xdmf file :::" % name
      print_text(s, 'green')#cls=self.this)
      f.write(u, float(t))
    else :
      s       = "::: saving %sxdmf/%s.xdmf file :::" % (self.out_dir, name)
      print_text(s, 'green')#cls=self.this)
      f = XDMFFile(self.out_dir + 'xdmf/' +  name + '.xdmf')
      f.write(u)


#===============================================================================
# model properties :
out_dir  = 'output/'
order    = 1
n_x      = 100
dt       = 0.1

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
model    = Model(out_dir, order, n_x, dt)

# calculate the particle basis :
model.formulate_material_basis_functions(M1)

# interpolate the material to the grid :
model.interpolate_particle_mass_to_grid(M1)
model.interpolate_particle_velocity_to_grid(M1)
model.calculate_particle_velocity_gradient(M1)

# files for saving :
m_file = File(out_dir + '/m.pvd')
u_file = File(out_dir + '/u.pvd')

# save the result :
model.save_pvd(model.m,   'm', f=m_file, t=0.0)
model.save_pvd(model.U3, 'U3', f=u_file, t=0.0)

# move the model forward in time :
model.advect_material_particles(M1)
model.formulate_material_basis_functions(M1)
model.interpolate_particle_mass_to_grid(M1)
model.interpolate_particle_velocity_to_grid(M1)
model.save_pvd(model.m,   'm', f=m_file, t=dt)
model.save_pvd(model.U3, 'U3', f=u_file, t=dt)

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
