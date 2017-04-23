# -*- coding: iso-8859-15 -*-

import inspect
from   fenics   import *
from   colored  import fg, attr
#import fenicstools       as ft
import numpy             as np
import matplotlib.pyplot as plt


def raiseNotDefined():
  fileName = inspect.stack()[1][1]
  line     = inspect.stack()[1][2]
  method   = inspect.stack()[1][3]
       
  text = "*** Method not implemented: %s at line %s of %s"
  print text % (method, line, fileName)
  sys.exit(1)

def print_min_max(u, title, color='97'):
  r"""
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
  r"""
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
  r"""
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


class GridModel(object):
  r"""
  Representation of the model on a finite-element grid. 
    
  :param out_dir: directory to save results, defalult is ``./output/``.
  :param mesh: the finite-element mesh.
  :type out_dir: string
  :type mesh: :class:`~fenics.Mesh`
  """

  def __init__(self, mesh, out_dir='./output/'):
    """
    Create and instance of the model.
    """
    self.this = self
    #self.this = super(type(self), self)  # pointer to this base class
    
    # have the compiler generate code for evaluating basis derivatives :
    parameters['form_compiler']['no-evaluate_basis_derivatives'] = False
  
    s = "::: INITIALIZING BASE MODEL :::"
    print_text(s, cls=self.this)
    
    parameters['form_compiler']['quadrature_degree']  = 2
    parameters["std_out_all_processes"]               = False
    parameters['form_compiler']['cpp_optimize']       = True

    PETScOptions.set("mat_mumps_icntl_14", 100.0)

    self.mesh        = mesh
    self.out_dir     = out_dir
    self.MPI_rank    = MPI.rank(mpi_comm_world())
    
    self.generate_function_spaces()
    self.initialize_variables()

  def color(self):
    r"""
    The color used for printing messages to the screen.

    :rtype: string
    """
    return '148'

  def generate_function_spaces(self, order=1, use_periodic=False):
    r"""
    Generates the finite-element function spaces used with topological dimension :math:`d` set by variable ``self.top_dim``.

    :param order:        order :math:`k` of the shape function, currently only supported are Lagrange :math:`P_1` elements.
    :param use_periodic: use periodic boundaries along lateral boundary (currently not supported).
    :type use_periodic:  bool

    The element shape-functions available from this method are :

    * ``self.Q`` -- :math:`\mathcal{H}^k(\Omega)`
    * ``self.V`` -- :math:`[\mathcal{H}^k(\Omega)]^d`  formed using :class:`~fenics.VectorFunctionSpace`
    * ``self.T`` -- :math:`[\mathcal{H}^k(\Omega)]^{d \times d}` formed using :class:`~fenics.TensorFunctionSpace`
    """
    s = "::: generating fundamental function spaces of order %i :::" % order
    print_text(s, cls=self.this)

    if use_periodic:
      self.generate_pbc()
    else:
      self.pBC = None

    order        = 1
    space        = 'CG'
    self.Q       = FunctionSpace(self.mesh, space, order)
    self.V       = VectorFunctionSpace(self.mesh, space, order)
    self.T       = TensorFunctionSpace(self.mesh, space, order)
    
    s = "    - fundamental function spaces created - "
    print_text(s, cls=self.this)

  def initialize_variables(self):
    r"""
    Initialize the model variables to default values.  The variables 
    defined here are:

    Various things :
    
    * ``self.element``    -- the finite-element
    * ``self.top_dim``    -- the topological dimension
    * ``self.dofmap``     -- :class:`~fenics.DofMap` for converting between vertex to nodal indicies
    * ``self.h``          -- :class:`~fenics.CellSize` for ``self.mesh``
    
    Grid velocity vector :math:`\mathbf{u}_i = [u\ v\ w]^{\intercal}`:

    * ``self.U_mag``      -- velocity vector magnitude
    * ``self.U3``         -- velocity vector
    * ``self.u``          -- :math:`x`-component of velocity vector
    * ``self.v``          -- :math:`y`-component of velocity vector
    * ``self.w``          -- :math:`z`-component of velocity vector
    
    Grid acceleration vector :math:`\mathbf{a}_i = [a_x\ a_y\ a_z]^{\intercal}`:

    * ``self.a_mag``      -- acceleration vector magnitude
    * ``self.a3``         -- acceleration vector
    * ``self.a_x``        -- :math:`x`-component of acceleration vector
    * ``self.a_y``        -- :math:`y`-component of acceleration vector
    * ``self.a_z``        -- :math:`z`-component of acceleration vector

    Grid internal force vector :math:`\mathbf{f}_i^{\mathrm{int}} = [f_x^{\mathrm{int}}\ f_y^{\mathrm{int}}\ f_z^{\mathrm{int}}]^{\intercal}`:

    * ``self.f_int_mag``  -- internal force vector magnitude
    * ``self.f_int``      -- internal force vector
    * ``self.f_int_x``    -- :math:`x`-component of internal force vector
    * ``self.f_int_y``    -- :math:`y`-component of internal force vector
    * ``self.f_int_z``    -- :math:`z`-component of internal force vector

    Grid mass :math:`m_i`:

    * ``self.m``          -- mass :math:`m_i` 
    * ``self.m0``         -- inital mass :math:`m_i^0`
    """
    # the finite-element used :
    self.element = self.Q.element()
    
    # topological dimension :
    self.top_dim = self.element.topological_dimension()

    # map from verticies to nodes :
    self.dofmap  = self.Q.dofmap()

    # cell diameter :
    self.h       = project(CellSize(self.mesh), self.Q)  # cell diameter vector
    
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

    # grid internal force vector :
    self.f_int_mag      = Function(self.Q, name='f_int_mag')
    self.f_int          = Function(self.V, name='f_int')
    self.f_int_x, self.f_int_y  = self.f_int.split()
    self.f_int_x.rename('f_int_x', '')
    self.f_int_y.rename('f_int_y', '')

    # grid mass :
    self.m             = Function(self.Q, name='m')
    self.m0            = Function(self.Q, name='m0')

    # function assigners speed assigning up :
    self.assu       = FunctionAssigner(self.u.function_space(),       self.Q)
    #                                   self.V.sub(0))                
    self.assv       = FunctionAssigner(self.v.function_space(),       self.Q)
    #                                   self.V.sub(1))                
    self.assa_x     = FunctionAssigner(self.a_x.function_space(),     self.Q)
    self.assa_y     = FunctionAssigner(self.a_y.function_space(),     self.Q)
    self.assf_int_x = FunctionAssigner(self.f_int_x.function_space(), self.Q)
    self.assf_int_y = FunctionAssigner(self.f_int_y.function_space(), self.Q)
    self.assm       = FunctionAssigner(self.m.function_space(),       self.Q)

  def get_particle_basis_functions(self, x):
    r"""
    Create particle basis functions for the coordinates :math:`x`.

    :param x: global coordinate to evaluate.
    :type x: :class:`~numpy.ndarray`, int, float
    """
    mesh    = self.mesh
    element = self.element

    # find the cell with point :
    x_pt       = Point(x)
    cell_id    = mesh.bounding_box_tree().compute_first_entity_collision(x_pt)
    cell       = Cell(mesh, cell_id)
    coord_dofs = cell.get_vertex_coordinates()       # local coordinates
    
    # array for all basis functions of the cell :
    phi = np.zeros(element.space_dimension(), dtype=float)
    
    # array for values with derivatives of all 
    # basis functions, 2 * element dim :
    grad_phi = np.zeros(2*element.space_dimension(), dtype=float)
    
    # compute basis function values :
    element.evaluate_basis_all(phi, x, coord_dofs, cell.orientation())
    
    # compute 1st order derivatives :
    element.evaluate_basis_derivatives_all(1, grad_phi, x, 
                                           coord_dofs, cell.orientation())

    # reshape such that rows are [d/dx, d/dy] :
    grad_phi = grad_phi.reshape((-1, 2))

    # get corresponding vertex indices, in dof indicies : 
    vrt = self.dofmap.cell_dofs(cell.index())

    return vrt, phi, grad_phi

  def update_mass(self, m):
    r"""
    Update the grid mass :math:`m_i`, ``self.m`` to parameter ``m``.
    
    :param m: grid mass
    :type m: :class:`~fenics.Function`,
    """
    # assign the mass to the model variable :
    self.assm.assign(self.m, m)

  def update_velocity(self, U):
    r"""
    Update the grid velocity :math:`\mathbf{u}_i = [u\ v\ w]^{\intercal}`, ``self.U3`` to parameter ``U``.
    
    :param U: grid velocity
    :type U: list of :class:`~fenics.Function`\s
    """
    # assign the variables to the functions :
    self.assu.assign(self.u, U[0])
    self.assv.assign(self.v, U[1])

  def update_acceleration(self, a):
    r"""
    Update the grid acceleration :math:`\mathbf{a}_i = [a_x\ a_y\ a_z]^{\intercal}`, ``self.a3`` to parameter ``a``.
    
    :param a: grid acceleration
    :type a: list of :class:`~fenics.Function`\s
    """
    # assign the variables to the functions :
    self.assa_x.assign(self.a_x, a[0])
    self.assa_y.assign(self.a_y, a[1])

  def update_internal_force_vector(self, f_int):
    r"""
    Update the grid acceleration :math:`\mathbf{f}_i^{\mathrm{int}} = [f_x^{\mathrm{int}}\ f_y^{\mathrm{int}}\ f_z^{\mathrm{int}}]^{\intercal}` to paramter ``f_int``.
    
    :param f_int: grid internal force
    :type f_int: list of :class:`~fenics.Function`\s
    """
    # assign the variables to the functions :
    self.assf_int_x.assign(self.f_int_x, f_int[0])
    self.assf_int_y.assign(self.f_int_y, f_int[1])

  def assign_variable(self, u, var):
    r"""
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
    r"""
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
      s       = "::: saving pvd file :::"
      print_text(s, 'green')#cls=self.this)
      f << (u, float(t))
    else :
      s       = "::: saving %spvd/%s.pvd file :::" % (self.out_dir, name)
      print_text(s, 'green')#cls=self.this)
      f = File(self.out_dir + 'pvd/' +  name + '.pvd')
      f << (u, float(t))

  def save_xdmf(self, u, name, f=None, t=0.0):
    r"""
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


class Material(object):
  r"""
  Representation of an abstract material with initial conditions given by the 
  parameters ``m``, ``x`` and ``u``.

  :param m: particle mass vector :math:`\mathbf{m}_p`
  :param x: particle position vector :math:`\mathbf{x}_p`
  :param u: particle velocity vector :math:`\mathbf{u}_p`
  :type m: :class:`~numpy.ndarray`
  :type x: :class:`~numpy.ndarray`
  :type u: :class:`~numpy.ndarray`
  """
  def __init__(self, m, x, u):
    """
    """
    self.this     = super(type(self), self)  # pointer to this base class

    s = "::: INITIALIZING BASE MATERIAL :::"
    print_text(s, cls=self.this)

    self.N        = len(x[:,0])        # number of particles
    self.d        = len(x[0])          # topological dimension
    self.m        = m                  # mass vector
    self.x        = x                  # position vector
    self.u        = u                  # velocity vector
    self.u_star   = u                  # grid velocity interpolation
    self.a        = None               # acceleration vector
    self.grad_u   = None               # velocity gradient tensor
    self.vrt      = None               # grid nodal indicies for points
    self.phi      = None               # grid basis values at points
    self.grad_phi = None               # grid basis gradient values at points
    self.rho      = None               # density vector
    self.V0       = None               # initial volume vector
    self.V        = None               # volume vector
    self.F        = None               # deformation gradient tensor
    self.sigma    = None               # stress tensor
    self.epsilon  = None               # strain-rate tensor

    # identity tensors :
    self.I        = np.array([np.identity(self.d)]*self.N)
      
  def color(self):
    """
    The color used for printing messages to the screen.

    :rtype: string
    """
    return '148'

  def calculate_strain_rate(self):
    r"""
    Calculate the particle strain-rate tensor
    
    .. math::

      \dot{\epsilon}_p = \frac{1}{2} \left( \nabla \mathbf{u}_p + \left( \nabla \mathbf{u}_p \right)^{\intercal} \right)

    from particle velocity :math:`\mathbf{u}_p`.
    """
    epsilon_n = []

    # calculate particle deformation gradients :
    for grad_u_p in self.grad_u:
      dudx   = grad_u_p[0,0]
      dudy   = grad_u_p[0,1]
      dvdx   = grad_u_p[1,0]
      dvdy   = grad_u_p[1,1]
      eps_xx = dudx
      eps_xy = 0.5*(dudy + dvdx)
      eps_yy = dvdy
      eps    = np.array( [[eps_xx, eps_xy], [eps_xy, eps_yy]], dtype=float )
      epsilon_n.append(eps)
    return np.array(epsilon_n, dtype=float)

  def calculate_stress(self):
    r"""
    This method must be implemented by child classes.
    """
    raiseNotDefined()

  def plot(self):
    r"""
    Plot the positions of each particle with coordinates ``self.X`` to the 
    screen.
    """
    plt.subplot(111)
    plt.plot(self.X[:,0], self.X[:,1], 'r*')
    plt.axis('equal')
    plt.show()


class ElasticMaterial(Material):
  r"""
  Representation of an elastic material with initial conditions given by the 
  parameters ``m``, ``x``, ``u``, ``E``, and ``nu``.

  :param m: particle mass vector :math:`\mathbf{m}_p`
  :param x: particle position vector :math:`\mathbf{x}_p`
  :param u: particle velocity vector :math:`\mathbf{u}_p`
  :param E: Young's modulus :math:`E`
  :param nu: Poisson's ratio :math:`\nu`
  :type m: :class:`~numpy.ndarray`
  :type x: :class:`~numpy.ndarray`
  :type u: :class:`~numpy.ndarray`
  :type E: float
  :type nu: float
  """
  def __init__(self, m, x, u, E, nu):
    """
    """
    s = "::: INITIALIZING ELASTIC MATERIAL :::"
    print_text(s, cls=self)

    Material.__init__(self, m, x, u)

    self.E        = E         # Young's modulus
    self.nu       = nu        # Poisson's ratio

    # Lamé parameters :
    self.mu       = E / (2.0*(1.0 + nu))
    self.lmbda    = E*nu / ((1.0 + nu)*(1.0 - 2.0*nu))

  def color(self):
    return '150'

  def calculate_stress(self):
    r"""
    Calculate elastic particle-Cauchy-stress tensor
    
    .. math::

      \sigma_p = 2 \mu \dot{\epsilon}_p + \lambda \mathrm{tr} \left( \dot{\epsilon}_p \right) I

    with Lamé parameters :math:`\mu` and :math:`\lambda`.
    """
    sigma = []

    # calculate particle stress :
    for epsilon_p in self.epsilon:
      #sigma =  2.0*self.mu*eps + self.lmbda*tr(eps)*Identity(self.dim)
      trace_eps  = epsilon_p[0,0] + epsilon_p[1,1]
      c1         = 2.0 * self.mu
      c2         = self.lmbda * trace_eps
      sig_xx     = c1 * epsilon_p[0,0] + c2
      sig_xy     = c1 * epsilon_p[0,1] 
      sig_yy     = c1 * epsilon_p[1,1] + c2
      sigma_p    = np.array( [[sig_xx, sig_xy], [sig_xy, sig_yy]], dtype=float )
      sigma.append(sigma_p)
    
    # return particle Cauchy stress tensors :
    return np.array(sigma, dtype=float)


class Model(object):
  r"""
  A material point method model.

  :param out_dir: directory to save results, defalult is ``./output/``.  Currently not used by this class.
  :param grid_model: the finite-element model instance.
  :param dt: the timestep :math:`\Delta t`.
  :type out_dir: string
  :type mesh: :class:`~GridModel`
  :type dt: float
  """

  def __init__(self, out_dir, grid_model, dt):
    """
    This class connects the grid to each material.
    """
    self.out_dir    = out_dir      # output directory
    self.grid_model = grid_model   # grid model
    self.dt         = dt           # time step
    self.materials  = []           # list of Material objects, initially none

  def add_material(self, M):
    r"""
    Add :class:`~Material` ``M`` to the list of materials ``self.materials``.
    """
    self.materials.append(M)

  def formulate_material_basis_functions(self):
    r"""
    Iterate through each material and calculate the particle basis function
    value for each position.
    """
    # iterate through all materials :
    for M in self.materials:

      vrt      = []  # grid nodal indicies for points
      phi      = []  # grid basis values at points
      grad_phi = []  # grid basis gradient values at points

      # iterate through particle positions :
      for x_p in M.x: 
        # get the grid node indices, basis values, and basis gradient values :
        out = self.grid_model.get_particle_basis_functions(x_p)
      
        # append these to a list corresponding with particles : 
        vrt.append(out[0])
        phi.append(out[1])
        grad_phi.append(out[2])

      # save as array within each material :
      M.vrt      = np.array(vrt)
      M.phi      = np.array(phi, dtype=float)
      M.grad_phi = np.array(grad_phi, dtype=float)

  def interpolate_material_mass_to_grid(self):
    r"""
    """
    # new mass must start at zero :
    m    = Function(self.grid_model.Q)

    # iterate through all materials :
    for M in self.materials:

      # interpolation of mass to the grid :
      for p, phi_p, m_p in zip(M.vrt, M.phi, M.m):
        m.vector()[p] += phi_p * m_p
      
    # assign the new mass to the grid model variable :
    self.grid_model.update_mass(m)

  def interpolate_material_velocity_to_grid(self):
    r"""
    """
    # new velocity must start at zero :
    #model.assign_variable(self.U3, DOLFIN_EPS)
    #u,v = self.U3.split(True)
    u    = Function(self.grid_model.Q)
    v    = Function(self.grid_model.Q)
    
    # iterate through all materials :
    for M in self.materials:

      # interpolation of mass-conserving velocity to the grid :
      for p, phi_p, m_p, u_p in zip(M.vrt, M.phi, M.m, M.u):
        m_i = self.grid_model.m.vector()[p]
        u.vector()[p] += u_p[0] * phi_p * m_p / m_i
        v.vector()[p] += u_p[1] * phi_p * m_p / m_i

    # assign the variables to the functions
    self.grid_model.update_velocity([u,v])

  def calculate_material_density(self):
    r"""
    """
    h   = self.grid_model.h.vector().array()    # cell diameter
    m   = self.grid_model.m.vector().array()

    # iterate through all materials :
    for M in self.materials:

      rho = []

      # calculate particle densities :
      for i, phi_i in zip(M.vrt, M.phi):
        rho_p = np.sum( m[i] * phi_i / h[i]**3 )
        rho.append(rho_p)
      
      # update material density :
      M.rho = np.array(rho, dtype=float)

  def calculate_material_initial_volume(self):
    r"""
    """
    # iterate through all materials :
    for M in self.materials:

      # calculate inital volume from particle mass and density :
      M.V = np.array(M.m / M.rho, dtype=float)

  def calculate_material_velocity_gradient(self):
    r"""
    Calculate particle velocity gradient for each material :

    * ``self.grad_U``     -- particle velocity gradient tensor :math:`\nabla \mathbf{u}_p`
    * ``self.dudx``       -- :math:`\frac{\partial u_p}{\partial x}`
    * ``self.dudy``       -- :math:`\frac{\partial u_p}{\partial y}`
    * ``self.dvdx``       -- :math:`\frac{\partial v_p}{\partial x}`
    * ``self.dvdy``       -- :math:`\frac{\partial v_p}{\partial y}`
    """
    # recover the grid nodal velocities :
    u,v = self.grid_model.U3.split(True)

    # iterate through all materials :
    for M in self.materials:

      grad_U_p_v = []
      
      # calculate particle velocity gradients :
      for i, grad_phi_i in zip(M.vrt, M.grad_phi):
        u_i    = u.vector()[i]
        v_i    = v.vector()[i]
        dudx_p = np.sum(grad_phi_i[:,0] * u.vector()[i])
        dudy_p = np.sum(grad_phi_i[:,1] * u.vector()[i])
        dvdx_p = np.sum(grad_phi_i[:,0] * v.vector()[i])
        dvdy_p = np.sum(grad_phi_i[:,1] * v.vector()[i])
        grad_U_p_v.append(np.array( [[dudx_p, dudy_p], [dvdx_p, dvdy_p]] ))
      
      # update the particle velocity gradients :
      M.grad_u = np.array(grad_U_p_v, dtype=float)

  def interpolate_grid_velocity_to_material(self):
    r"""
    """
    u, v  = self.grid_model.U3.split(True)

    # iterate through all materials :
    for M in self.materials:

      v_p_v = []

      # iterate through each particle :
      for i, phi_i in zip(M.vrt, M.phi):
        u_p = np.sum(phi_i * u.vector()[i])
        v_p = np.sum(phi_i * v.vector()[i])
        v_p_v.append(np.array([u_p, v_p]))

      # update material velocity :
      M.u_star = np.array(v_p_v, dtype=float)

  def interpolate_grid_acceleration_to_material(self):
    r"""
    """
    a_x, a_y = self.grid_model.a3.split(True)

    # iterate through all materials :
    for M in self.materials:

      a_p_v = []

      for i, phi_i in zip(M.vrt, M.phi):
        a_x_p = np.sum(phi_i * a_x.vector()[i])
        a_y_p = np.sum(phi_i * a_y.vector()[i])
        a_p_v.append(np.array([a_x_p, a_y_p]))

      # update material acceleration :
      M.a = np.array(a_p_v, dtype=float)

  def initialize_material_tensors(self):
    r"""
    """
    self.calculate_material_velocity_gradient()

    # iterate through all materials :
    for M in self.materials:
      M.dF      = M.I + M.grad_u * self.dt
      M.F       = M.dF
      M.epsilon = M.calculate_strain_rate()
      M.sigma   = M.calculate_stress()

  def update_material_volume(self):
    r"""
    """
    # iterate through all materials :
    for M in self.materials:
      M.V = (M.dF[:,0,0] * M.dF[:,1,1] + M.dF[:,1,0] * M.dF[:,0,1]) * M.V

  def update_material_deformation_gradient(self):
    r"""
    """
    # iterate through all materials :
    for M in self.materials:
      M.dF = M.I + M.grad_u * self.dt
      M.F *= M.dF

  def update_material_stress(self):
    r"""
    """
    # iterate through all materials :
    for M in self.materials:
      epsilon_n  = M.calculate_strain_rate()
      M.epsilon += epsilon_n * self.dt
      M.sigma    = M.calculate_stress()

  def calculate_grid_internal_forces(self):
    r"""
    """
    # new internal forces start at zero :
    f_int_x  = Function(self.grid_model.Q)
    f_int_y  = Function(self.grid_model.Q)

    # iterate through all materials :
    for M in self.materials:

      # interpolation particle internal forces to grid :
      for p, grad_phi_p, F_p, sig_p, V_p in zip(M.vrt, M.grad_phi,
                                                M.F, M.sigma,
                                                M.V):
        f_int = []
        for grad_phi_i in grad_phi_p:
          f_int.append(- np.dot(sig_p, grad_phi_i) * V_p)
        f_int = np.array(f_int, dtype=float)
        
        f_int_x.vector()[p] += f_int[:,0].astype(float)
        f_int_y.vector()[p] += f_int[:,1].astype(float)
      
    # assign the variables to the functions
    self.grid_model.update_internal_force_vector([f_int_x, f_int_y])

  def update_grid_velocity(self):
    r"""
    """
    # calculate the new grid velocity :
    u_i   = self.grid_model.U3.vector().array()
    a_i   = self.grid_model.a3.vector().array()
    u_i_n = u_i + a_i * self.dt
    
    # assign the new velocity vector :
    self.grid_model.assign_variable(self.grid_model.U3, u_i_n)

  def calculate_grid_acceleration(self):
    r"""
    """
    f_int_x, f_int_y = self.grid_model.f_int.split(True)
    f_int_x_a = f_int_x.vector().array()
    f_int_y_a = f_int_y.vector().array()
    m_a       = self.grid_model.m.vector().array()

    eps_m               = 1e-2
    f_int_x_a[m_a == 0] = 0.0
    f_int_y_a[m_a == 0] = 0.0
    m_a[m_a < eps_m]    = eps_m
    
    a_x    = Function(self.grid_model.Q)
    a_y    = Function(self.grid_model.Q)

    a_x.vector().set_local(f_int_x_a / m_a)
    a_y.vector().set_local(f_int_y_a / m_a)
    self.grid_model.update_acceleration([a_x, a_y])

  def advect_material_particles(self):
    r"""
    """
    # interpolate the grid acceleration from the grid to the particles : 
    self.interpolate_grid_acceleration_to_material()

    # interpolate the velocities from the grid back to the particles :
    self.interpolate_grid_velocity_to_material()

    # iterate through all materials :
    for M in self.materials:

      # calculate the new material velocity :
      M.u += M.a * self.dt

      # advect the material :
      M.x += M.u_star * self.dt

  def mpm(self, t_start, t_end):
    r"""
    """
    t = t_start
      
    # files for saving :
    m_file = File(self.out_dir + '/m.pvd')
    u_file = File(self.out_dir + '/u.pvd')
    a_file = File(self.out_dir + '/a.pvd')
    f_file = File(self.out_dir + '/f.pvd')

    while t <= t_end:
      self.formulate_material_basis_functions()
      self.interpolate_material_mass_to_grid()
      self.interpolate_material_velocity_to_grid()
      
      # initialization step :
      if t == t_start:
        self.initialize_material_tensors()
        self.calculate_material_density()
        self.calculate_material_initial_volume()
     
      self.calculate_grid_internal_forces()
      self.calculate_grid_acceleration()
      self.update_grid_velocity()

      self.calculate_material_velocity_gradient()
      self.update_material_deformation_gradient()
      self.update_material_volume()
      self.update_material_stress()
      
      # save the result :
      self.grid_model.save_pvd(self.grid_model.m,   'm', f=m_file, t=t)
      self.grid_model.save_pvd(self.grid_model.U3, 'U3', f=u_file, t=t)
      self.grid_model.save_pvd(self.grid_model.a3, 'a3', f=a_file, t=t)
      self.grid_model.save_pvd(self.grid_model.f_int, 'f_int', f=f_file, t=t)
      
      # move the model forward in time :
      self.advect_material_particles()

      # increment time step :
      t += self.dt


