# -*- coding: iso-8859-15 -*-

from   fenics            import *
from   fenics_mpm.helper import print_text, print_min_max
import numpy                 as np
import os
import inspect


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

    # open the cpp code for evaluating the basis functions :
    cpp_src_dir = os.path.dirname(os.path.abspath(__file__)) + "/cpp/"
    header_file = open(cpp_src_dir + "MPMModel.h", "r")
    code        = header_file.read()
    header_file.close()

    # compile this with Instant JIT compiler :
    inst_params = {'code'                      : code,
                   'module_name'               : "MPMModel",
                   'source_directory'          : cpp_src_dir,
                   'sources'                   : ["MPMModel.cpp"],
                   'additional_system_headers' : ["petscsys.h"],
                   'include_dirs'              : [".", cpp_src_dir]}
    self.mpm_module = compile_extension_module(**inst_params)
    
    # have the compiler generate code for evaluating basis derivatives :
    #parameters['form_compiler']['no-evaluate_basis_derivatives'] = False
  
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

    # create a Probe instance from mpm_model.cpp :
    self.probe = self.mpm_model.Probe(self.Q)
    
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
    Create particle basis functions for the single particle coordinate vector :math:`\mathbf{x}_p`.  In what follows, the topological dimension of the mesh is :math:`d` and the number of nodes per element is :math:`n_n`.

    Returns a :py:obj:`tuple` of :class:`~numpy.ndarray`\s, respectively :

    * ``vrt`` -- set of :math:`n_n` node indices :math:`i` for this particle's cell.  Shape is :math:`1 \times n_n`.
    * ``phi`` -- interpolation function values :math:`\phi_i(\mathbf{x}_p)` corresponding to each node :math:`i` in ``vrt``.  Shape is :math:`1 \times n_n`.
    * ``grad_phi`` -- gradient of interpolation function values :math:`\nabla \phi_i(\mathbf{x}_p)` corresponding to each node :math:`i` in ``vrt``.  Shape is :math:`d \times n_n`.  For example, in three dimensions :math:`\mathbf{x}_p = [x\ y\ z]^{\intercal}`, and with nodal indicies :math:`1,2,\ldots,n` ``grad_phi`` is

    .. math:: \nabla \phi_i(\mathbf{x}_p) = 
                                   \begin{bmatrix}
                                      \frac{\partial \phi_1}{\partial x} & \frac{\partial \phi_1}{\partial y} & \frac{\partial \phi_1}{\partial z} \\
                                      \frac{\partial \phi_2}{\partial x} & \frac{\partial \phi_2}{\partial y} & \frac{\partial \phi_2}{\partial z} \\
                                      \vdots & \vdots & \vdots \\
                                      \frac{\partial \phi_n}{\partial x} & \frac{\partial \phi_n}{\partial y} & \frac{\partial \phi_3}{\partial z}
                                    \end{bmatrix}

    :param x: global coordinate to evaluate.
    :type x:  :py:obj:`int` or :py:obj:`float` or :class:`~numpy.ndarray`

    :rtype: :py:obj:`tuple` (:class:`~numpy.ndarray`, :class:`~numpy.ndarray`, :class:`~numpy.ndarray`)
    """
    #mesh    = self.mesh
    #element = self.element

    ## find the cell with point :
    #x_pt       = Point(x)
    #cell_id    = mesh.bounding_box_tree().compute_first_entity_collision(x_pt)
    #cell       = Cell(mesh, cell_id)
    #coord_dofs = cell.get_vertex_coordinates()       # local coordinates
    #
    ## array for all basis functions of the cell :
    #phi = np.zeros(element.space_dimension(), dtype=float)
    #
    ## array for values with derivatives of all 
    ## basis functions, 2 * element dim :
    #grad_phi = np.zeros(2*element.space_dimension(), dtype=float)
    #
    ## compute basis function values :
    #element.evaluate_basis_all(phi, x, coord_dofs, cell.orientation())
    #
    ## compute 1st order derivatives :
    #element.evaluate_basis_derivatives_all(1, grad_phi, x, 
    #                                       coord_dofs, cell.orientation())

    ## reshape such that rows are [d/dx, d/dy] :
    #grad_phi = grad_phi.reshape((-1, 2))

    ## get corresponding vertex indices, in dof indicies : 
    #vrt = self.dofmap.cell_dofs(cell.index())
    self.probe(x);
    vrt      = self.probe.get_vrt()
    phi      = self.probe.get_phi()
    grad_phi = self.probe.get_grad_phi()

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
