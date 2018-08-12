# -*- coding: iso-8859-15 -*-

from   fenics_mpm          import mpm_module
from   fenics_mpm.helper   import print_text, get_text, print_min_max
from   time                import time
import dolfin                  as dol
import numpy                   as np
import sys


class Model(object):
  r"""
  A model which links a :class:`~gridmodel.GridModel` to a set of :class:`~material.Material`\s; the material point method algorithm.

  :param out_dir: directory to save results, defalult is ``./output/``.  Currently not used by this class.
  :param grid_model: the finite-element model instance.
  :param dt: the timestep :math:`\Delta t`.
  :type out_dir: string
  :type grid_model: :class:`~gridmodel.GridModel`
  :type dt: float

  The instantiation of this class creates the following class variables :

  * ``self.grid_model`` -- the :class:`~gridmodel.GridModel` instance for this problem
  * ``self.dt`` -- the time-step :math:`\Delta t` to use
  * ``self.materials`` -- an initially empty :py:obj:`list` of materials
  """

  def __init__(self, out_dir, grid_model, dt, verbose=True):
    """
    This class connects the grid to each material.
    """
    self.this = self

    s = "::: INITIALIZING MPM MODEL :::"
    print_text(s, cls=self.this)

    self.out_dir    = out_dir      # output directory
    self.grid_model = grid_model   # grid model
    self.dt         = dt           # time step
    self.t          = None         # starting time, set by self.mpm()
    self.iter       = 0            # the timestep iteration
    self.verbose    = verbose      # print stuff or not
    self.materials  = []           # list of Material objects, initially none
    
    # create an MPMMaterial instance from the module just created :
    self.mpm_cpp = mpm_module.MPMModel(self.grid_model.Q, self.grid_model.dofs,
                                       np.array([1,1,0], dtype='intc'),
                                       dt, verbose)
    # intialize the cell diameter :
    self.mpm_cpp.set_h(self.grid_model.h.vector().get_local())
    
    # intialize the cell volume :
    self.mpm_cpp.set_V(self.grid_model.Ve.vector().get_local())
  
    # TODO: set the boundary conditions for C++ code :
    #self.set_boundary_conditions()
  
  def color(self):
    return 'cyan'

  def add_material(self, M):
    r"""
    Add :class:`~material.Material` ``M`` to the list of materials ``self.materials``.
    """
    s = "::: ADDING MATERIAL :::"
    print_text(s, cls=self.this)

    cpp_mat = M.get_cpp_material(self.grid_model.element)
    M.set_cpp_material(cpp_mat)           # give the material a cpp class
    self.mpm_cpp.add_material(cpp_mat)    # add it to MPMModel.cpp
    self.materials.append(M)              # keep track in Python

  def set_boundary_conditions(self):
    """
    """
    self.mpm_cpp.set_boundary_conditions(self.grid_model.bc_vrt,
                                         self.grid_model.bc_val)

  def formulate_material_basis_functions(self):
    r"""
    Iterate through each particle for each material ``M`` in :py:obj:`list` ``self.materials`` and calculate the particle interpolation function :math:`\phi_i(\mathbf{x}_p)` and gradient function :math:`\nabla \phi_i(\mathbf{x}_p)` values for each of the :math:`n_n` nodes of the corresponding grid cell.  This overwrites each :class:`~material.Material`\s ``M.vrt``, ``M.phi``, and ``M.grad_phi`` values.
    """
    if self.verbose:
      s = "::: FORMULATING BASIS FUNCTIONS :::"
      print_text(s, cls=self.this)

    self.mpm_cpp.formulate_material_basis_functions()

  def interpolate_material_mass_to_grid(self):
    r"""
    Iterate through each ``M`` :class:`~material.Material`\s in ``self.materials`` and interpolate the :math:`p=1,2,\ldots,n_p` particle masses :math:`m_p` given by ``M.m`` to the :class:`~gridmodel.GridModel` instance ``self.grid_model.m``.  That is,

    .. math::
      m_i = \sum_{p=1}^{n_p} \phi_p(\mathbf{x}_p) m_p
    """
    if self.verbose:
      s = "::: INTERPOLATING MATERIAL MASS TO GRID  :::"
      print_text(s, cls=self.this)

    self.mpm_cpp.interpolate_material_mass_to_grid()

  def interpolate_material_velocity_to_grid(self):
    r"""
    Iterate through each ``M`` :class:`~material.Material`\s in ``self.materials`` and interpolate the :math:`p=1,2,\ldots,n_p` particle velocity vectors :math:`\mathbf{u}_p` given by ``M.u`` to the :class:`~gridmodel.GridModel` instance ``self.grid_model.U3``.  In order to conserve velocity, weight by particle weight fraction :math:`m_p / m_i` for each :math:`i = 1,2,\ldots,n_n` nodes.  That is,

    .. math::
      \mathbf{u}_i = \sum_{p=1}^{n_p} \frac{m_p}{m_i} \phi_p(\mathbf{x}_p) \mathbf{u}_p

    Note that this requires that :math:`m_i` be calculated by calling :meth:`~model.Model.interpolate_material_mass_to_grid`.
    """
    if self.verbose:
      s = "::: INTERPOLATING MATERIAL VELOCITY TO GRID :::"
      print_text(s, cls=self.this)

    self.mpm_cpp.interpolate_material_velocity_to_grid()

  def calculate_material_initial_mass(self):
    r"""
    Iterate through each ``M`` :class:`~material.Material`\s in ``self.materials`` and calculate the :math:`p=1,2,\ldots,n_p` particle masses :math:`m_p` given by ``M.m`` using particle volume :math:`V_p` and constant material density :math:`\rho` :

    .. math::
      m_p = \frac{V_p}{\rho}.
    """
    if self.verbose:
      s = "::: CALCULATING MATERIAL INITIAL MASS :::"
      print_text(s, cls=self.this)

    # calculate particle masses :
    self.mpm_cpp.calculate_material_initial_mass()

  def calculate_material_initial_density(self):
    r"""
    Iterate through each ``M`` :class:`~material.Material`\s in ``self.materials`` and calculate the :math:`p=1,2,\ldots,n_p` particle densities :math:`\rho_p` given by ``M.rho`` by interpolating the :math:`i=1,2,\ldots,n_n` nodal masses :math:`m_i` and nodal cell diameter volume estimates :math:`v_i = \frac{4}{3} \pi \left(\frac{h_i}{2}\right)^3` using approximate nodal cell diameter :math:`h_i`.  That is,

    .. math::
      \rho_p = \sum_{i=1}^{n_n} \phi_i(\mathbf{x}_p) \frac{m_i}{v_i}
    
    Note that this is useful only for the initial density :math:`\rho_p^0` calculation and aftwards should evolve thereafter with :math:`\rho_p = \rho_p^0 / \mathrm{det}(F_p)`.
    """
    if self.verbose:
      s = "::: CALCULATING MATERIAL INITIAL DENSITY :::"
      print_text(s, cls=self.this)

    # calculate particle densities :
    #NOTE: this is not needed now -> self.mpm_cpp.calculate_grid_volume()
    self.mpm_cpp.calculate_material_initial_density()

  def calculate_material_initial_volume(self):
    r"""
    Iterate through each ``M`` :class:`~material.Material`\s in ``self.materials`` and calculate the :math:`p=1,2,\ldots,n_p` particle volumes :math:`V_p` given by ``M.V`` from particle mass :math:`m_p` and density :math:`\rho_p`.  That is,

    .. math::
      V_p = \frac{m_p}{\rho_p}.
    
    Note that this is useful only for the initial particle volume :math:`V_p^0` calculation and aftwards should evolve with :math:`V_p = V_p^0 \mathrm{det}(F_p)`.  Also, this requires that the particle density be initialized by calling :meth:`~model.Model.calculate_material_density`.
    """
    if self.verbose:
      s = "::: CALCULATING MATERIAL INITIAL VOLUME :::"
      print_text(s, cls=self.this)

    # calculate particle densities :
    self.mpm_cpp.calculate_material_initial_volume()

  def calculate_material_velocity_gradient(self):
    r"""
    Iterate through each ``M`` :class:`~material.Material`\s in ``self.materials`` and calculate the :math:`p=1,2,\ldots,n_p` particle velocity gradient vectors :math:`\nabla \mathbf{u}_p` given by ``M.grad_u`` by interpolating the :math:`i=1,2,\ldots,n_n` nodal velocity vectors :math:`\nabla \mathbf{u}_i` using the grid basis function gradients evaluated at the particle position :math:`\nabla \phi_i(\mathbf{x}_p)`.  That is,

    .. math::
      \nabla \mathbf{u}_p = \sum_{i=1}^{n_n} \nabla \phi_i(\mathbf{x}_p) \mathbf{u}_i.
    """
    if self.verbose:
      s = "::: CALCULATING MATERIAL VELOCITY GRADIENT :::"
      print_text(s, cls=self.this)

    # calculate particle velocity gradients :
    self.mpm_cpp.calculate_material_velocity_gradient()

  def interpolate_grid_velocity_to_material(self):
    r"""
    Iterate through each ``M`` :class:`~material.Material`\s in ``self.materials`` and interpolate the :math:`i=1,2,\ldots,n_n` grid velocity vectors :math:`\mathbf{u}_i` to each of the :math:`p=1,2,\ldots,n_p` particle velocity vectors :math:`\mathbf{u}_p^*` given by ``M.u_star``.  That is,

    .. math::
      \mathbf{u}_p^* = \sum_{i=1}^{n_n} \phi_i(\mathbf{x}_p) \mathbf{u}_i
    
    Note that this is an intermediate step used by :meth:`~model.Model.advect_material_particles`.
    """
    if self.verbose:
      s = "::: INTERPOLATING GRID VELOCITY TO MATERIAL :::"
      print_text(s, cls=self.this)
    
    self.mpm_cpp.interpolate_grid_velocity_to_material()

  def interpolate_grid_acceleration_to_material(self):
    r"""
    Iterate through each ``M`` :class:`~material.Material`\s in ``self.materials`` and interpolate the :math:`i=1,2,\ldots,n_n` grid acceleration vectors :math:`\mathbf{a}_i` to each of the :math:`p=1,2,\ldots,n_p` particle acceleration vectors :math:`\mathbf{a}_p` given by ``M.a``.  That is,

    .. math::
      \mathbf{a}_p = \sum_{i=1}^{n_n} \phi_i(\mathbf{x}_p) \mathbf{a}_i
    
    These particle accelerations are used to calculate the new particle velocities by :meth:`~model.Model.advect_material_particles`.
    """
    if self.verbose:
      s = "::: INTERPOLATING GRID ACCELERATION TO MATERIAL :::"
      print_text(s, cls=self.this)
  
    self.mpm_cpp.interpolate_grid_acceleration_to_material()

  def initialize_material_tensors(self):
    r"""
    Iterate through each ``M`` :class:`~material.Material`\s in ``self.materials`` and calculate the :math:`p=1,2,\ldots,n_p` particle incremental deformation gradient tensors set to ``M.dF`` as

    .. math::
      \mathrm{d}F_p = I + \Delta t \nabla \mathbf{u}_p

    with particle velocity gradient :math:`\nabla \mathbf{u}_p` given by ``M.grad_u`` and time-step :math:`\Delta t` from ``self.dt``; the deformation gradient tensors 

    .. math::
      F_p = \mathrm{d}F_p

    set to ``M.F``; strain-rate tensor :math:`\dot{\epsilon}_p` given by :func:`~material.Material.calculate_strain_rate` set to ``M.epsilon``; and Cauchy-stress tensor :math:`\sigma_p` given by :func:`~material.Material.calculate_stress` set to ``M.sigma``.
    """
    if self.verbose:
      s = "::: INITIALIZING MATERIAL TENSORS :::"
      print_text(s, cls=self.this)

    self.calculate_material_velocity_gradient()
    self.mpm_cpp.initialize_material_tensors()

  def update_material_density(self):
    r"""
    Iterate through each ``M`` :class:`~material.Material`\s in ``self.materials`` and calculate the :math:`p=1,2,\ldots,n_p` particle densities from the incremental particle deformation gradient tensors :math:`\mathrm{d}F_p` given by ``M.dF`` at the previous time-step :math:`t-1` from the formula

    .. math::
      \rho_p^t = \mathrm{det}(\mathrm{d}F_p)^{-1} \rho_p^{t-1}.

    This is equivalent to the operation

    .. math::
      V_p^t = \mathrm{det}(F_p)^{-1} \rho_p^0,

    with particle deformation gradient tensor :math:`F_p` given by ``M.F`` and initial density :math:`\rho_p^0` calculated by :func:`~model.Model.calculate_material_initial_density` and set to ``M.rho0``.
    """
    if self.verbose:
      s = "::: UPDATING MATERIAL VOLUME :::"
      print_text(s, cls=self.this)
    
    self.mpm_cpp.update_material_density()

  def update_material_volume(self):
    r"""
    Iterate through each ``M`` :class:`~material.Material`\s in ``self.materials`` and calculate the :math:`p=1,2,\ldots,n_p` particle volumes from the incremental particle deformation gradient tensors :math:`\mathrm{d}F_p` given by ``M.dF`` at the previous time-step :math:`t-1` from the formula

    .. math::
      V_p^t = \mathrm{det}(\mathrm{d}F_p) V_p^{t-1}.

    This is equivalent to the operation

    .. math::
      V_p^t = \mathrm{det}(F_p) V_p^0,

    with particle deformation gradient tensor :math:`F_p` given by ``M.F`` and initial volume :math:`V_p^0` calculated by :func:`~model.Model.calculate_material_initial_volume` and set to ``M.V0``.
    """
    if self.verbose:
      s = "::: UPDATING MATERIAL VOLUME :::"
      print_text(s, cls=self.this)
    
    self.mpm_cpp.update_material_volume()

  def update_material_deformation_gradient(self):
    r"""
    Iterate through each ``M`` :class:`~material.Material`\s in ``self.materials`` and update the :math:`p=1,2,\ldots,n_p` particle incremental deformation gradient tensors set to ``M.dF`` as

    .. math::
      \mathrm{d}F_p^t = I + \frac{1}{2} \left( \nabla \mathbf{u}_p^{t-1} + \nabla \mathbf{u}_p^t \right) \Delta t

    with particle velocity gradient :math:`\nabla \mathbf{u}_p` given by ``M.grad_u`` and time-step :math:`\Delta t` from ``self.dt``; and update the deformation gradient tensors 

    .. math::
      F_p^t = \mathrm{d}F_p^t \circ F_p^{t-1}

    set to ``M.F``.  Here, :math:`\circ` is the element-wise Hadamard product.
    """
    if self.verbose:
      s = "::: UPDATING MATERIAL DEFORMATION GRADIENT :::"
      print_text(s, cls=self.this)

    self.mpm_cpp.update_material_deformation_gradient()

  def update_material_stress(self):
    r"""
    Iterate through each ``M`` :class:`~material.Material`\s in ``self.materials`` and calculate the :math:`p=1,2,\ldots,n_p` particle strain rate tensors :math:`\dot{\epsilon}_p` returned by :func:`~material.Material.calculate_strain_rate` saved to ``M.depsilon``; then use these incremental strain rates to update the particle strain tensors ``M.epsilon`` by the explicit forward-Euler finite-difference scheme

    .. math::
      \epsilon_p^t = \epsilon_p^{t-1} + \dot{\epsilon}_p \Delta t 

    with time-step :math:`\Delta t` from ``self.dt``.  This updated strain tensor is then used to update the material stress :math:`\sigma_p` by :func:`~material.Material.calculate_stress`. 
    """
    if self.verbose:
      s = "::: UPDATING MATERIAL STRESS :::"
      print_text(s, cls=self.this)

    self.mpm_cpp.update_material_stress()

  def calculate_grid_internal_forces(self):
    r"""
    Iterate through each ``M`` :class:`~material.Material`\s in ``self.materials`` and interpolate the :math:`p=1,2,\ldots,n_p` particle stress divergence terms :math:`\nabla \cdot \sigma_p` to the :math:`i=1,2,\ldots,n_n` internal force vectors :math:`\mathbf{f}_i^{\mathrm{int}}` containted at ``self.grid_model.f_int`` by

    .. math::
      \mathbf{f}_i^{\mathrm{int}} = - \sum_{p=1}^{n_p} \nabla \phi_i(\mathbf{x}_p) \cdot \sigma_p V_p

    This is the weak-stress-divergence volume integral.
    """
    if self.verbose:
      s = "::: CALCULATING GRID INTERNAL FORCES :::"
      print_text(s, cls=self.this)

    self.mpm_cpp.calculate_grid_internal_forces()

  def update_grid_velocity(self):
    r"""
    Update the grid velocity :math:`\mathbf{u}_i` located at ``self.grid_model.U3`` from the current acceleration vector :math:`\mathbf{a}_i` and time-step :math:`\Delta t` from the Crank-Nicolson finite-difference scheme

    .. math::
      \mathbf{u}_i^t = \mathbf{u}_i^{t-1} + \frac{1}{2} \left( \mathbf{a}_i^t + \mathbf{a}_i^{t-1} \right) \Delta t.
    """
    if self.verbose:
      s = "::: UPDATING GRID VELOCITY :::"
      print_text(s, cls=self.this)

    self.mpm_cpp.update_grid_velocity()

  def calculate_grid_acceleration(self):
    r"""
    Calculate the :math:`i=1,2,\ldots,n_n` grid acceleration vectors :math:`\mathbf{a}_i` containted at ``self.grid_model.a3`` by

    .. math::
      \mathbf{a}_i = \frac{\mathbf{f}_i^{\mathrm{int}} + \mathbf{f}_i^{\mathrm{ext}}}{m_i},

    where the grid mass :math:`m_i` has been limited to be :math:`\geq \varepsilon = 1 \times 10^{-2}`, and external forces are currently only :math:`\mathbf{f}_i^{\mathrm{ext}} = \mathbf{0}`.
    """
    if self.verbose:
      s = "::: CALCULATING GRID ACCELERATIONS :::"
      print_text(s, cls=self.this)

    self.mpm_cpp.calculate_grid_acceleration()

  def advect_material_particles(self):
    r"""
    First, this method expects that the :math:`i=1,2,\ldots,n_n` grid accelerations :math:`\mathbf{a}_i` and velocities :math:`\mathbf{u}_i` have been interpolated from the grid to the particles by the functions :func:`~model.Model.interpolate_grid_acceleration_to_material` and :func:`~model.Model.interpolate_grid_velocity_to_material` respectively.
    These Python functions in turn call C++ functions which create intermediate velocities :math:`\mathbf{u}_p^*` and accelerations :math:`\mathbf{a}_p^*` that are only availabe within the C++ code.
    Then calling this function will iterate through each of the :class:`~material.Material`\s in ``self.materials`` and increment the :math:`p=1,2,\ldots,n_p` particle velocities :math:`\mathbf{u}_p` and particle positions :math:`\mathbf{x}_p` using the intermediate particle velocities :math:`\mathbf{u}_p^*` and acclerations :math:`\mathbf{a}_p^*` by the explicit second-order accurate finite-difference scheme

    .. math::
      \mathbf{u}_p^t &= \mathbf{u}_p^{t-1} + \frac{1}{2} \left( \mathbf{a}_p^{t-1} + \mathbf{a}_p^* \right) \Delta t \\
      \mathbf{x}_p^t &= \mathbf{x}_p^{t-1} + \mathbf{u}_p^* \Delta t + \frac{1}{2} \mathbf{a}_p^* \Delta t^2.
    """
    if self.verbose:
      s = "::: ADVECTING MATERIAL PARTICLES :::"
      print_text(s, cls=self.this)

    # advect the material particles :
    self.mpm_cpp.advect_material_particles()
  
  def retrieve_cpp_grid_m(self):
    """
    Get the mass vector from C++ model ``self.mpm_cpp`` and set to ``self.grid_model.m``.
    """
    #FIXME: figure out a way to directly update grid_model.m :
    m = dol.Function(self.grid_model.Q, name='m')
    self.grid_model.assign_variable(m, self.mpm_cpp.get_m())
      
    # assign the new mass to the grid model variable :
    self.grid_model.update_mass(m)

  def retrieve_cpp_grid_U3(self):
    """
    Get the velocity vector from C++ model ``self.mpm_cpp`` and set to ``self.grid_model.u`` and ``self.grid_model.v``.
    """
    #FIXME: figure out a way to directly update grid_model.U3 :
    u = dol.Function(self.grid_model.Q, name='u')
    v = dol.Function(self.grid_model.Q, name='v')
    self.grid_model.assign_variable(u, self.mpm_cpp.get_u_x())
    self.grid_model.assign_variable(v, self.mpm_cpp.get_u_y())

    # assign the variables to the functions
    self.grid_model.assu.assign(self.grid_model.u, u)
    self.grid_model.assv.assign(self.grid_model.v, v)

  def retrieve_cpp_grid_f_int(self):
    """
    Get the internal force vector from C++ model ``self.mpm_cpp`` and set to ``self.grid_model.f_int_x`` and ``self.grid_model.f_int_y``.
    """
    #FIXME: figure out a way to directly update grid_model.f_int :
    f_int_x  = dol.Function(self.grid_model.Q, name='f_int_x')
    f_int_y  = dol.Function(self.grid_model.Q, name='f_int_y')
    self.grid_model.assign_variable(f_int_x, self.mpm_cpp.get_f_int_x())
    self.grid_model.assign_variable(f_int_y, self.mpm_cpp.get_f_int_y())

    # assign the variables to the functions
    self.grid_model.update_internal_force_vector([f_int_x, f_int_y])
    
  def retrieve_cpp_grid_a3(self):
    """
    Get the acceleration vector from C++ model ``self.mpm_cpp`` and set to ``self.grid_model.a_x`` and ``self.grid_model.a_y``.
    """
    #FIXME: figure out a way to directly update grid_model.a3 :
    a_x = dol.Function(self.grid_model.Q, name='a_x')
    a_y = dol.Function(self.grid_model.Q, name='a_y')
    self.grid_model.assign_variable(a_x, self.mpm_cpp.get_a_x())
    self.grid_model.assign_variable(a_y, self.mpm_cpp.get_a_y())
    self.grid_model.update_acceleration([a_x, a_y])

  def retrieve_cpp_grid_properties(self):
    """
    Transfer grid properties from C++ back to python for further analysis
    with PyLab.
    """
    self.retrieve_cpp_grid_m()
    self.retrieve_cpp_grid_U3()
    self.retrieve_cpp_grid_f_int()
    self.retrieve_cpp_grid_a3()
  
  def retrieve_cpp_material_properties(self):
    """
    Transfer material properties from C++ back to python for further analysis
    with PyLab.
    """
    # iterate through all materials :
    for M in self.materials:
      M.retrieve_cpp_vrt()
      M.retrieve_cpp_phi()
      M.retrieve_cpp_grad_phi()
      M.retrieve_cpp_grad_u()
      M.retrieve_cpp_x()
      M.retrieve_cpp_u()
      M.retrieve_cpp_a()
      M.retrieve_cpp_F()
      M.retrieve_cpp_epsilon()
      M.retrieve_cpp_sigma()
      M.retrieve_cpp_rho()
      M.retrieve_cpp_V()
      M.retrieve_cpp_V0()

      if self.verbose:
        print_min_max(M.grad_u,  'M.grad_u')
        print_min_max(M.x,       'M.x')
        print_min_max(M.u,       'M.u')
        print_min_max(M.a,       'M.a')
        print_min_max(M.F,       'M.F_0')
        print_min_max(M.epsilon, 'M.epsilon_0')
        print_min_max(M.sigma,   'M.sigma_0')
        print_min_max(M.rho,     'M.rho_0')
        print_min_max(M.V,       'M.V')
        print_min_max(M.V0,      'M.V_0')

  def mpm(self, t_start, t_end, cb_ftn=None):
    r"""
    The material point method algorithm performed from time ``t_start`` to ``t_end``.

    :param t_start: starting time of the simulation
    :param t_end: ending time of the simulation
    :param cb_ftn: callback function
    :type t_start: float
    :type t_end: float
    :type cb_ftn: function
  
    For any given time-step, the algorithm consists of:

    * :func:`~model.Model.formulate_material_basis_functions`
    * :func:`~model.Model.interpolate_material_mass_to_grid`
    * :func:`~model.Model.interpolate_material_velocity_to_grid`
    
    If this is the initialization step (``t == t_start``):

    * :func:`~model.Model.initialize_material_tensors`
    * :func:`~model.Model.calculate_material_initial_density`
    * :func:`~model.Model.calculate_material_initial_volume`

    Then continue the grid calculation stage :
    
    * :func:`~model.Model.calculate_grid_internal_forces`
    * :func:`~model.Model.calculate_grid_acceleration`
    * :func:`~model.Model.update_grid_velocity`
  
    Then the particle calculation stage :

    * :func:`~model.Model.calculate_material_velocity_gradient`
    * :func:`~model.Model.update_material_deformation_gradient`
    * :func:`~model.Model.update_material_volume`
    * :func:`~model.Model.update_material_stress`
    * :func:`~model.Model.interpolate_grid_acceleration_to_material`
    * :func:`~model.Model.interpolate_grid_velocity_to_material`
    * :func:`~model.Model.advect_material_particles`

    At the end of the algorithm, the grid properties are made available from the C++ backend : 

    * :func:`~model.Model.retrieve_cpp_material_properties`
    * :func:`~model.Model.retrieve_cpp_grid_properties`
    
    """
    s = "::: BEGIN MPM ALGORITHM :::"
    print_text(s, cls=self.this)

    # initialize counter :
    self.t = t_start

    # starting time :
    t0 = time()

    # set up screen printing :
    s0    = '\r>>> simulation time: '
    s2    = ' s, CPU time for last dt: '
    s4    = ' s <<<'
    text0 = get_text(s0, 'red')
    text2 = get_text(s2, 'red')
    text4 = get_text(s4, 'red')
      
    while self.t < t_end - self.dt:

      # we are now on the next iteration :
      self.iter += 1

      # start time over :
      tic = time()

      #=========================================================================
      # begin MPM algorithm :
      if self.t == t_start:
        self.mpm_cpp.mpm(True)
      else:
        self.mpm_cpp.mpm(False)

      ## get basis function values at material point locations :
      #self.formulate_material_basis_functions()
      #
      ## interpolation from particle stage :
      #self.interpolate_material_mass_to_grid()
      #self.interpolate_material_velocity_to_grid()
      #
      ## initialization step :
      #if self.t == t_start:
      #  self.initialize_material_tensors()
      #  self.calculate_material_initial_density()
      #  self.calculate_material_initial_volume()
      #  
      ## grid calculation stage : 
      #self.calculate_grid_internal_forces()
      #self.calculate_grid_acceleration()
      #self.update_grid_velocity()

      ## particle calculation stage :
      #self.calculate_material_velocity_gradient()
      #self.update_material_deformation_gradient()
      #self.update_material_volume()
      #self.update_material_stress()
      #self.interpolate_grid_acceleration_to_material()
      #self.interpolate_grid_velocity_to_material()
      #self.advect_material_particles()
      
      # : end MPM algorithm
      #=========================================================================
      
      # call the callback function, if desired :
      if cb_ftn is not None:
        if self.verbose :
          s    = "::: calling callback function :::"
          print_text(s, cls=self.this)
        cb_ftn()

      # increment time step :
      self.t += self.dt

      # print the time to the screen :
      s1    = '%.3e' % self.t
      s3    = '%.3e' % (time() - tic)
      text1 = get_text(s1, 'red', 1)
      text3 = get_text(s3, 'red', 1)
      text  = text0 + text1 + text2 + text3 + text4

      # don't continue to fill the screen with numbers if verbosity is off :
      if self.verbose == False and self.grid_model.verbose == False: 
        sys.stdout.write(text)
        sys.stdout.flush()
      else:
        print text

    # calculate total time to compute
    dt = time() - t0
    m  = dt / 60.0
    h  = m / 60.0
    s  = dt % 60
    m  = m % 60
    text = "\ntotal time to perform transient run: %02d:%02d:%02d (%.3e s)"
    print_text(text % (h,m,s,dt), 'red', 1)
   
    # always get the properties back from C++ land : 
    self.retrieve_cpp_material_properties()
    self.retrieve_cpp_grid_properties()



