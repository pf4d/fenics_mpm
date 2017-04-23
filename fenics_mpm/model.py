# -*- coding: iso-8859-15 -*-

from   fenics import *
import numpy  as np


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
    Add :class:`~material.Material` ``M`` to the list of materials ``self.materials``.
    """
    self.materials.append(M)

  def formulate_material_basis_functions(self):
    r"""
    Iterate through each particle for each material ``M`` in :py:obj:`list` ``self.materials`` and calculate the particle interpolation function :math:`\phi_i(\mathbf{x}_p)` and gradient function :math:`\nabla \phi_i(\mathbf{x}_p)` values for each of the :math:`n_n` nodes of the corresponding grid cell.  This overwrites each :class:`~material.Material`\s ``M.vrt``, ``M.phi``, and ``M.grad_phi`` values.
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
    Iterate through each ``M`` :class:`~material.Material`\s in ``self.materials`` and interpolate the :math:`p=1,2,\ldots,n_p` particle masses :math:`m_p` given by ``M.m`` to the :class:`~gridmodel.GridModel` instance ``self.grid_model.m``.  That is,

    .. math::
      m_i = \sum_{p=1}^{n_p} \phi_p(\mathbf{x}_p) m_p
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
    Iterate through each ``M`` :class:`~material.Material`\s in ``self.materials`` and interpolate the :math:`p=1,2,\ldots,n_p` particle velocity vectors :math:`\mathbf{u}_p` given by ``M.u`` to the :class:`~gridmodel.GridModel` instance ``self.grid_model.U3``.  In order to conserve velocity, weight by particle weight fraction :math:`m_p / m_i` for each :math:`i = 1,2,\ldots,n_n` nodes.  That is,

    .. math::
      \mathbf{u}_i = \sum_{p=1}^{n_p} \frac{m_p}{m_i} \phi_p(\mathbf{x}_p) \mathbf{u}_p

    Note that this requires that :math:`m_i` be calculated by calling :meth:`~model.Model.interpolate_material_mass_to_grid`.
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
    Iterate through each ``M`` :class:`~material.Material`\s in ``self.materials`` and calculate the :math:`p=1,2,\ldots,n_p` particle densities :math:`\rho_p` given by ``M.rho`` with grid mass :math:`m_i` and using the cell diameter :math:`h_i` to approximate the cell volume :math:`v_i = \frac{4}{3} \pi \left(\frac{h_i}{2}\right)^3`.  That is,

    .. math::
      \rho_p = \sum_{i=1}^{n_n} \phi_i(\mathbf{x}_p) \frac{m_i}{v_i}
    
    Note that this is useful only for the initial density :math:`\rho_p^0` calculation and aftwards should evolve with :math:`\rho_p = \rho_p^0 / \mathrm{det}(F_p)`.
    """
    h   = self.grid_model.h.vector().array()    # cell diameter
    m   = self.grid_model.m.vector().array()

    # iterate through all materials :
    for M in self.materials:

      rho = []

      # calculate particle densities :
      for i, phi_i in zip(M.vrt, M.phi):
        v_i   = 4.0/3.0 * pi * (h[i]/2.0)**3
        rho_p = np.sum( m[i] * phi_i / v_i )
        rho.append(rho_p)
      
      # update material density :
      M.rho = np.array(rho, dtype=float)

  def calculate_material_initial_volume(self):
    r"""
    Iterate through each ``M`` :class:`~material.Material`\s in ``self.materials`` and calculate the :math:`p=1,2,\ldots,n_p` particle volumes :math:`V_p` given by ``M.V`` from particle mass :math:`m_p` and density :math:`\rho_p`.  That is,

    .. math::
      V_p = \frac{m_p}{\rho_p}.
    
    Note that this is useful only for the initial particle volume :math:`V_p^0` calculation and aftwards should evolve with :math:`V_p = V_p^0 \mathrm{det}(F_p)`.  Also, this requires that the particle density be initialized by calling :meth:`~model.Model.calculate_material_density`.
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
    Iterate through each ``M`` :class:`~material.Material`\s in ``self.materials`` and interpolate the :math:`i=1,2,\ldots,n_n` grid velocity vectors :math:`\mathbf{u}_i` to each of the :math:`p=1,2,\ldots,n_p` particle velocity vectors :math:`\mathbf{u}_p^*` given by ``M.u_star``.  That is,

    .. math::
      \mathbf{u}_p^* = \sum_{i=1}^{n_n} \phi_i(\mathbf{x}_p) \mathbf{u}_i
    
    Note that this is an intermediate step used by :meth:`~model.Model.advect_material_particles`.
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
    Iterate through each ``M`` :class:`~material.Material`\s in ``self.materials`` and interpolate the :math:`i=1,2,\ldots,n_n` grid acceleration vectors :math:`\mathbf{a}_i` to each of the :math:`p=1,2,\ldots,n_p` particle acceleration vectors :math:`\mathbf{a}_p` given by ``M.a``.  That is,

    .. math::
      \mathbf{a}_p = \sum_{i=1}^{n_n} \phi_i(\mathbf{x}_p) \mathbf{a}_i
    
    These particle accelerations are used to calculate the new particle velocities by :meth:`~model.Model.advect_material_particles`.
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
    Iterate through each ``M`` :class:`~material.Material`\s in ``self.materials`` and calculate the :math:`p=1,2,\ldots,n_p` particle incremental deformation gradient tensors set to ``M.dF`` as

    .. math::
      \mathrm{d}F_p = I + \Delta t \nabla \mathbf{u}_p

    with particle velocity gradient :math:`\nabla \mathbf{u}_p` given by ``M.grad_u`` and time-step :math:`\Delta t` from ``self.dt``; the deformation gradient tensors 

    .. math::
      F_p = \mathrm{d}F_p

    set to ``M.F``; strain-rate tensor :math:`\dot{\epsilon}_p` given by :func:`~material.Material.calculate_strain_rate` set to ``M.epsilon``; and Cauchy-stress tensor :math:`\sigma_p` given by :func:`~material.Material.calculate_stress` set to ``M.sigma``.
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
    Iterate through each ``M`` :class:`~material.Material`\s in ``self.materials`` and calculate the :math:`p=1,2,\ldots,n_p` particle volumes from the incremental deformation gradient tensors :math:`\mathrm{d}F_p` given by ``M.dF`` at the previous time-step :math:`t-1` from the formula

    .. math::
      V_p^t = \mathrm{det}(\mathrm{d}F_b) V_p^{t-1}.
    """
    # iterate through all materials :
    for M in self.materials:
      M.V = (M.dF[:,0,0] * M.dF[:,1,1] + M.dF[:,1,0] * M.dF[:,0,1]) * M.V

  def update_material_deformation_gradient(self):
    r"""
    Iterate through each ``M`` :class:`~material.Material`\s in ``self.materials`` and update the :math:`p=1,2,\ldots,n_p` particle incremental deformation gradient tensors set to ``M.dF`` as

    .. math::
      \mathrm{d}F_p = I + \left( \nabla \mathbf{u}_p \right) \Delta t 

    with particle velocity gradient :math:`\nabla \mathbf{u}_p` given by ``M.grad_u`` and time-step :math:`\Delta t` from ``self.dt``; and update the deformation gradient tensors 

    .. math::
      F_p^t = \mathrm{d}F_p \circ F_p^{t-1}

    set to ``M.F``.  Here, :math:`\circ` is the element-wise Hadamard product.
    """
    # iterate through all materials :
    for M in self.materials:
      M.dF = M.I + M.grad_u * self.dt
      M.F *= M.dF

  def update_material_stress(self):
    r"""
    Iterate through each ``M`` :class:`~material.Material`\s in ``self.materials`` and calculate the :math:`p=1,2,\ldots,n_p` incremental particle strain rate tensors :math:`\dot{\epsilon}_p^*` returned by :func:`~material.Material.calculate_strain_rate`; then use these incremental strain rates to update the particle strain-rate tensors ``M.epsilon`` by the explicit backward-Euler finite-difference scheme

    .. math::
      \dot{\epsilon}_p^t = \dot{\epsilon}_p^{t-1} + \dot{\epsilon}_p^* \Delta t 

    with time-step :math:`\Delta t` from ``self.dt``.  This updated strain-rate tensor is then used to update the material stress :math:`\sigma_p` by :func:`~material.Material.calculate_stress`. 
    """
    # iterate through all materials :
    for M in self.materials:
      epsilon_n  = M.calculate_strain_rate()
      M.epsilon += epsilon_n * self.dt
      M.sigma    = M.calculate_stress()

  def calculate_grid_internal_forces(self):
    r"""
    Iterate through each ``M`` :class:`~material.Material`\s in ``self.materials`` and interpolate the :math:`p=1,2,\ldots,n_p` particle stress divergence terms :math:`\nabla \cdot \sigma_p` to the :math:`i=1,2,\ldots,n_n` internal force vectors :math:`\mathbf{f}_i^{\mathrm{int}}` containted at ``self.grid_model.f_int`` by

    .. math::
      \mathbf{f}_i^{\mathrm{int}} = - \sum_{p=1}^{n_p} \nabla \phi_i(\mathbf{x}_p) \cdot \sigma_p V_p

    This is the weak-stress-divergence volume integral.
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
    Update the grid velocity :math:`\mathbf{u}_i` located at ``self.grid_model.U3`` from the current acceleration vector :math:`\mathbf{a}_i` and time-step :math:`\Delta t` from the explicit backward-Euler finite-difference scheme

    .. math::
      \mathbf{u}_i^t = \mathbf{u}_i^{t-1} + \mathbf{a}_i \Delta t.
    """
    # calculate the new grid velocity :
    u_i   = self.grid_model.U3.vector().array()
    a_i   = self.grid_model.a3.vector().array()
    u_i_n = u_i + a_i * self.dt
    
    # assign the new velocity vector :
    self.grid_model.assign_variable(self.grid_model.U3, u_i_n)

  def calculate_grid_acceleration(self):
    r"""
    Calculate the :math:`i=1,2,\ldots,n_n` grid acceleration vectors :math:`\mathbf{a}_i` containted at ``self.grid_model.a3`` by

    .. math::
      \mathbf{a}_i = \frac{\mathbf{f}_i^{\mathrm{int}} + \mathbf{f}_i^{\mathrm{ext}}}{m_i},

    where the grid mass :math:`m_i` has been limited to be :math:`\geq \varepsilon = 1 \times 10^{-2}`, and external forces is currently only :math:`\mathbf{f}_i^{\mathrm{ext}} = \mathbf{0}`.
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
    First, interpolate the :math:`i=1,2,\ldots,n_n` grid accelerations :math:`\mathbf{a}_i` and velocities :math:`\mathbf{u}_i` by the functions :func:`~model.Model.interpolate_grid_acceleration_to_material` and :func:`~model.Model.interpolate_grid_velocity_to_material` respectively.  Then iterate through each ``M`` :class:`~material.Material`\s in ``self.materials`` and increment the :math:`p=1,2,\ldots,n_p` intermediate particle velocities :math:`\mathbf{u}_p^*` and particle positions :math:`\mathbf{x}_p` by the explicit backward-Euler finite-difference scheme

    .. math::
      \mathbf{u}_p^t &= \mathbf{u}_p^{t-1} + \mathbf{a}_p \Delta t \\
      \mathbf{x}_p^t &= \mathbf{x}_p^{t-1} + \mathbf{u}_p^* \Delta t.
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
    The material point method algorithm performed from time ``t_start`` to ``t_end``.

    :param t_start: starting time of the simulation
    :param t_end: ending time of the simulation
    :type t_start: float
    :type t_end: float
    
    For any given time-step, the algorithm consists of:

    * :func:`~model.Model.formulate_material_basis_functions`
    * :func:`~model.Model.interpolate_material_mass_to_grid`
    * :func:`~model.Model.interpolate_material_velocity_to_grid`
    
    If this is the initialization step (``t == t_end``):

    * :func:`~model.Model.initialize_material_tensors`
    * :func:`~model.Model.calculate_material_density`
    * :func:`~model.Model.calculate_material_initial_volume`

    Then continue :
    
    * :func:`~model.Model.calculate_grid_internal_forces`
    * :func:`~model.Model.calculate_grid_acceleration`
    * :func:`~model.Model.update_grid_velocity`

    * :func:`~model.Model.calculate_material_velocity_gradient`
    * :func:`~model.Model.update_material_deformation_gradient`
    * :func:`~model.Model.update_material_volume`
    * :func:`~model.Model.update_material_stress`
    * :func:`~model.Model.advect_material_particles`

    There are ``pvd`` files saved each time-step to ``self.out_dir``.
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


