# -*- coding: iso-8859-15 -*-

import numpy                  as np
import matplotlib.pyplot      as plt
from   fenics_mpm.helper      import print_text, raiseNotDefined
from   fenics_mpm             import mpm_module
import sys


class Material(object):
  r"""
  Representation of an abstract material with initial conditions given by the 
  parameters length :math:`n_p` :class:`~numpy.ndarray` vectors ``m``, ``x`` 
  and ``u``.

  :param x: particle position vector :math:`\mathbf{x}_p`
  :param u: particle velocity vector :math:`\mathbf{u}_p`
  :param m: particle mass vector :math:`\mathbf{m}_p`
  :param V: particle volume vector :math:`\mathbf{V}_p`
  :param rho: particle density :math:`\rho`
  :type x: :class:`~numpy.ndarray`
  :type u: :class:`~numpy.ndarray`
  :type m: :class:`~numpy.ndarray`
  :type V: :class:`~numpy.ndarray`
  :type rho: float

  In creation of this class, a number of particle state paramters are initialized.  These are:

  * ``self.n``        -- number of particles :math:`n_p`
  * ``self.d``        -- topological dimension :math:`d`
  * ``self.m``        -- :math:`n_p \times 1` mass vector :math:`\mathbf{m}_p`
  * ``self.x``        -- :math:`n_p \times d` position vector :math:`\mathbf{x}_p`
  * ``self.u``        -- :math:`n_p \times d` velocity vector :math:`\mathbf{u}_p`
  * ``self.a``        -- :math:`n_p \times d` acceleration vector :math:`\mathbf{a}_p`
  * ``self.grad_u``   -- :math:`n_p \times (d \times d)` velocity gradient tensors :math:`\nabla \mathbf{u}_p`
  * ``self.vrt``      -- :math:`n_p \times n_n` grid nodal indicies for points :math:`i`
  * ``self.phi``      -- :math:`n_p \times n_n` grid basis values at points :math:`\phi(\mathbf{x}_p)`
  * ``self.grad_phi`` -- :math:`n_p \times (n_n \times d)` grid basis gradient tensors at points :math:`\nabla \phi_i(\mathbf{x}_p)`
  * ``self.rho0``     -- :math:`n_p \times 1` initial density vector :math:`\rho_p^0`
  * ``self.rho``      -- :math:`n_p \times 1` density vector :math:`\rho_p^t`
  * ``self.V0``       -- :math:`n_p \times 1` initial volume vector :math:`V_p^0`
  * ``self.V``        -- :math:`n_p \times 1` volume vector :math:`V_p`
  * ``self.F``        -- :math:`n_p \times (d \times d)` deformation gradient tensors :math:`F_p`
  * ``self.sigma``    -- :math:`n_p \times (d \times d)` stress tensors :math:`\sigma_p`
  * ``self.epsilon``  -- :math:`n_p \times (d \times d)` strain-rate tensors :math:`\dot{\epsilon}_p`
  * ``self.I``        -- :math:`n_p \times (d \times d)` identity tensors :math:`I`
  * ``self.vrt``      -- :math:`n_p \times n_v` nodal vertex indices for each particle
  """
  def __init__(self, name, x, u, m=None, V=None, rho=None):
    """
    """
    self.this     = super(type(self), self)  # pointer to this base class
    self.name     = name

    s = "::: INITIALIZING BASE MATERIAL `%s` :::" % name
    print_text(s, cls=self.this)

    if     (m is None and rho is None and V is None) \
        or (m is not None and (rho is not None or V is not None)):
      s = ">>> MATERIALS ARE CREATED EITHER WITH MASS VECTOR `m` OR WITH A " \
          "DENSITY `rho` AND VOLUME VECTOR `V` <<<"
      print_text(s , 'red', 1)
      sys.exit(1)

    # cpp arguments, if needed these are set in child classes
    # (find example child classes after this class) :
    self.cpp_args = ()

    self.n        = len(x[:,0])        # number of particles
    self.d        = len(x[0])          # topological dimension
    self.x        = x                  # position vector
    self.u        = u                  # velocity vector
    self.m        = m                  # mass vector
    self.rho      = rho                # density vector
    self.rho0     = None               # initial density vector
    self.V        = V                  # volume vector
    self.V0       = None               # initial volume vector
    self.a        = None               # acceleration vector
    self.grad_u   = None               # velocity gradient tensor
    self.vrt      = None               # grid nodal indicies for points
    self.phi      = None               # grid basis values at points
    self.grad_phi = None               # grid basis gradient values at points
    self.F        = None               # deformation gradient tensor
    self.sigma    = None               # stress tensor
    self.epsilon  = None               # strain-rate tensor

    # identity tensors :
    self.I        = np.array([np.identity(self.d).flatten()]*self.n)
    
  def cpp_module(self):
    r"""
    return the appropriate cpp module to instantiate this material.

    This method must be implemented by child classes.
    """
    raiseNotDefined()
  
  def get_cpp_material(self, element):
    r"""
    instantiate and return the cpp module for this material with element type
    ``element``.
    """
    # create the underlying C++ code representation :
    mod = self.cpp_module()(self.name,
                            self.n, 
                            self.x.flatten(),
                            self.u.flatten(),
                            element, *self.cpp_args)

    # if the material has been initialized with a mass vector :
    if self.m is not None:
      mod.set_initialized_by_mass(True)
      mod.initialize_mass(self.m)

    # otherwise, the material has been initialized with a constant density :
    if self.rho is not None and self.V is not None:
      mod.set_initialized_by_mass(False)
      mod.initialize_volume(self.V)
      mod.initialize_mass_from_density(self.rho)

    return mod

  def set_cpp_material(self, cpp_mat):
    r"""
    Instantiante the C++ code object ``self.cpp_mat`` for this material
    to ``cpp_mat``.
    
    :param element: The FEniCS element used.
    :type element: :class:`~fenics.FiniteElement`
    """
    self.cpp_mat = cpp_mat 

  def color(self):
    r"""
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

    # calculate particle strain-rate tensors :
    for grad_u_p in self.grad_u:
      dudx   = grad_u_p[0]
      dudy   = grad_u_p[1]
      dvdx   = grad_u_p[2]
      dvdy   = grad_u_p[3]
      eps_xx = dudx
      eps_xy = 0.5*(dudy + dvdx)
      eps_yy = dvdy
      eps    = np.array( [eps_xx, eps_xy, eps_xy, eps_yy], dtype=float )
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
    plt.plot(self.x[:,0], self.x[:,1], 'r*')
    plt.axis('equal')
    plt.show()

  def retrieve_cpp_vrt(self):
    """
    Get the nodal indices at each particle's position from the C++ model.
    """
    self.vrt_1    = np.array(self.cpp_mat.get_vrt_1(),    dtype=int)
    self.vrt_2    = np.array(self.cpp_mat.get_vrt_2(),    dtype=int)
    self.vrt_3    = np.array(self.cpp_mat.get_vrt_3(),    dtype=int)
    self.vrt_4    = np.array(self.cpp_mat.get_vrt_4(),    dtype=int)

  def retrieve_cpp_phi(self):
    """
    Get the grid basis function values evaluated at each particle's current position from the C++ model.
    """
    self.phi_1    = np.array(self.cpp_mat.get_phi_1(),    dtype=float)
    self.phi_2    = np.array(self.cpp_mat.get_phi_2(),    dtype=float)
    self.phi_3    = np.array(self.cpp_mat.get_phi_3(),    dtype=float)
    self.phi_4    = np.array(self.cpp_mat.get_phi_4(),    dtype=float)

  def retrieve_cpp_grad_phi(self):
    """
    Get the particle basis function gradient values from the C++ model.
    """
    self.grad_phi_1x   = np.array(self.cpp_mat.get_grad_phi_1x(),  dtype=float)
    self.grad_phi_1y   = np.array(self.cpp_mat.get_grad_phi_1y(),  dtype=float)
    self.grad_phi_1z   = np.array(self.cpp_mat.get_grad_phi_1z(),  dtype=float)
    self.grad_phi_2x   = np.array(self.cpp_mat.get_grad_phi_2x(),  dtype=float)
    self.grad_phi_2y   = np.array(self.cpp_mat.get_grad_phi_2y(),  dtype=float)
    self.grad_phi_2z   = np.array(self.cpp_mat.get_grad_phi_2z(),  dtype=float)
    self.grad_phi_3x   = np.array(self.cpp_mat.get_grad_phi_3x(),  dtype=float)
    self.grad_phi_3y   = np.array(self.cpp_mat.get_grad_phi_3y(),  dtype=float)
    self.grad_phi_3z   = np.array(self.cpp_mat.get_grad_phi_3z(),  dtype=float)
    self.grad_phi_4x   = np.array(self.cpp_mat.get_grad_phi_4x(),  dtype=float)
    self.grad_phi_4y   = np.array(self.cpp_mat.get_grad_phi_4y(),  dtype=float)
    self.grad_phi_4z   = np.array(self.cpp_mat.get_grad_phi_4z(),  dtype=float)

  def retrieve_cpp_grad_u(self):
    """
    Get the particle velocity gradient from the C++ model.
    """
    self.grad_u_xx = np.array(self.cpp_mat.get_grad_u_xx(), dtype=float)
    self.grad_u_xy = np.array(self.cpp_mat.get_grad_u_xy(), dtype=float)
    self.grad_u_xz = np.array(self.cpp_mat.get_grad_u_xz(), dtype=float)
    self.grad_u_yx = np.array(self.cpp_mat.get_grad_u_yx(), dtype=float)
    self.grad_u_yy = np.array(self.cpp_mat.get_grad_u_yy(), dtype=float)
    self.grad_u_yz = np.array(self.cpp_mat.get_grad_u_yz(), dtype=float)
    self.grad_u_zx = np.array(self.cpp_mat.get_grad_u_zx(), dtype=float)
    self.grad_u_zy = np.array(self.cpp_mat.get_grad_u_zy(), dtype=float)
    self.grad_u_zz = np.array(self.cpp_mat.get_grad_u_zz(), dtype=float)

  def retrieve_cpp_x(self):
    """
    Get the particle position vector from the C++ model.
    """
    self.x = np.array(self.cpp_mat.get_x(), dtype=float)
    self.y = np.array(self.cpp_mat.get_y(), dtype=float)
    self.z = np.array(self.cpp_mat.get_z(), dtype=float)

  def retrieve_cpp_u(self):
    """
    Get the particle velocity vector from the C++ model.
    """
    self.u_x = np.array(self.cpp_mat.get_u_x(), dtype=float)
    self.u_y = np.array(self.cpp_mat.get_u_y(), dtype=float)
    self.u_z = np.array(self.cpp_mat.get_u_z(), dtype=float)

  def retrieve_cpp_a(self):
    """
    Get the particle acceleration vector from the C++ model.
    """
    self.a_x = np.array(self.cpp_mat.get_a_x(), dtype=float)
    self.a_y = np.array(self.cpp_mat.get_a_y(), dtype=float)
    self.a_z = np.array(self.cpp_mat.get_a_z(), dtype=float)

  def retrieve_cpp_F(self):
    """
    Get the particle deformation gradient from the C++ model.
    """
    self.F_xx = np.array(self.cpp_mat.get_F_xx(), dtype=float)
    self.F_xy = np.array(self.cpp_mat.get_F_xy(), dtype=float)
    self.F_xz = np.array(self.cpp_mat.get_F_xz(), dtype=float)
    self.F_yx = np.array(self.cpp_mat.get_F_yx(), dtype=float)
    self.F_yy = np.array(self.cpp_mat.get_F_yy(), dtype=float)
    self.F_yz = np.array(self.cpp_mat.get_F_yz(), dtype=float)
    self.F_zx = np.array(self.cpp_mat.get_F_zx(), dtype=float)
    self.F_zy = np.array(self.cpp_mat.get_F_zy(), dtype=float)
    self.F_zz = np.array(self.cpp_mat.get_F_zz(), dtype=float)

  def retrieve_cpp_epsilon(self):
    """
    Get the particle strain tensor from the C++ model.
    """
    self.epsilon_xx = np.array(self.cpp_mat.get_epsilon_xx(), dtype=float)
    self.epsilon_xy = np.array(self.cpp_mat.get_epsilon_xy(), dtype=float)
    self.epsilon_xz = np.array(self.cpp_mat.get_epsilon_xz(), dtype=float)
    self.epsilon_yy = np.array(self.cpp_mat.get_epsilon_yy(), dtype=float)
    self.epsilon_yz = np.array(self.cpp_mat.get_epsilon_yz(), dtype=float)
    self.epsilon_zz = np.array(self.cpp_mat.get_epsilon_zz(), dtype=float)

  def retrieve_cpp_sigma(self):
    """
    Get the particle stress tensor from the C++ model.
    """
    self.sigma_xx = np.array(self.cpp_mat.get_sigma_xx(), dtype=float)
    self.sigma_xy = np.array(self.cpp_mat.get_sigma_xy(), dtype=float)
    self.sigma_xz = np.array(self.cpp_mat.get_sigma_xz(), dtype=float)
    self.sigma_yy = np.array(self.cpp_mat.get_sigma_yy(), dtype=float)
    self.sigma_yz = np.array(self.cpp_mat.get_sigma_yz(), dtype=float)
    self.sigma_zz = np.array(self.cpp_mat.get_sigma_zz(), dtype=float)
    
  def retrieve_cpp_rho0(self):
    """
    Get the initial particle density vector from the C++ model.
    """
    self.rho0 = np.array(self.cpp_mat.get_rho0(), dtype=float)
    
  def retrieve_cpp_rho(self):
    """
    Get the current particle density vector from the C++ model.
    """
    self.rho  = np.array(self.cpp_mat.get_rho(),  dtype=float)

  def retrieve_cpp_V(self):
    """
    Get the current particle volume vector from the C++ model.
    """
    self.V    = np.array(self.cpp_mat.get_V(),    dtype=float)

  def retrieve_cpp_V0(self):
    """
    Get the initial particle volume vector from the C++ model.
    """
    self.V0   = np.array(self.cpp_mat.get_V0(),   dtype=float)




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
  def __init__(self, name, x, u, E, nu, m=None, V=None, rho=None):
    """
    """
    s = "::: INITIALIZING ELASTIC MATERIAL `%s` :::" % name
    print_text(s, cls=self)

    Material.__init__(self, name, x, u, m, V, rho)

    self.E        = E         # Young's modulus
    self.nu       = nu        # Poisson's ratio

    self.cpp_args = {self.E, self.nu}

    # Lamé parameters :
    self.mu       = E / (2.0*(1.0 + nu))
    self.lmbda    = E*nu / ((1.0 + nu)*(1.0 - 2.0*nu))
  
  def cpp_module(self):
    r"""
    return the appropriate cpp module to instantiate this material.

    This method must be implemented by child classes.
    """
    # create the underlying C++ code representation :
    return mpm_module.MPMElasticMaterial

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
      trace_eps  = epsilon_p[0] + epsilon_p[3]
      c1         = 2.0 * self.mu
      c2         = self.lmbda * trace_eps
      sig_xx     = c1 * epsilon_p[0] + c2
      sig_xy     = c1 * epsilon_p[1] 
      sig_yy     = c1 * epsilon_p[3] + c2
      sigma_p    = np.array( [sig_xx, sig_xy, sig_xy, sig_yy], dtype=float )
      sigma.append(sigma_p)
    
    # return particle Cauchy stress tensors :
    return np.array(sigma, dtype=float)




class ImpenetrableMaterial(Material):
  r"""
  Representation of an impenetrable boundary with initial conditions given by the 
  parameters ``m``, ``x``, and ``u``.

  :param m: particle mass vector :math:`\mathbf{m}_p`
  :param x: particle position vector :math:`\mathbf{x}_p`
  :param u: particle velocity vector :math:`\mathbf{u}_p`
  :type m: :class:`~numpy.ndarray`
  :type x: :class:`~numpy.ndarray`
  :type u: :class:`~numpy.ndarray`
  """
  def __init__(self, name, x, u, m=None, V=None, rho=None):
    """
    """
    s = "::: INITIALIZING IMPENETRABLE MATERIAL `%s` :::" % name
    print_text(s, cls=self)

    Material.__init__(self, name, x, u, m, V, rho)
  
  def cpp_module(self):
    r"""
    return the appropriate cpp module to instantiate this material.

    This method must be implemented by child classes.
    """
    # create the underlying C++ code representation :
    return mpm_module.MPMImpenetrableMaterial

  def color(self):
    return '150'



