# -*- coding: iso-8859-15 -*-

import numpy                  as np
import matplotlib.pyplot      as plt
from   fenics_mpm.helper      import print_text, raiseNotDefined
from   fenics_mpm             import mpm_module


class Material(object):
  r"""
  Representation of an abstract material with initial conditions given by the 
  parameters length :math:`n_p` :class:`~numpy.ndarray` vectors ``m``, ``x`` 
  and ``u``.

  :param m: particle mass vector :math:`\mathbf{m}_p`
  :param x: particle position vector :math:`\mathbf{x}_p`
  :param u: particle velocity vector :math:`\mathbf{u}_p`
  :type m: :class:`~numpy.ndarray`
  :type x: :class:`~numpy.ndarray`
  :type u: :class:`~numpy.ndarray`

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
  * ``self.rho``      -- :math:`n_p \times 1` density vector :math:`\rho_p`
  * ``self.V0``       -- :math:`n_p \times 1` initial volume vector :math:`V_p^0`
  * ``self.V``        -- :math:`n_p \times 1` volume vector :math:`V_p`
  * ``self.F``        -- :math:`n_p \times (d \times d)` deformation gradient tensors :math:`F_p`
  * ``self.sigma``    -- :math:`n_p \times (d \times d)` stress tensors :math:`\sigma_p`
  * ``self.epsilon``  -- :math:`n_p \times (d \times d)` strain-rate tensors :math:`\dot{\epsilon}_p`
  * ``self.I``        -- :math:`n_p \times (d \times d)` identity tensors :math:`I`
  """
  def __init__(self, m, x, u):
    """
    """
    self.this     = super(type(self), self)  # pointer to this base class

    s = "::: INITIALIZING BASE MATERIAL :::"
    print_text(s, cls=self.this)

    self.n        = len(x[:,0])        # number of particles
    self.d        = len(x[0])          # topological dimension
    self.m        = m                  # mass vector
    self.x        = x                  # position vector
    self.u        = u                  # velocity vector
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
    self.I        = np.array([np.identity(self.d).flatten()]*self.n)
    
  def get_cpp_material(self, element):
    r"""
    return the appropriate cpp module to instantiate this material.

    This method must be implemented by child classes.
    """
    raiseNotDefined()

  def set_cpp_material(self, cpp_mat):
    r"""
    Instantiante the C++ code for this material with .
    
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
    """
    vrt         = []
    # iterate through particle positions :
    # append these to a list corresponding with particles : 
    for i in range(self.cpp_mat.get_num_particles()):
      vrt.append(self.cpp_mat.get_vrt(i))
    self.vrt      = np.array(vrt)

  def retrieve_cpp_phi(self):
    """
    """
    phi         = []
    # iterate through particle positions :
    # append these to a list corresponding with particles : 
    for i in range(self.cpp_mat.get_num_particles()):
      phi.append(self.cpp_mat.get_phi(i))
    self.phi      = np.array(phi,          dtype=float)

  def retrieve_cpp_grad_phi(self):
    """
    """
    grad_phi    = []
    # iterate through particle positions :
    # append these to a list corresponding with particles : 
    for i in range(self.cpp_mat.get_num_particles()):
      grad_phi.append(self.cpp_mat.get_grad_phi(i))
    self.grad_phi = np.array(grad_phi,     dtype=float)

  def retrieve_cpp_grad_u(self):
    """
    """
    grad_u_p_v  = []
    # iterate through particle positions :
    # append these to a list corresponding with particles : 
    for i in range(self.cpp_mat.get_num_particles()):
      grad_u_p_v.append(self.cpp_mat.get_grad_u(i))
    self.grad_u   = np.array(grad_u_p_v,   dtype=float)

  def retrieve_cpp_x(self):
    """
    """
    x_p_v       = []
    # iterate through particle positions :
    # append these to a list corresponding with particles : 
    for i in range(self.cpp_mat.get_num_particles()):
      x_p_v.append(self.cpp_mat.get_x(i))
    self.x        = np.array(x_p_v,        dtype=float)

  def retrieve_cpp_u(self):
    """
    """
    u_p_v       = []
    # iterate through particle positions :
    # append these to a list corresponding with particles : 
    for i in range(self.cpp_mat.get_num_particles()):
      u_p_v.append(self.cpp_mat.get_u(i))
    self.u        = np.array(u_p_v,        dtype=float)

  def retrieve_cpp_a(self):
    """
    """
    a_p_v       = []
    # iterate through particle positions :
    # append these to a list corresponding with particles : 
    for i in range(self.cpp_mat.get_num_particles()):
      a_p_v.append(self.cpp_mat.get_a(i))
    self.a        = np.array(a_p_v,        dtype=float)

  def retrieve_cpp_F(self):
    """
    """
    F_p_v       = []
    # iterate through particle positions :
    # append these to a list corresponding with particles : 
    for i in range(self.cpp_mat.get_num_particles()):
      F_p_v.append(self.cpp_mat.get_F(i))
    self.F        = np.array(F_p_v,        dtype=float)

  def retrieve_cpp_epsilon(self):
    """
    """
    epsilon_p_v = []
    # iterate through particle positions :
    # append these to a list corresponding with particles : 
    for i in range(self.cpp_mat.get_num_particles()):
      epsilon_p_v.append(self.cpp_mat.get_epsilon(i))
    self.epsilon  = np.array(epsilon_p_v,  dtype=float)

  def retrieve_cpp_sigma(self):
    """
    """
    sigma_p_v   = []
    # iterate through particle positions :
    # append these to a list corresponding with particles : 
    for i in range(self.cpp_mat.get_num_particles()):
      sigma_p_v.append(self.cpp_mat.get_sigma(i))
    self.sigma    = np.array(sigma_p_v,    dtype=float)
    
  def retrieve_cpp_rho(self):
    """
    """
    self.rho = np.array(self.cpp_mat.get_rho(), dtype=float)

  def retrieve_cpp_V(self):
    """
    """
    self.V   = np.array(self.cpp_mat.get_V(),   dtype=float)

  def retrieve_cpp_V0(self):
    """
    """
    self.V0  = np.array(self.cpp_mat.get_V0(),  dtype=float)


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
  
  def get_cpp_material(self, element):
    r"""
    return the appropriate cpp module to instantiate this material.

    This method must be implemented by child classes.
    """
    return mpm_module.MPMElasticMaterial(self.m,
                                         self.x.flatten(),
                                         self.u.flatten(),
                                         element,
                                         self.E, self.nu)

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
  def __init__(self, m, x, u):
    """
    """
    s = "::: INITIALIZING IMPENETRABLE MATERIAL :::"
    print_text(s, cls=self)

    Material.__init__(self, m, x, u)

  def get_cpp_material(self, element):
    r"""
    return the appropriate cpp module to instantiate this material.

    This method must be implemented by child classes.
    """
    return mpm_module.MPMImpenetrableMaterial(self.m,
                                              self.x.flatten(),
                                              self.u.flatten(),
                                              element)

  def color(self):
    return '150'



