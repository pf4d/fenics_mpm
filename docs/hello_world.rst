Hello fenics_mpm!
=======================

This tutorial walks through the operations required to perform the two elastic disk problem of [sulsky_1994]_.

First, import the ``fenics_mpm`` package::

  from fenics_mpm import *

Now we can use the ``sunflower`` pattern defined `here <https://doi.org/10.1016/0025-5564(79)90080-4/>`_ to create our two disks::

  # radial measure :  
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

define some model parameters::

  out_dir    = 'output/'   # output directory
  n_x        = 20          # number of grid x- and y-divisions
  dt         = 0.002       # time-step (seconds)
  E          = 1000.0      # Young's modulus
  nu         = 0.3         # Poisson's ratio
  u_mag      = 0.1         # velocity magnitude (m/s)
  m_mag      = 0.15        # particle mass
  n          = 100         # number of particles
  r_max      = 0.15        # disk radius

create two elastic materials using the :class:`~material.ElasticMaterial` class.  First generate coordinate vector :math:`\mathbf{x}_p`, mass :math:`m_p`, and velocity vector :math:`\mathbf{u}_p` for the upper-right disk::

  X1         = sunflower(n, 2, 0.66, 0.66, r_max)
  M1         =  m_mag * np.ones(n)
  U1         = -u_mag * np.ones([n,2])

then the lower-left disk::
 
  X2         = sunflower(n, 2, 0.34, 0.34, r_max)
  M2         = m_mag * np.ones(n)
  U2         = u_mag * np.ones([n,2])

instantiate the objects::
 
  M1         = ElasticMaterial(M1, X1, U1, E, nu)
  M2         = ElasticMaterial(M2, X2, U2, E, nu)

create a FEniCS finite-element :class:`~fenics.mesh`` object::

  mesh       = UnitSquareMesh(n_x, n_x)
  
initialize the model::

  grid_model = GridModel(mesh, out_dir)
  model      = Model(out_dir, grid_model, dt)

add the materials to the model you just created::

  model.add_material(M1)
  model.add_material(M2)
  
and finally perform the material point method algorithm from ``t=0`` to ``t=5``::

  model.mpm(0, 5)

Currently, the simulation will terminate with an error when a particle moves beyond the grid.  Check the ``output_dir`` directory for ``.pvd`` files.  Open them with ``ParaView``.  You can find the entire simulation file in the ``fenics_mpm/simulations/`` directory.




