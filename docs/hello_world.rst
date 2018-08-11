Hello fenics_mpm!
=======================

This tutorial walks through the operations required to perform the two elastic disk problem of [sulsky_1994]_.

First, import the ``fenics_mpm`` package::

  from fenics_mpm import *

:class:`~material.Material` s are initialized with particle (superscript :math:`\mathrm{p}`) position vector :math:`\mathbf{x}^{\mathrm{p}}` and velocity vector :math:`\mathbf{x}^{\mathrm{p}}`, as well as specifying either the mass :math:`m^{\mathrm{p}}` vector alone or both the density :math:`\rho^{\mathrm{p}}` and volume :math:`V^{\mathrm{p}}` vectors.
For example, we can use the ``sunflower`` pattern defined `here <https://doi.org/10.1016/0025-5564(79)90080-4>`_ to create our two disks positions::

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

Next, we can define some model parameters::

  in_dir     = 'data/'     # input directory
  out_dir    = 'output/'   # output directory
  n_x        = 100         # number of grid x- and y-divisions
  E          = 1000.0      # Young's modulus
  nu         = 0.3         # Poisson's ratio
  rho        = 1000.0      # material density     [kg/m^3]
  r_max      = 0.15        # disk radius          [m]
  u_mag      = 0.1         # velocity magnitude   [m/s]
  dt_save    = 0.01        # time between saves   [s]
  dt         = 0.0002      # time-step            [s]
  t0         = 0.0         # starting time        [s]
  tf         = 1.5         # ending time          [s]
  
  # calculate the number of iterations between saves :
  save_int   = int(dt_save / dt)

New we can create two elastic :class:`~material.Material` s using the :class:`~material.ElasticMaterial` class.
First generate coordinate vector :math:`\mathbf{x}^{\mathrm{p}}`, mass :math:`m^{\mathrm{p}}`, and velocity vector :math:`\mathbf{u}^{\mathrm{p}}` for the upper-right disk::

  X1         = sunflower(n, 2, 0.66, 0.66, r_max)
  M1         =  m_mag * np.ones(n)
  U1         = -u_mag * np.ones([n,2])

then the lower-left disk::
 
  X2         = sunflower(n, 2, 0.34, 0.34, r_max)
  M2         = m_mag * np.ones(n)
  U2         = u_mag * np.ones([n,2])

instantiate the objects::
 
  ball1      = ElasticMaterial('disk1', X1, U1, E, nu, m=M1)
  ball2      = ElasticMaterial('disk2', X2, U2, E, nu, m=M2)

create a FEniCS finite-element :class:`~dolfin.cpp.mesh.Mesh` object to define the computational domain::

  mesh       = UnitSquareMesh(n_x, n_x)
  
initialize the finite-element :class:`~gridmodel.GridModel` and MPM :class:`~model.Model`::

  grid_model = GridModel(mesh, out_dir, verbose=False)
  model      = Model(out_dir, grid_model, dt, verbose=False)

add the :class:`~material.ElasticMaterial` s to the model you just created::

  model.add_material(ball_1)
  model.add_material(ball_2)

Next, we can create a function that will be called each iteration to save data as ``pvd`` files viewable with ParaView::

  # files for saving grid variables :
  m_file = File(out_dir + '/m.pvd')  # mass
  u_file = File(out_dir + '/u.pvd')  # velocity
  a_file = File(out_dir + '/a.pvd')  # acceleration
  f_file = File(out_dir + '/f.pvd')  # internal force vector
   
  # callback function saves result :
  def cb_ftn():
    if model.iter % save_int == 0:
      model.retrieve_cpp_grid_m()
      model.retrieve_cpp_grid_U3()
      model.retrieve_cpp_grid_f_int()
      model.retrieve_cpp_grid_a3()
      grid_model.save_pvd(grid_model.m,     'm',     f=m_file, t=model.t)
      grid_model.save_pvd(grid_model.U3,    'U3',    f=u_file, t=model.t)
      grid_model.save_pvd(grid_model.a3,    'a3',    f=a_file, t=model.t)
      grid_model.save_pvd(grid_model.f_int, 'f_int', f=f_file, t=model.t)
  
and finally perform the material-point method from :math:`t =` ``t0`` to :math:`t =` ``tf`` with :func:`~model.Model.mpm`::

  model.mpm(t_start = t0, t_end = tf, cb_ftn = cb_ftn)

Currently, the simulation will terminate with an error when a particle moves beyond the grid.  Check the ``output_dir`` directory for ``.pvd`` files.  Open them with ``ParaView``.  You can find the entire simulation file in the ``fenics_mpm/simulations/`` directory.




