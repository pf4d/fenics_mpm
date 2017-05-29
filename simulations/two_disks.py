from fenics_mpm import *

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


#===============================================================================
# model properties :
out_dir    = 'output/'   # output directory
n_x        = 20          # number of grid x- and y-divisions
E          = 1000.0      # Young's modulus
nu         = 0.3         # Poisson's ratio
u_mag      = 0.1         # velocity magnitude   [m/s]
m_mag      = 0.012       # particle mass        [kg]
dt_save    = 0.01        # time between saves   [s]
dt         = 0.002       # time-step            [s]
t0         = 0.0         # starting time        [s]
tf         = 10          # ending time          [s]

# calculate the number of iterations between saves :
save_int   = int(dt_save / dt)

# create a material :
n          = 1000        # number of particles
r_max      = 0.15        # disk radius          [m]

# upper-right disk :
X1         = sunflower(n, 2, 0.66, 0.66, r_max)
M1         =  m_mag * np.ones(n)
U1         =  u_mag * np.ones([n,2])

# lower-left disk : 
X2         = sunflower(n, 2, 0.34, 0.34, r_max)
M2         =  m_mag * np.ones(n)
U2         = -u_mag * np.ones([n,2])

# corresponding Material objects : 
M1         = ElasticMaterial(M1, X1, U1, E, nu)
M2         = ElasticMaterial(M2, X2, U2, E, nu)

# the finite-element mesh used :    
mesh       = UnitSquareMesh(n_x, n_x)

# the exterior boundary :
class Boundary(SubDomain):
  def inside(self, x, on_boundary):
    return on_boundary
boundary = Boundary()

# initialize the model :
grid_model = GridModel(mesh, out_dir, verbose=False)

# set the velocity along the entire boundary to 0.0 :
grid_model.set_boundary_conditions(boundary, 0.0)

# create the main model to perform MPM calculations :
model      = Model(out_dir, grid_model, dt, verbose=False)

# add the materials to the model :
model.add_material(M1)
model.add_material(M2)

# files for saving grid variables :
m_file = File(out_dir + '/m.pvd')
u_file = File(out_dir + '/u.pvd')
a_file = File(out_dir + '/a.pvd')
f_file = File(out_dir + '/f.pvd')
 
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

# perform the material point method algorithm :
model.mpm(t_start = t0, t_end = tf, cb_ftn = cb_ftn)



