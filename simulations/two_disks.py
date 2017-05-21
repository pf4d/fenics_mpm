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
dt         = 0.002       # time-step (seconds)
E          = 1000.0      # Young's modulus
nu         = 0.3         # Poisson's ratio
u_mag      = 0.1         # velocity magnitude (m/s)
m_mag      = 0.15        # particle mass

# create a material :
n          = 2         # number of particles
r_max      = 0.15        # disk radius

# upper-right disk :
X1         = sunflower(n, 2, 0.66, 0.66, r_max)
M1         =  m_mag * np.ones(n)
U1         = -u_mag * np.ones([n,2])

# lower-left disk : 
X2         = sunflower(n, 2, 0.34, 0.34, r_max)
M2         = m_mag * np.ones(n)
U2         = u_mag * np.ones([n,2])

# corresponding Material objects : 
M1         = ElasticMaterial(M1, X1, U1, E, nu)
M2         = ElasticMaterial(M2, X2, U2, E, nu)

# the finite-element mesh used :    
mesh       = UnitSquareMesh(n_x, n_x)

# initialize the model :
grid_model = GridModel(mesh, out_dir)
model      = Model(out_dir, grid_model, dt)

# add the materials to the model :
model.add_material(M1)
model.add_material(M2)

# perform the material point method algorithm :
model.mpm(0, 5)



