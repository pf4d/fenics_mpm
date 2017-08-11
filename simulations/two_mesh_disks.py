from fenics_mpm import *
from mshr       import *

#===============================================================================
# model properties :
out_dir    = 'output/'   # output directory
n_x        = 100         # number of grid x- and y-divisions
E          = 1000.0      # Young's modulus
nu         = 0.3         # Poisson's ratio
rho        = 1000.0      # material density     [kg/m^3]
u_mag      = 0.1         # velocity magnitude   [m/s]
dt_save    = 0.01        # time between saves   [s]
dt         = 0.0002      # time-step            [s]
t0         = 0.0         # starting time        [s]
tf         = 1.5         # ending time          [s]

# calculate the number of iterations between saves :
save_int   = int(dt_save / dt)

# create a material :
r_max      = 0.15        # disk radius          [m]
res        = 1000        # disk mesh resolution

# upper-right disk :
domain1    = Circle(Point(0.66, 0.66), r_max)
mesh1      = generate_mesh(domain1, res)
X1,V1      = calculate_mesh_midpoints_and_volumes(mesh1)
n1         = np.shape(X1)[0]
U1         = -u_mag * np.ones([n1,2])

# lower-left disk : 
domain2    = Circle(Point(0.34, 0.34), r_max)
mesh2      = generate_mesh(domain2, res)
X2,V2      = calculate_mesh_midpoints_and_volumes(mesh2)
n2         = np.shape(X2)[0]
U2         =  u_mag * np.ones([n2,2])

# corresponding Material objects : 
disk1      = ElasticMaterial('disk1', X1, U1, E, nu, V=V1, rho=rho)
disk2      = ElasticMaterial('disk2', X2, U2, E, nu, V=V2, rho=rho)

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
model      = Model(out_dir, grid_model, dt, verbose=True)

# add the materials to the model :
model.add_material(disk1)
model.add_material(disk2)

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



