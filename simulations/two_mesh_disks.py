from fenics_mpm import *
from fenics     import UnitSquareMesh, File

#===============================================================================
# model properties :
in_dir     = 'data/'     # input directory
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

# load the data created by the "gen_data.py" script :
X1 = np.loadtxt(in_dir + 'X1.txt')
X2 = np.loadtxt(in_dir + 'X2.txt')
V1 = np.loadtxt(in_dir + 'V1.txt')
V2 = np.loadtxt(in_dir + 'V2.txt')

# disk one velocity :
n1 = np.shape(X1)[0]
U1 = -u_mag * np.ones([n1,2])

# disk two velocity :
n2 = np.shape(X2)[0]
U2 =  u_mag * np.ones([n2,2])

# corresponding Material objects : 
disk1 = ElasticMaterial('disk1', X1, U1, E, nu, V=V1, rho=rho)
disk2 = ElasticMaterial('disk2', X2, U2, E, nu, V=V2, rho=rho)

# the finite-element mesh used :    
mesh  = UnitSquareMesh(n_x, n_x)

# initialize the model :
grid_model = GridModel(mesh, out_dir, verbose=False)

# create the main model to perform MPM calculations :
model      = Model(out_dir, grid_model, dt, verbose=False)

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



