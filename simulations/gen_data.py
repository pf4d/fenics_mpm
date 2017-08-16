from fenics_mpm import *
from mshr       import *

#===============================================================================
# model properties :
out_dir    = 'data/high/'# output directory

# create a material :
r_max      = 0.15        # disk radius          [m]
res        = 100         # disk mesh resolution

# upper-right disk :
domain1    = Circle(Point(0.66, 0.66), r_max)
mesh1      = generate_mesh(domain1, res)
X1,V1      = calculate_mesh_midpoints_and_volumes(mesh1)

# lower-left disk : 
domain2    = Circle(Point(0.34, 0.34), r_max)
mesh2      = generate_mesh(domain2, res)
X2,V2      = calculate_mesh_midpoints_and_volumes(mesh2)

# load the data created by the "gen_data.py" script :
X1 = np.savetxt(out_dir + 'X1.txt', X1)
X2 = np.savetxt(out_dir + 'X2.txt', X2)
V1 = np.savetxt(out_dir + 'V1.txt', V1)
V2 = np.savetxt(out_dir + 'V2.txt', V2)
