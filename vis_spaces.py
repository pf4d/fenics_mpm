import dolfin

mesh = dolfin.UnitSquareMesh(2,2)
#V = dolfin.FunctionSpace(mesh, 'CG', 1)
#v = dolfin.Function(V)
#v.vector()[4] = 1
#dolfin.plot(v)
#dolfin.interactive()

## continue....
V = dolfin.FunctionSpace(mesh, 'R', 2)
v = dolfin.Function(V)
v.vector()[1] = 1

mesh2 = dolfin.UnitSquareMesh(100,100)
V2 = dolfin.FunctionSpace(mesh2, 'CG', 2)
v2 = dolfin.interpolate(v, V2)
dolfin.plot(v2)
dolfin.interactive()

