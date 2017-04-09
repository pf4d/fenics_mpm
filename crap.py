from fenics import *

mesh = UnitSquareMesh(1,1)
V    = FunctionSpace(mesh, 'CG', 1)

element = V.element()
dofmap = V.dofmap()
for cell in cells(mesh):
  print cell.index(), dofmap.cell_dofs(cell.index())
  print cell.get_vertex_coordinates() 
  print element.tabulate_dof_coordinates(cell)
  #print dofmap.tabulate_entity_dofs(cell)

