#include "MPMModel.h"

using namespace dolfin;

MPMModel::MPMModel(const FunctionSpace& V)
{
  Q       = &V;
  mesh    = V.mesh();
  element = V.element();
  
  gdim    = mesh->geometry().dim();
  sdim    = element->space_dimension();
      
  phi.resize(element->space_dimension());
  vrt.resize(element->space_dimension());
  grad_phi.resize(gdim*sdim);
}

void MPMModel::add_material(const MPMMaterial& M)
{
  materials.push_back(&M);
}

void MPMModel::eval(const Array<double>& x)
{

  const Point x_pt(gdim, x.data());
  cell_id = mesh->bounding_box_tree()->compute_first_entity_collision(x_pt);
  cell.reset(new Cell( *mesh, (size_t) cell_id));

  cell->get_vertex_coordinates(vertex_coordinates);
  element->evaluate_basis_all(&phi[0], &x[0],
                              vertex_coordinates.data(),
                              cell_orientation);
 
  element->evaluate_basis_derivatives_all(deriv_order, &grad_phi[0], &x[0],
                                          vertex_coordinates.data(),
                                          cell_orientation);

  for (std::size_t j = 0; j < Q->dofmap()->cell_dofs(0).size(); j++)
    vrt[j] = Q->dofmap()->cell_dofs(cell->index())[j];
}



