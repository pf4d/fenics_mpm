#include "MPMModel.h"

using namespace dolfin;

MPMModel::MPMModel(const FunctionSpace& V, const unsigned int N_p) : n_p(N_p)
{
  printf("n_p = %u \n", n_p);

  Q       = &V;
  mesh    = V.mesh();
  element = V.element();
  
  gdim    = mesh->geometry().dim();
  sdim    = element->space_dimension();

  phi.resize(n_p);
  vrt.resize(n_p);
  grad_phi.resize(n_p);
  for (unsigned int i = 0; i < n_p; i++)
  {
    phi[i].resize(element->space_dimension());
    vrt[i].resize(element->space_dimension());
    grad_phi[i].resize(gdim*sdim);
  }
}                              

void MPMModel::eval(const Array<double>& x)
{
  Array<double> x_temp(gdim);

  for (std::size_t i = 0; i < n_p; i++)
  {
    for (std::size_t j = 0; j < gdim; j++)
    {
      x_temp[j] = x[i*gdim + j];
    }
    const Point x_pt(gdim, x_temp.data());

    cell_id = mesh->bounding_box_tree()->compute_first_entity_collision(x_pt);
    cell.reset(new Cell( *mesh, (size_t) cell_id));
    
    cell->get_vertex_coordinates(vertex_coordinates);
    element->evaluate_basis_all(&phi[i][0], 
                                &x_temp[0],
                                vertex_coordinates.data(),
                                cell_orientation);
    
    element->evaluate_basis_derivatives_all(deriv_order,
                                            &grad_phi[i][0],
                                            &x_temp[0],
                                            vertex_coordinates.data(),
                                            cell_orientation);
    
    for (std::size_t j = 0; j < Q->dofmap()->cell_dofs(0).size(); j++)
      vrt[i][j] = Q->dofmap()->cell_dofs(cell->index())[j];
   }
}



