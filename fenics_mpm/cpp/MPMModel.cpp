#include "MPMModel.h"

using namespace dolfin;

MPMModel::MPMModel(const FunctionSpace& V)
{
  Q       = &V;
  mesh    = V.mesh();
  element = V.element();
  
  gdim    = mesh->geometry().dim();
  sdim    = element->space_dimension();
  
  printf("::: MPMModelcpp with gDim = %zu \t sDim = %zu :::\n", gdim, sdim);
}

void MPMModel::add_material(MPMMaterial& M)
{
  materials.push_back(&M);
}

void MPMModel::update_particle_basis_functions(MPMMaterial* M)
{
  // iterate through particle positions :
  for (unsigned int i = 0; i < M->get_num_particles(); i++) 
  {
    // update the grid node indices, basis values, and basis gradient values :
    Point* x_pt    = M->get_x_pt(i);
    cell_id = mesh->bounding_box_tree()->compute_first_entity_collision(*x_pt);
    cell.reset(new Cell( *mesh, cell_id));

    std::vector<double> x_i         = M->get_x(i);
    std::vector<double> phi_i       = M->get_phi(i);
    std::vector<double> grad_phi_i  = M->get_grad_phi(i);
    std::vector<unsigned int> vrt_i = M->get_vrt(i);

    cell->get_vertex_coordinates(vertex_coordinates);
    element->evaluate_basis_all(&phi_i[0],
                                &x_i[0],
                                vertex_coordinates.data(),
                                cell_orientation);
 
    element->evaluate_basis_derivatives_all(deriv_order,
                                            &grad_phi_i[0],
                                            &x_i[0],
                                            vertex_coordinates.data(),
                                            cell_orientation);

    for (unsigned int j = 0; j < Q->dofmap()->cell_dofs(0).size(); j++)
    {
      vrt_i[j] = Q->dofmap()->cell_dofs(cell->index())[j];
    }

    printf("x[%u] = [%f,%f], \n phi[%u] = [%f,%f,%f], \n vrt[%u] = [%u,%u,%u]\n\n", i, x_i[0], x_i[1], i, phi_i[0], phi_i[1], phi_i[2], i, vrt_i[0], vrt_i[1], vrt_i[2]);

    M->set_x(i, x_i);
    M->set_phi(i, phi_i);
    M->set_grad_phi(i, grad_phi_i);
    M->set_vrt(i, vrt_i);

  }
}


void MPMModel::formulate_material_basis_functions()
{
  // iterate through all materials :
  for (unsigned int i = 0; i < materials.size(); i++)
  {
    update_particle_basis_functions(materials[i]);
  }
}



