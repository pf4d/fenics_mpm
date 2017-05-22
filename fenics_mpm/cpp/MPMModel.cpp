#include "MPMModel.h"

using namespace dolfin;

MPMModel::MPMModel(const FunctionSpace& V,
                   const unsigned int num_dofs,
                   const Array<int>& coords) :
                   dofs(num_dofs)
{
  Q       = &V;
  mesh    = V.mesh();
  element = V.element();
  
  gdim    = mesh->geometry().dim();
  sdim    = element->space_dimension();
  
  printf("::: MPMModelcpp with gDim = %zu \t sDim = %zu :::\n", gdim, sdim);

  h_grid.resize(dofs);
  m_grid.resize(dofs);
  V_grid.resize(dofs);
  U3_grid.resize(3);
  a3_grid.resize(3);
  f_int_grid.resize(3);
  for (unsigned int i = 0; i < 3; i++)
  {
    coord_arr[i] = coords[i];
    U3_grid[i].resize(coords[i]*dofs);
    a3_grid[i].resize(coords[i]*dofs);
    f_int_grid[i].resize(coords[i]*dofs);
  }
}

void MPMModel::set_h(const Array<double>& h)
{
  for (std::size_t i = 0; i < h.size(); i++)
    h_grid[i] = h[i];
}

std::vector<double>  MPMModel::get_U3(unsigned int index) const
{
  return U3_grid.at(index);
}

void  MPMModel::set_U3(unsigned int index, std::vector<double>& value)
{
  U3_grid.at(index) = value;
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

    printf("x[%u] = [%f,%f], \n phi[%u] = [%f,%f,%f], \n grad_phi[%u] = [%f,%f,%f,%f,%f,%f], \n vrt[%u] = [%u,%u,%u]\n\n",
           i, x_i[0], x_i[1],
           i, phi_i[0], phi_i[1], phi_i[2],
           i, grad_phi_i[0], grad_phi_i[1], grad_phi_i[2], grad_phi_i[3], grad_phi_i[4], grad_phi_i[5],
           i, vrt_i[0], vrt_i[1], vrt_i[2]);

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

void MPMModel::interpolate_material_mass_to_grid()
{
  // iterate through all materials :
  for (unsigned int i = 0; i < materials.size(); i++)
  {
    // iterate through particles :
    for (unsigned int j = 0; j < materials[i]->get_num_particles(); j++) 
    {
      std::vector<unsigned int> idx = materials[i]->get_vrt(j);  // node index
      std::vector<double> phi_p     = materials[i]->get_phi(j);  // basis
      double m_p                    = materials[i]->get_m(j);    // mass

      // interpolate the particle mass to each node of its cell :
      for (unsigned int q = 0; q < sdim; q++)
      {
        printf("\tphi_p[%u] = %f,\t m_p[%u] = %f\n", q, phi_p[q], j, m_p);
        m_grid[idx[q]] += phi_p[q] * m_p;
      }
    }
  }
}

void MPMModel::interpolate_material_velocity_to_grid() 
{
  // iterate through all materials :
  for (unsigned int i = 0; i < materials.size(); i++)
  {
    // iterate through particles :
    for (unsigned int j = 0; j < materials[i]->get_num_particles(); j++) 
    {
      std::vector<unsigned int> idx = materials[i]->get_vrt(j);  // node index
      std::vector<double> phi_p     = materials[i]->get_phi(j);  // basis
      std::vector<double> u_p       = materials[i]->get_u(j);    // velocity
      double m_p                    = materials[i]->get_m(j);    // mass
      unsigned int ctr = 0;                                      // vel. count

      // iterate through each dimension :
      for (unsigned int k = 0; k < 3; k++)
      {
        // if this dimension has a velocity :
        if (coord_arr[k] == 1)
        {
          // calculate mass-conserving grid velocity at the node :
          for (unsigned int q = 0; q < sdim; q++)
          {
            printf("\tU3[%u][%u] ::   u_p[%u] = %f,\t phi_p[%u] = %f,\t m_p[%u] = %f,\t m_grid[%u] = %f\n",
                   k, idx[q], ctr, u_p[ctr], q, phi_p[q], j, m_p, idx[q], m_grid[idx[q]]);
            U3_grid[k][idx[q]] += u_p[ctr] * phi_p[q] * m_p / m_grid[idx[q]];
          }
          ctr++;  // increment counter because num(u_p) may != num(u_grid)
        }
      }
    }
  }
}

void MPMModel::calculate_grid_volume()
{
  for (std::size_t i = 0; i < dofs; i++)
    V_grid[i] = 4.0/3.0 * M_PI * pow(h_grid[i]/2.0, 3);
}

  
void MPMModel::calculate_material_density()
{
  // iterate through all materials :
  for (unsigned int i = 0; i < materials.size(); i++)
  {
    // iterate through particles :
    for (unsigned int j = 0; j < materials[i]->get_num_particles(); j++) 
    {
      std::vector<unsigned int> idx = materials[i]->get_vrt(j);  // node index
      std::vector<double> phi_p     = materials[i]->get_phi(j);  // basis
      double m_p                    = materials[i]->get_m(j);    // mass
      double rho_j                  = 0;                         // sum identity

      // interpolate the particle mass to each node of its cell :
      for (unsigned int q = 0; q < sdim; q++)
      {
        rho_j += m_grid[idx[q]] * phi_p[q] / V_grid[idx[q]];
      }
      materials[i]->set_rho(j, rho_j);
    }
  }
}
  
void MPMModel::calculate_material_initial_volume()
{
  // iterate through all materials :
  for (unsigned int i = 0; i < materials.size(); i++)
  {
    // iterate through particles :
    for (unsigned int j = 0; j < materials[i]->get_num_particles(); j++) 
    {
      // calculate inital volume from particle mass and density :
      double V0_j = materials[i]->get_m(j)/materials[i]->get_rho(j);
      materials[i]->set_V0(j, V0_j);
      materials[i]->set_V(j,  V0_j);
    }
  }
}
  
void MPMModel::calculate_material_velocity_gradient()
{
  // iterate through all materials :
  for (unsigned int i = 0; i < materials.size(); i++)
  {
    // iterate through particles :
    for (unsigned int j = 0; j < materials[i]->get_num_particles(); j++) 
    {
      std::vector<unsigned int> idx  = materials[i]->get_vrt(j);  // node index
      std::vector<double> grad_phi_p = materials[i]->get_phi(j);  // basis grad.
      unsigned int ctr = 0;                                       // vel. count

      // iterate through each dimension :
      for (unsigned int k = 0; k < 3; k++)
      {
        // if this dimension has a velocity :
        if (coord_arr[k] == 1)
        {
          /*
          u_i    = u.vector()[i]
          v_i    = v.vector()[i]
          dudx_p = np.sum(grad_phi_i[:,0] * u.vector()[i])
          dudy_p = np.sum(grad_phi_i[:,1] * u.vector()[i])
          dvdx_p = np.sum(grad_phi_i[:,0] * v.vector()[i])
          dvdy_p = np.sum(grad_phi_i[:,1] * v.vector()[i])
          grad_U_p_v.append(np.array( [[dudx_p, dudy_p], [dvdx_p, dvdy_p]] ))
          */
          ctr++;  // increment counter because num(u_p) may != num(u_grid)
        }
      }
    }
  } 
}



