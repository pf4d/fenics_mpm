#include "MPMModel.h"

using namespace dolfin;

MPMModel::MPMModel(const FunctionSpace& V,
                   const unsigned int num_dofs,
                   const Array<int>& coords,
                   double time_step) :
                   dofs(num_dofs), dt(time_step)
{
  Q       = &V;
  mesh    = V.mesh();
  element = V.element();
  
  gdim    = mesh->geometry().dim();
  sdim    = element->space_dimension();
  
  printf("::: MPMModelcpp with gDim = %zu \t sDim = %zu, dt = %f :::\n",
          gdim, sdim, dt);

  // FIXME: make the vectors of size gdim instead of 3:
  //        this will require some work with the setting of vector
  //        variables in the python code.
  h_grid.resize(dofs);     // cell diameter at node
  m_grid.resize(dofs);     // mass
  V_grid.resize(dofs);     // volume
  U3_grid.resize(3);       // velocity vector
  a3_grid.resize(3);       // acceleration vector
  f_int_grid.resize(3);    // internal force vector
  for (unsigned int i = 0; i < 3; i++)
  {
    // set index of vector with a dimension : 
    if (coords[i] == 1)  coord_arr.push_back(i);
    U3_grid[i].resize(coords[i]*dofs);
    a3_grid[i].resize(coords[i]*dofs);
    f_int_grid[i].resize(coords[i]*dofs);
  }
}

double MPMModel::calculate_determinant(std::vector<double>& u)
{
  unsigned int n = u.size(); // n x n tensor of rank n - 1
  double det;                // the determinant

  if      (n == 1) det = u[0];
  else if (n == 4) det = u[0] * u[3] - u[2] * u[1];
  else if (n == 9) det = + u[0] * u[4] * u[8]
                         + u[1] * u[5] * u[6]
                         + u[2] * u[3] * u[7]
                         - u[2] * u[4] * u[6]
                         - u[1] * u[3] * u[8]
                         - u[0] * u[5] * u[7];
  return det;
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

std::vector<double>  MPMModel::get_a3(unsigned int index) const
{
  return a3_grid.at(index);
}

void  MPMModel::set_a3(unsigned int index, std::vector<double>& value)
{
  a3_grid.at(index) = value;
}

std::vector<double>  MPMModel::get_f_int(unsigned int index) const
{
  return f_int_grid.at(index);
}

void  MPMModel::set_f_int(unsigned int index, std::vector<double>& value)
{
  f_int_grid.at(index) = value;
}

void MPMModel::add_material(MPMMaterial& M)
{
  materials.push_back(&M);
}

void MPMModel::update_points()
{
  // iterate through all materials :
  for (unsigned int i = 0; i < materials.size(); i++)
  {
    // iterate through particle positions :
    for (unsigned int j = 0; j < materials[i]->get_num_particles(); j++) 
    {
      Point* pt_j = materials[i]->get_x_pt(j);          // the particle point
      std::vector<double> x_j = materials[i]->get_x(j); // the particle coord's

      // iterate through dimension and set the coordinate :
      for (unsigned int k = 0; k < gdim; k++)
        pt_j->coordinates()[coord_arr[k]] = x_j[coord_arr[k]];
    }
  }
}

void MPMModel::update_particle_basis_functions(MPMMaterial* M)
{
  update_points();  // update the point coordinates

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

    //printf("x[%u] = [%f,%f], \n phi[%u] = [%f,%f,%f], \n grad_phi[%u] = [%f,%f,%f,%f,%f,%f], \n vrt[%u] = [%u,%u,%u]\n\n",
    //       i, x_i[0], x_i[1],
    //       i, phi_i[0], phi_i[1], phi_i[2],
    //       i, grad_phi_i[0], grad_phi_i[1], grad_phi_i[2], grad_phi_i[3], grad_phi_i[4], grad_phi_i[5],
    //       i, vrt_i[0], vrt_i[1], vrt_i[2]);

    M->set_x(i, x_i);
    M->set_phi(i, phi_i);
    M->set_grad_phi(i, grad_phi_i);
    M->set_vrt(i, vrt_i);

  }
}

void MPMModel::formulate_material_basis_functions()
{
  printf("--- C++ formulate_material_basis_functions() ---\n");
  // iterate through all materials :
  for (unsigned int i = 0; i < materials.size(); i++)
  {
    update_particle_basis_functions(materials[i]);
  }
}

void MPMModel::interpolate_material_mass_to_grid()
{
  printf("--- C++ interpolate_material_mass_to_grid() ---\n");
  // first reset the mass to zero :
  std::fill(m_grid.begin(), m_grid.end(), 0.0);

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
        //printf("\tphi_p[%u] = %f,\t m_p[%u] = %f\n", q, phi_p[q], j, m_p);
        m_grid[idx[q]] += phi_p[q] * m_p;
      }
    }
  }
}

void MPMModel::interpolate_material_velocity_to_grid()
{
  printf("--- C++ interpolate_material_velocity_to_grid() ---\n");
  // first reset the velocity to zero :
  for (unsigned int k = 0; k < gdim; k++)
  {
    std::fill(U3_grid[coord_arr[k]].begin(),
              U3_grid[coord_arr[k]].end(),
              0.0);
  }

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

      // iterate through each coordinate :
      for (unsigned int k = 0; k < gdim; k++)
      {
        // calculate mass-conserving grid velocity at each node :
        for (unsigned int q = 0; q < sdim; q++)
        {
          //printf("\tU3[%u][%u] ::   u_p[%u] = %f,\t phi_p[%u] = %f,\t m_p[%u] = %f,\t m_grid[%u] = %f\n",
          //       k, idx[q], k, u_p[k], q, phi_p[q], j, m_p, idx[q], m_grid[idx[q]]);
          U3_grid[coord_arr[k]][idx[q]] += u_p[k] * 
                                           phi_p[q] * 
                                           m_p / m_grid[idx[q]];
        }
      }
    }
  }
}

void MPMModel::calculate_grid_volume()
{
  printf("--- C++ calculate_grid_volume() ---\n");
  for (std::size_t i = 0; i < dofs; i++)
    V_grid[i] = 4.0/3.0 * M_PI * pow(h_grid[i]/2.0, 3);
}

  
void MPMModel::calculate_material_density()
{
  printf("--- C++ calculate_material_density() ---\n");
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
  printf("--- C++ calculate_material_initial_volume() ---\n");
  // iterate through all materials :
  for (unsigned int i = 0; i < materials.size(); i++)
  {
    // iterate through particles :
    for (unsigned int j = 0; j < materials[i]->get_num_particles(); j++) 
    {
      // calculate inital volume from particle mass and density :
      double V0_j = materials[i]->get_m(j) / materials[i]->get_rho(j);
      materials[i]->set_V0(j, V0_j);
      materials[i]->set_V(j,  V0_j);
    }
  }
}

void MPMModel::calculate_material_velocity_gradient()
{
  printf("--- C++ calculate_material_velocity_gradient() ---\n");
  unsigned int vtx, stx;

  // iterate through all materials :
  for (unsigned int i = 0; i < materials.size(); i++)
  {
    // iterate through particles :
    for (unsigned int j = 0; j < materials[i]->get_num_particles(); j++) 
    {
      std::vector<unsigned int> idx  = materials[i]->get_vrt(j);
      std::vector<double> grad_phi_p = materials[i]->get_grad_phi(j);
      std::vector<double> dudk_p (gdim*gdim, 0.0);

      // iterate through each component of velocity :
      for (unsigned int p = 0; p < gdim; p++)
      {
        // iterate through each coordinate dimension :
        for (unsigned int q = 0; q < gdim; q++)
        {
          vtx = p*gdim + q;
          stx = q;
          // iterate through each node of the particle's cell :
          for (unsigned int s = 0; s < sdim; s++)
          {
            dudk_p[vtx] += grad_phi_p[stx] * 
                           U3_grid[coord_arr[p]][idx[s]];
            stx += gdim;
          }
            /*
            dudx_p = np.sum(grad_phi_i[np.array([0,2,4])] * u.vector()[i])
            dudy_p = np.sum(grad_phi_i[np.array([1,3,5])] * u.vector()[i])
            
            dvdx_p = np.sum(grad_phi_i[np.array([0,2,4])] * v.vector()[i])
            dvdy_p = np.sum(grad_phi_i[np.array([1,3,5])] * v.vector()[i])
            
            grad_U_p_v.append(np.array( [[dudx_p, dudy_p], [dvdx_p, dvdy_p]] ))
            */
        }
      }
      materials[i]->set_grad_u(j, dudk_p);
    }
  } 
}

void MPMModel::initialize_material_tensors()
{
  printf("--- C++ initialize_material_tensors() ---\n");
  std::vector<double> I = materials[0]->get_I();

  // iterate through all materials :
  for (unsigned int i = 0; i < materials.size(); i++)
  {
    // iterate through particles :
    for (unsigned int j = 0; j < materials[i]->get_num_particles(); j++) 
    {
      std::vector<double> dF_j     = materials[i]->get_dF(j);
      std::vector<double> grad_u_j = materials[i]->get_grad_u(j);

      // iterate through each element of the tensor :
      for (unsigned int k = 0; k < gdim*gdim; k++)
        dF_j[k] = I[k] + grad_u_j[k] * dt;
      materials[i]->set_dF(j, dF_j);
      materials[i]->set_F(j,  dF_j);
      materials[i]->calculate_strain_rate();
      materials[i]->calculate_stress();
    }
  }
}

void MPMModel::interpolate_grid_velocity_to_material()
{
  printf("--- C++ interpolate_grid_velocity_to_material() ---\n");
  // iterate through all materials :
  for (unsigned int i = 0; i < materials.size(); i++)
  {
    // iterate through particles :
    for (unsigned int j = 0; j < materials[i]->get_num_particles(); j++) 
    {
      std::vector<unsigned int> idx = materials[i]->get_vrt(j);  // node index
      std::vector<double> phi_p     = materials[i]->get_phi(j);  // basis
      std::vector<double> u_p (gdim, 0.0);                       // velocity

      // iterate through each velocty compoenent :
      for (unsigned int k = 0; k < gdim; k++)
      {
        // interpolate grid velocity back to particles from each node :
        for (unsigned int q = 0; q < sdim; q++)
        {
          u_p[k] += phi_p[q] * U3_grid[coord_arr[k]][idx[q]];
        }
      }
      materials[i]->set_u_star(j, u_p);
    }
  }
}

void MPMModel::interpolate_grid_acceleration_to_material()
{
  printf("--- C++ interpolate_grid_acceleration_to_material() ---\n");

  // iterate through all materials :
  for (unsigned int i = 0; i < materials.size(); i++)
  {
    // iterate through particles :
    for (unsigned int j = 0; j < materials[i]->get_num_particles(); j++) 
    {
      std::vector<unsigned int> idx = materials[i]->get_vrt(j);  // node index
      std::vector<double> phi_p     = materials[i]->get_phi(j);  // basis
      std::vector<double> a_p (gdim, 0.0);                       // acceleration

      // iterate through each component of velocity :
      for (unsigned int k = 0; k < gdim; k++)
      {
        // interpolate grid accleration to particles at each node q :
        for (unsigned int q = 0; q < sdim; q++)
        {
          a_p[k] += phi_p[q] * a3_grid[coord_arr[k]][idx[q]];
        }
      }
      materials[i]->set_a(j, a_p);
    }
  }
}

void MPMModel::update_material_volume()
{
  printf("--- C++ update_material_volume() ---\n");
  // iterate through all materials :
  for (unsigned int i = 0; i < materials.size(); i++)
  {
    // iterate through particles :
    for (unsigned int j = 0; j < materials[i]->get_num_particles(); j++) 
    {
      std::vector<double> dF_p_j = materials[i]->get_dF(j);  // inc. deform.
      double det_dF_j = calculate_determinant(dF_p_j);       // determinant
      double V_p_j    = det_dF_j * materials[i]->get_V(j);   // new volume
      materials[i]->set_V(j, V_p_j);
    }
  }
}

void MPMModel::update_material_deformation_gradient()
{
  printf("--- C++ update_material_deformation_gradient() ---\n");
  std::vector<double> I = materials[0]->get_I();

  // iterate through all materials :
  for (unsigned int i = 0; i < materials.size(); i++)
  {
    // iterate through particles :
    for (unsigned int j = 0; j < materials[i]->get_num_particles(); j++) 
    {
      std::vector<double> dF_j     = materials[i]->get_dF(j);
      std::vector<double> F_j      = materials[i]->get_F(j);
      std::vector<double> grad_u_j = materials[i]->get_grad_u(j);

      // iterate through each component of the tensor :
      for (unsigned int k = 0; k < gdim*gdim; k++)
      {
        dF_j[k]  = I[k] + grad_u_j[k] * dt;
        F_j[k]  *= dF_j[k];
      }
      materials[i]->set_dF(j, dF_j);
      materials[i]->set_F(j,  F_j);
    }
  }
}

void MPMModel::update_material_stress()
{
  printf("--- C++ update_material_stress() ---\n");
  // iterate through all materials :
  for (unsigned int i = 0; i < materials.size(); i++)
  {
    materials[i]->calculate_incremental_strain_rate();

    // iterate through particles :
    for (unsigned int j = 0; j < materials[i]->get_num_particles(); j++) 
    {
      std::vector<double> eps_j    = materials[i]->get_epsilon(j);
      std::vector<double> deps_j   = materials[i]->get_depsilon(j);

      // iterate through each component of the tensor :
      for (unsigned int k = 0; k < gdim*gdim; k++)
        eps_j[k] += deps_j[k] * dt;
      materials[i]->set_epsilon(j, eps_j);
      materials[i]->calculate_stress();
    }
  }
}

void MPMModel::calculate_grid_internal_forces()
{
  printf("--- C++ update_grid_internal_forces() ---\n");
  unsigned int vtx, stx;

  // first reset the forces to zero :
  for (unsigned int k = 0; k < gdim; k++)
  {
    std::fill(f_int_grid[coord_arr[k]].begin(),
              f_int_grid[coord_arr[k]].end(),
              0.0);
  }

  // iterate through all materials :
  for (unsigned int i = 0; i < materials.size(); i++)
  {
    // iterate through particles :
    for (unsigned int j = 0; j < materials[i]->get_num_particles(); j++) 
    {
      std::vector<unsigned int> idx  = materials[i]->get_vrt(j);
      std::vector<double> grad_phi_p = materials[i]->get_grad_phi(j);
      std::vector<double> sigma_p    = materials[i]->get_sigma(j);
      double V_p                     = materials[i]->get_V(j);
      stx                            = 0;

      // iterate through the nodes :
      for (unsigned int s = 0; s < sdim; s++)
      {
        // iterate through each component of force :
        for (unsigned int p = 0; p < gdim; p++)
        {
          // iterate through each geometric coordinate :
          for (unsigned int q = 0; q < gdim; q++)
          {
            vtx = p*gdim + q;
            f_int_grid[coord_arr[p]][idx[s]] -= sigma_p[vtx] * 
                                                grad_phi_p[stx] * V_p;
          }
        }
        stx += gdim;
      }
    }
  }
}
  /*
        f_int = []
        f_int.append( np.array([ np.dot(sig_p[0:2], grad_phi_p[0:2]),
                                 np.dot(sig_p[2:4], grad_phi_p[0:2])  ]) )
        f_int.append( np.array([ np.dot(sig_p[0:2], grad_phi_p[2:4]),
                                 np.dot(sig_p[2:4], grad_phi_p[2:4])  ]) )
        f_int.append( np.array([ np.dot(sig_p[0:2], grad_phi_p[4:6]),
                                 np.dot(sig_p[2:4], grad_phi_p[4:6])  ]) )
        f_int = - np.array(f_int, dtype=float) * V_p
        
        f_int_x.vector()[p] += f_int[:,0].astype(float)
        f_int_y.vector()[p] += f_int[:,1].astype(float)
  */

void MPMModel::update_grid_velocity()
{
  printf("--- C++ update_grid_velocity() ---\n");

  // iterate through each component of velocity :
  for (unsigned int k = 0; k < gdim; k++)
  {
    // iterate through each node :
    for (unsigned int i = 0; i < dofs; i++)
    {
      U3_grid[coord_arr[k]][i] += a3_grid[coord_arr[k]][i] * dt;
    }
  }
}

void MPMModel::calculate_grid_acceleration(double m_min)
{
  printf("--- C++ calculate_grid_acceleration() ---\n");
  /*
    f_int_x, f_int_y, f_int_z = self.grid_model.f_int.split(True)
    f_int_x_a = f_int_x.vector().array()
    f_int_y_a = f_int_y.vector().array()
    m_a       = self.grid_model.m.vector().array()

    eps_m               = 1e-2
    f_int_x_a[m_a == 0] = 0.0
    f_int_y_a[m_a == 0] = 0.0
    m_a[m_a < eps_m]    = eps_m
    
    a_x    = Function(self.grid_model.Q)
    a_y    = Function(self.grid_model.Q)

    a_x.vector().set_local(f_int_x_a / m_a)
    a_y.vector().set_local(f_int_y_a / m_a)
    self.grid_model.update_acceleration([a_x, a_y])
  */
  // iterate through each component of acceleration :
  for (unsigned int k = 0; k < gdim; k++)
  {
    // iterate through each node :
    for (unsigned int i = 0; i < dofs; i++)
    {
      // if there is a mass on the grid node :
      if (m_grid[i] > 0)
      {
        // set the minimum mass :
        if (m_grid[i] < m_min)  m_grid[i] = m_min;
        // update acceleration :
        a3_grid[coord_arr[k]][i] = f_int_grid[coord_arr[k]][i] / m_grid[i];
        //printf("f_int_grid[%u][%u] = %f\n", coord_arr[k], i,
        //                                f_int_grid[coord_arr[k]][i]);
      }
      // otherwise set the grid values to zero :
      else
      {
        f_int_grid[coord_arr[k]][i] = 0.0;
        a3_grid[coord_arr[k]][i]    = 0.0;
      }
    }
  }
}


void MPMModel::advect_material_particles()
{
  printf("--- C++ advect_material_particles() ---\n");
  /*
      # calculate the new material velocity :
      M.u += M.a * self.dt

      # advect the material :
      M.x += M.u_star * self.dt
  */
  // interpolate the grid acceleration to the particles : 
  interpolate_grid_acceleration_to_material();

  // interpolate the velocities back to the particles :
  interpolate_grid_velocity_to_material();
  
  // iterate through all materials :
  for (unsigned int i = 0; i < materials.size(); i++)
  {
    // iterate through particles :
    for (unsigned int j = 0; j < materials[i]->get_num_particles(); j++) 
    {
      std::vector<double> a_j      = materials[i]->get_a(j);
      std::vector<double> u_j      = materials[i]->get_u(j);
      std::vector<double> u_star_j = materials[i]->get_u_star(j);
      std::vector<double> x_j      = materials[i]->get_x(j);

      for (unsigned int k = 0; k < gdim; k++)
      {
        u_j[k] += a_j[k] * dt;
        x_j[k] += u_star_j[k] * dt;
      }
      materials[i]->set_u(j, u_j);
      materials[i]->set_x(j, x_j);
    }
  }
}



