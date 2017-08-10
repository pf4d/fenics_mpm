#include "MPMModel.h"

using namespace dolfin;

MPMModel::MPMModel(const FunctionSpace& V,
                   const unsigned int num_dofs,
                   const Array<int>& coords,
                   double time_step,
                   bool verbosity) :
                   dofs(num_dofs), dt(time_step), verbose(verbosity)
{
  Q       = &V;
  mesh    = V.mesh();
  element = V.element();
  
  gdim    = mesh->geometry().dim();
  sdim    = element->space_dimension();
  
  printf("::: initializing MPMModelcpp with gDim = %zu,  sDim = %zu,"
         " dt = %f :::\n", gdim, sdim, dt);

  // basis function variables :
  phi_temp.resize(sdim);
  grad_phi_temp.resize(gdim*sdim);

  // scalars :
  h_grid.resize(dofs);             // cell diameter at node
  m_grid.resize(dofs);             // mass
  V_grid.resize(dofs);             // volume

  // vectors :
  u_x_grid.resize(dofs);           // velocity vector
  u_x_grid_new.resize(dofs);       // next velocity vector
  a_x_grid.resize(dofs);           // acceleration vector
  a_x_grid_new.resize(dofs);       // next acceleration vector
  f_x_int_grid.resize(dofs);       // internal force vector

  // in two or three dimensions, need y components :
  if (gdim == 2 or gdim == 3)
  {
    u_y_grid.resize(dofs);           // velocity vector
    u_y_grid_new.resize(dofs);       // next velocity vector
    a_y_grid.resize(dofs);           // acceleration vector
    a_y_grid_new.resize(dofs);       // next acceleration vector
    f_y_int_grid.resize(dofs);       // internal force vector
  }
  
  // in three dimensions, need z components :
  if (gdim == 2 or gdim == 3)
  {
    u_z_grid.resize(dofs);           // velocity vector
    u_z_grid_new.resize(dofs);       // next velocity vector
    a_z_grid.resize(dofs);           // acceleration vector
    a_z_grid_new.resize(dofs);       // next acceleration vector
    f_z_int_grid.resize(dofs);       // internal force vector
  }
}

void MPMModel::set_h(const Array<double>& h_a)
{
  for (std::size_t i = 0; i < h_a.size(); i++)
    h_grid[i] = h_a[i];
}

void MPMModel::set_V(const Array<double>& V_a)
{
  for (std::size_t i = 0; i < V_a.size(); i++)
    V_grid[i] = V_a[i];
}

void MPMModel::add_material(MPMMaterial& M)
{
  materials.push_back(&M);
}

void MPMModel::set_boundary_conditions(const Array<int>& vertices,
                                       const Array<double>& values)
{
  for (unsigned int i = 0; i < vertices.size(); i++)
    bc_vrt.push_back(vertices[i]);
  for (unsigned int i = 0; i < values.size(); i++)
    bc_val.push_back(values[i]);
}

void MPMModel::update_points()
{
  // iterate through all materials :
  for (unsigned int i = 0; i < materials.size(); i++)
  {
    std::vector<Point*> pt_i = materials[i]->get_x_pt(); // the particle Points

    // iterate through particle positions :
    for (unsigned int j = 0; j < materials[i]->get_num_particles(); j++)
    {
      // always have an x component :
      pt_i[j]->coordinates()[0] = materials[i]->get_x()[j];
      
      // two and three dimensions also has a y component :
      if (gdim == 2 or gdim == 3)
        pt_i[j]->coordinates()[1] = materials[i]->get_y()[j];

      // three dimensions also has a z component :
      if (gdim == 3)
        pt_i[j]->coordinates()[2] = materials[i]->get_z()[j];
    }
  }
}

void MPMModel::update_particle_basis_functions(MPMMaterial* M)
{
  update_points();  // update the point coordinates
    
  std::vector<Point*> x_pt      = M->get_x_pt();
  std::vector<double> phi       = M->get_phi();
  std::vector<double> grad_phi  = M->get_grad_phi();
  std::vector<unsigned int> vrt = M->get_vrt();

  // iterate through particle positions :
  for (unsigned int i = 0; i < M->get_num_particles(); i++) 
  {
    // update the grid node indices, basis values, and basis gradient values :
    c_id = mesh->bounding_box_tree()->compute_first_entity_collision(*x_pt[i]);
    cell.reset(new Cell(*mesh, c_id));


    cell->get_vertex_coordinates(vertex_coordinates);
    element->evaluate_basis_all(&phi_temp[0],
                                &x_pt[i]->coordinates(),
                                vertex_coordinates.data(),
                                cell_orientation);
 
    element->evaluate_basis_derivatives_all(deriv_order,
                                            &grad_phi_temp[0],
                                            &x_pt[i]->coordinates(),
                                            vertex_coordinates.data(),
                                            cell_orientation);

    // all cells have at least two vertices :
    vrt_1[i] = Q->dofmap()->cell_dofs(cell->index())[1];
    vrt_2[i] = Q->dofmap()->cell_dofs(cell->index())[2];

    // two basis functions :
    phi_1[i] = phi_temp[0];
    phi_2[i] = phi_temp[2];
  
    // two or three dimensions have a third basis function : 
    if (gdim == 2 or gdim == 3)
    {
      vrt_3[i] = Q->dofmap()->cell_dofs(cell->index())[3];
      phi_3[i] = phi_temp[3];
    }
    
    // three dimensions another :
    if (gdim == 3)
    {
      vrt_4[i]       = Q->dofmap()->cell_dofs(cell->index())[4];
      phi_4[i]       = phi_temp[4];
    }

    // one dimension has two basis function derivatives :
    if (gdim == 1)
    {
      grad_phi_1x[i] = grad_phi_temp[0];
      grad_phi_2x[i] = grad_phi_temp[1];
    }

    // two dimensions have six : 
    else if (gdim == 2)
    {
      grad_phi_1x[i] = grad_phi_temp[0];
      grad_phi_1y[i] = grad_phi_temp[1];
      grad_phi_2x[i] = grad_phi_temp[2];
      grad_phi_2y[i] = grad_phi_temp[3];
      grad_phi_3x[i] = grad_phi_temp[4];
      grad_phi_3y[i] = grad_phi_temp[5];
    }
    
    // three dimensions have twelve :
    else if (gdim == 3)
    {
      grad_phi_1x[i] = grad_phi_temp[0];
      grad_phi_1y[i] = grad_phi_temp[1];
      grad_phi_1z[i] = grad_phi_temp[2];
      grad_phi_2x[i] = grad_phi_temp[3];
      grad_phi_2y[i] = grad_phi_temp[4];
      grad_phi_2z[i] = grad_phi_temp[5];
      grad_phi_3x[i] = grad_phi_temp[6];
      grad_phi_3y[i] = grad_phi_temp[7];
      grad_phi_3z[i] = grad_phi_temp[8];
      grad_phi_4x[i] = grad_phi_temp[9];
      grad_phi_4y[i] = grad_phi_temp[10];
      grad_phi_4z[i] = grad_phi_temp[11];
    }
  }
}

void MPMModel::formulate_material_basis_functions()
{
  if (verbose == true)
    printf("--- C++ formulate_material_basis_functions() ---\n");
  // iterate through all materials :
  for (unsigned int i = 0; i < materials.size(); i++)
  {
    update_particle_basis_functions(materials[i]);
  }
}

void MPMModel::interpolate_material_mass_to_grid()
{
  if (verbose == true)
    printf("--- C++ interpolate_material_mass_to_grid() ---\n");

  // first reset the mass to zero :
  std::fill(m_grid.begin(), m_grid.end(), 0.0);

  // iterate through all materials :
  for (unsigned int i = 0; i < materials.size(); i++)
  {
    // node index :
    std::vector<unsigned int> vrt_1 = materials[i]->get_vrt_1();
    std::vector<unsigned int> vrt_2 = materials[i]->get_vrt_2();
    std::vector<unsigned int> vrt_3 = materials[i]->get_vrt_3();
    std::vector<unsigned int> vrt_4 = materials[i]->get_vrt_4();
   
    // basis functions : 
    std::vector<double> phi_1       = materials[i]->get_phi_1();
    std::vector<double> phi_2       = materials[i]->get_phi_2();
    std::vector<double> phi_3       = materials[i]->get_phi_3();
    std::vector<double> phi_4       = materials[i]->get_phi_4();

    // mass :
    std::vector<double> m           = materials[i]->get_m();

    // iterate through particles and interpolate the 
    // particle mass to each node of its cell :
    for (unsigned int j = 0; j < materials[i]->get_num_particles(); j++) 
    {
      // in one dimension, two vertices :
      m_grid[vrt_1[j]] += phi_1[j] * m[j];
      m_grid[vrt_2[j]] += phi_2[j] * m[j];

      // in two or three dimensions, one more :
      if (gdim == 2 or gdim == 3)
        m_grid[vrt_3[j]] += phi_3[j] * m[j];
      
      // in three dimensions, one more :
      if (gdim == 3)
        m_grid[vrt_4[j]] += phi_4[j] * m[j];
    }
  }
}

void MPMModel::interpolate_material_velocity_to_grid()
{
  if (verbose == true)
    printf("--- C++ interpolate_material_velocity_to_grid() ---\n");

  // first reset the velocity to zero :
  std::fill(u_x_grid.begin(), u_x_grid.end(), 0.0);
  
  // y-component :    
  if (gdim == 2 or gdim == 3)
    std::fill(u_y_grid.begin(), u_y_grid.end(), 0.0);
  
  // z-component :    
  if (gdim == 3)
    std::fill(u_z_grid.begin(), u_z_grid.end(), 0.0);

  // iterate through all materials :
  for (unsigned int i = 0; i < materials.size(); i++)
  {
    // node index :
    std::vector<unsigned int> vrt_1 = materials[i]->get_vrt_1();
    std::vector<unsigned int> vrt_2 = materials[i]->get_vrt_2();
    std::vector<unsigned int> vrt_3 = materials[i]->get_vrt_3();
    std::vector<unsigned int> vrt_4 = materials[i]->get_vrt_4();
   
    // basis functions : 
    std::vector<double> phi_1       = materials[i]->get_phi_1();
    std::vector<double> phi_2       = materials[i]->get_phi_2();
    std::vector<double> phi_3       = materials[i]->get_phi_3();
    std::vector<double> phi_4       = materials[i]->get_phi_4();

    // velocity :
    std::vector<double> u_x         = materials[i]->get_u_x();
    std::vector<double> u_y         = materials[i]->get_u_y();
    std::vector<double> u_z         = materials[i]->get_u_z();

    // mass :
    std::vector<double> m           = materials[i]->get_m();

    // iterate through particles :
    for (unsigned int j = 0; j < materials[i]->get_num_particles(); j++) 
    {
      // in one dimension, two vertices, one component of velocity :
      u_x_grid[vrt_1[j]] += u_x[j] * phi_1[j] * m[j] / m_grid[vrt_1[j]];
      u_x_grid[vrt_2[j]] += u_x[j] * phi_2[j] * m[j] / m_grid[vrt_2[j]];

      // in two or three dimensions, one more component and vertex :
      if (gdim == 2 or gdim == 3)
      {
        // extra node for x-velocity component :
        u_x_grid[vrt_3[j]] += u_x[j] * phi_3[j] * m[j] / m_grid[vrt_3[j]];

        // new y-component of velocity :
        u_y_grid[vrt_1[j]] += u_y[j] * phi_1[j] * m[j] / m_grid[vrt_1[j]];
        u_y_grid[vrt_2[j]] += u_y[j] * phi_2[j] * m[j] / m_grid[vrt_2[j]];
        u_y_grid[vrt_3[j]] += u_y[j] * phi_3[j] * m[j] / m_grid[vrt_3[j]];
      }
      
      // in three dimensions, one more component and vertex :
      if (gdim == 3)
      {
        // extra node for x- and y-velocity components :
        u_x_grid[vrt_4[j]] += u_x[j] * phi_4[j] * m[j] / m_grid[vrt_4[j]];
        u_y_grid[vrt_4[j]] += u_y[j] * phi_4[j] * m[j] / m_grid[vrt_4[j]];

        // new z-component of velocity :
        u_z_grid[vrt_1[j]] += u_z[j] * phi_1[j] * m[j] / m_grid[vrt_1[j]];
        u_z_grid[vrt_2[j]] += u_z[j] * phi_2[j] * m[j] / m_grid[vrt_2[j]];
        u_z_grid[vrt_3[j]] += u_z[j] * phi_3[j] * m[j] / m_grid[vrt_3[j]];
        u_z_grid[vrt_4[j]] += u_z[j] * phi_4[j] * m[j] / m_grid[vrt_4[j]];
      }
    }
  }
}

void MPMModel::calculate_grid_volume()
{
  if (verbose == true)
    printf("--- C++ calculate_grid_volume() ---\n");

  for (std::size_t i = 0; i < dofs; i++)
    V_grid[i] = 4.0/3.0 * M_PI * pow(h_grid[i]/2.0, 3);
}

  
void MPMModel::calculate_material_initial_density()
{
  printf("--- C++ calculate_material_initial_density() ---\n");

  // iterate through all materials :
  for (unsigned int i = 0; i < materials.size(); i++)
  {
    if (materials[i]->get_mass_init() == true)
    {
      printf("    - material `%s` has not been initialized with density -\n",
             materials[i]->get_name());

      // node index :
      std::vector<unsigned int> vrt_1 = materials[i]->get_vrt_1();
      std::vector<unsigned int> vrt_2 = materials[i]->get_vrt_2();
      std::vector<unsigned int> vrt_3 = materials[i]->get_vrt_3();
      std::vector<unsigned int> vrt_4 = materials[i]->get_vrt_4();
   
      // basis functions : 
      std::vector<double> phi_1       = materials[i]->get_phi_1();
      std::vector<double> phi_2       = materials[i]->get_phi_2();
      std::vector<double> phi_3       = materials[i]->get_phi_3();
      std::vector<double> phi_4       = materials[i]->get_phi_4();

      // mass and density :
      std::vector<double> m           = materials[i]->get_m();
      std::vector<double> rho         = materials[i]->get_rho();
      std::vector<double> rho0        = materials[i]->get_rho0();

      // iterate through particles :
      for (unsigned int j = 0; j < materials[i]->get_num_particles(); j++) 
      {
        // in one dimension, two nodes :
        rho0[j] += m_grid[vrt_1[j]] * phi_1[j] / V_grid[vrt_1[j]];
        rho0[j] += m_grid[vrt_2[j]] * phi_2[j] / V_grid[vrt_2[j]];

        // in two or three dimensions add the y-component node :
        if (gdim == 2 or gdim == 3)
          rho0[j] += m_grid[vrt_3[j]] * phi_3[j] / V_grid[vrt_3[j]];
        
        // in three dimensions add the z-component node :
        if (gdim == 3)
          rho0[j] += m_grid[vrt_4[j]] * phi_4[j] / V_grid[vrt_4[j]];

        rho[j] = rho0[j];  // set the current velocity too
      }
    }
    else
    {
      printf("    - material `%s` has been initialized with density,"
             " skipping calculation -\n", materials[i]->get_name());
    }
  }
}
  
void MPMModel::calculate_material_initial_volume()
{
  printf("--- C++ calculate_material_initial_volume() ---\n");

  // iterate through all materials :
  for (unsigned int i = 0; i < materials.size(); i++)
  {
    if (materials[i]->get_mass_init() == true)
    {
      printf("    - material `%s` has not been initialized with volume -\n",
             materials[i]->get_name());
      materials[i]->calculate_initial_volume();
    }
    else
    {
      printf("    - material `%s` has been initialized with volume,"
             " skipping calculation -\n", materials[i]->get_name());
    }
  }
}

void MPMModel::initialize_material_tensors()
{
  if (verbose == true)
    printf("--- C++ initialize_material_tensors() ---\n");

  std::vector<double> I = materials[0]->get_I();

  // iterate through all materials :
  for (unsigned int i = 0; i < materials.size(); i++)
  {
    materials[i]->initialize_tensors(dt);
  }
}

void MPMModel::calculate_grid_internal_forces()
{
  if (verbose == true)
    printf("--- C++ update_grid_internal_forces() ---\n");

  // first reset the forces to zero :
  std::fill(f_int_x_grid.begin(), f_int_x_grid.end(), 0.0);
  
  // y-component :    
  if (gdim == 2 or gdim == 3)
    std::fill(f_int_y_grid.begin(), f_int_y_grid.end(), 0.0);
  
  // z-component :    
  if (gdim == 3)
    std::fill(f_int_z_grid.begin(), f_int_z_grid.end(), 0.0);
      

  // iterate through all materials :
  for (unsigned int i = 0; i < materials.size(); i++)
  {
    // node index :
    std::vector<unsigned int> vrt_1 = materials[i]->get_vrt_1();
    std::vector<unsigned int> vrt_2 = materials[i]->get_vrt_2();
    std::vector<unsigned int> vrt_3 = materials[i]->get_vrt_3();
    std::vector<unsigned int> vrt_4 = materials[i]->get_vrt_4();
   
    // basis function derivatives :
    std::vector<double> grad_phi_1x = materials[i]->get_grad_phi_1x();
    std::vector<double> grad_phi_1y = materials[i]->get_grad_phi_1y();
    std::vector<double> grad_phi_1z = materials[i]->get_grad_phi_1z();
    std::vector<double> grad_phi_2x = materials[i]->get_grad_phi_2x();
    std::vector<double> grad_phi_2y = materials[i]->get_grad_phi_2y();
    std::vector<double> grad_phi_2z = materials[i]->get_grad_phi_2z();
    std::vector<double> grad_phi_3x = materials[i]->get_grad_phi_3x();
    std::vector<double> grad_phi_3y = materials[i]->get_grad_phi_3y();
    std::vector<double> grad_phi_3z = materials[i]->get_grad_phi_3z();
    std::vector<double> grad_phi_4x = materials[i]->get_grad_phi_4x();
    std::vector<double> grad_phi_4y = materials[i]->get_grad_phi_4y();
    std::vector<double> grad_phi_4z = materials[i]->get_grad_phi_4z();

    // stress tensor :
    std::vector<double> sigma_xx    = get_sigma_xx();
    std::vector<double> sigma_xy    = get_sigma_xy();
    std::vector<double> sigma_xz    = get_sigma_xz();
    std::vector<double> sigma_yy    = get_sigma_yy();
    std::vector<double> sigma_yz    = get_sigma_yz();
    std::vector<double> sigma_zz    = get_sigma_zz();

    // volume :
    std::vector<double> V           = materials[i]->get_V();
    
    // iterate through particles :
    for (unsigned int j = 0; j < materials[i]->get_num_particles(); j++)
    {
      // in one dimension, two vertices, one component of force :
      f_int_x_grid[vrt_1[j]] -= sigma_xx[j] * grad_phi_1x[j] * V[j];
      f_int_x_grid[vrt_2[j]] -= sigma_xx[j] * grad_phi_2x[j] * V[j];

      // in two or three dimensions, one more component and vertex :
      if (gdim == 2 or gdim == 3)
      {
        // extra node for x-force component :
        f_int_x_grid[vrt_3[j]] -= sigma_xx[j] * grad_phi_3x[j] * V[j];

        // new y-component of velocity :
      }
      
      // in three dimensions, one more component and vertex :
      if (gdim == 3)
      {
        // extra node for x- and y-force components :

        // new z-component of force :
      }
    }
    

    // iterate through particles :
    for (unsigned int j = 0; j < materials[i]->get_num_particles(); j++) 
    {
      stx = 0;

      // iterate through the nodes :
      for (unsigned int s = 0; s < sdim; s++)
      {
        // iterate through each component of force :
        for (unsigned int p = 0; p < gdim; p++)
        {
          vtx = p*gdim;
          // iterate through each geometric coordinate :
          for (unsigned int q = 0; q < gdim; q++)
          {
            f_int_grid[coord_arr[p]][idx[s]] -= sigma_p[vtx + q] * 
                                                grad_phi_p[stx + q] * V_p;
          }
        }
        stx += gdim;
      }
    }
  }
}

void MPMModel::calculate_grid_acceleration(double m_min)
{
  if (verbose == true)
    printf("--- C++ calculate_grid_acceleration() ---\n");
  
  // iterate through each component of acceleration :
  for (unsigned int k = 0; k < gdim; k++)
  {
    // iterate through each node :
    for (unsigned int i = 0; i < dofs; i++)
    {
      // FIXME: this needs to be done properly for the first iteration too.
      // update old acceleration :
      a3_grid[coord_arr[k]][i] = a3_grid_new[coord_arr[k]][i];
    }
  }

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
        a3_grid_new[coord_arr[k]][i] = f_int_grid[coord_arr[k]][i] / m_grid[i];
      }
      // otherwise set the grid values to zero :
      else
      {
        f_int_grid[coord_arr[k]][i]  = 0.0;
        a3_grid_new[coord_arr[k]][i] = 0.0;
      }
    }
  }
}

void MPMModel::update_grid_velocity()
{
  if (verbose == true)
    printf("--- C++ update_grid_velocity() ---\n");

  // iterate through each component of velocity :
  for (unsigned int k = 0; k < gdim; k++)
  {
    // iterate through each node :
    for (unsigned int i = 0; i < dofs; i++)
    {
      U3_grid_new[coord_arr[k]][i] = U3_grid[coord_arr[k]][i] +
                                     0.5 * (a3_grid[coord_arr[k]][i] + 
                                            a3_grid_new[coord_arr[k]][i]) * dt;
    }
  }
  /*
  // TODO: set boolean flag for using Dirichlet or not : 
  // apply boundary conditions if present :
  for (unsigned int k = 0; k < gdim; k++)
  {
    for (unsigned int i = 0; i < bc_val.size(); i++)
    {
      for (unsigned int j = 0; j < bc_vrt.size(); j++)
      {
        U3_grid[coord_arr[k]][bc_vrt[j]] = 0.0;
      }
    }
  }
  */
}

void MPMModel::calculate_material_velocity_gradient()
{
  if (verbose == true)
    printf("--- C++ calculate_material_velocity_gradient() ---\n");

  unsigned int vtx, stx;

  // iterate through all materials :
  for (unsigned int i = 0; i < materials.size(); i++)
  {
    // iterate through particles :
    for (unsigned int j = 0; j < materials[i]->get_num_particles(); j++) 
    {
      std::vector<unsigned int> idx      = materials[i]->get_vrt(j);
      std::vector<double> grad_phi_p     = materials[i]->get_grad_phi(j);
      std::vector<double> grad_u_star_p  = materials[i]->get_grad_u_star(j);
      std::vector<double> dudk_p (gdim*gdim, 0.0);
      
      // set the previous velocity gradient :
      materials[i]->set_grad_u(j, grad_u_star_p);

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
                           U3_grid_new[coord_arr[p]][idx[s]];
            stx += gdim;
          }
        }
      }
      materials[i]->set_grad_u_star(j, dudk_p);
    }
  } 
}

void MPMModel::update_material_deformation_gradient()
{
  if (verbose == true)
    printf("--- C++ update_material_deformation_gradient() ---\n");

  std::vector<double> I = materials[0]->get_I();

  // iterate through all materials :
  for (unsigned int i = 0; i < materials.size(); i++)
  {
    materials[i]->update_deformation_gradient(dt);
  }
}

void MPMModel::update_material_density()
{
  if (verbose == true)
    printf("--- C++ update_material_density() ---\n");

  // iterate through all materials :
  for (unsigned int i = 0; i < materials.size(); i++)
  {
    materials[i]->update_density();
  }
}

void MPMModel::update_material_volume()
{
  if (verbose == true)
    printf("--- C++ update_material_volume() ---\n");

  // iterate through all materials :
  for (unsigned int i = 0; i < materials.size(); i++)
  {
    materials[i]->update_volume();
  }
}

void MPMModel::update_material_stress()
{
  if (verbose == true)
    printf("--- C++ update_material_stress() ---\n");

  // iterate through all materials :
  for (unsigned int i = 0; i < materials.size(); i++)
  {
    materials[i]->update_stress(dt);
  }
}

void MPMModel::interpolate_grid_velocity_to_material()
{
  if (verbose == true)
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
          u_p[k] += phi_p[q] * U3_grid_new[coord_arr[k]][idx[q]];
        }
      }
      materials[i]->set_u_star(j, u_p);
    }
  }
}

void MPMModel::interpolate_grid_acceleration_to_material()
{
  if (verbose == true)
    printf("--- C++ interpolate_grid_acceleration_to_material() ---\n");

  // iterate through all materials :
  for (unsigned int i = 0; i < materials.size(); i++)
  {
    // iterate through particles :
    for (unsigned int j = 0; j < materials[i]->get_num_particles(); j++) 
    {
      std::vector<unsigned int> idx = materials[i]->get_vrt(j);  // node index
      std::vector<double> phi_p     = materials[i]->get_phi(j);  // basis
      std::vector<double> a_star_p  = materials[i]->get_a_star(j);
      std::vector<double> a_p (gdim, 0.0);                       // acceleration
      
      // set the previous acceleration :
      materials[i]->set_a(j, a_star_p);

      // iterate through each component of velocity :
      for (unsigned int k = 0; k < gdim; k++)
      {
        // interpolate grid accleration to particles at each node q :
        for (unsigned int q = 0; q < sdim; q++)
        {
          a_p[k] += phi_p[q] * a3_grid_new[coord_arr[k]][idx[q]];
        }
      }
      materials[i]->set_a_star(j, a_p);
    }
    //materials[i]->calc_pi();
  }
}

void MPMModel::advect_material_particles()
{
  if (verbose == true)
    printf("--- C++ advect_material_particles() ---\n");

  // iterate through all materials :
  for (unsigned int i = 0; i < materials.size(); i++)
  {
    materials[i]->advect_particles(dt);
  }
}



