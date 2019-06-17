#include "MPMModel.h"

using namespace dolfin;
using namespace fenics_mpm;

MPMModel::MPMModel(std::shared_ptr<const FunctionSpace> V,
                   const unsigned int num_dofs,
                   double time_step,
                   bool verbosity) :
                   dt(time_step),
                   verbose(verbosity),
                   num_cells(V->mesh()->num_cells()),
                   gdim(V->mesh()->geometry().dim()),
                   sdim(V->element()->space_dimension()),
                   dofs(num_dofs),
                   Q(V),
                   element(V->element()),
                   mesh(V->mesh()),
                   bbt(V->mesh()->bounding_box_tree())
{
	printf("::: initializing MPMModelcpp with gDim = %u,  sDim = %u,"
	       " num_cells = %u, dt = %f :::\n", gdim, sdim, num_cells, dt);

	// basis function variables :
	vertex_coordinates.resize(num_cells);
	cell_dofs.resize(num_cells);
	cells.resize(num_cells);

	// create mini-continguous arrays for each vertex index, and
	// create a vector of Cells in advance to save considerable
	// time during formulate_material_basis_functions() :
	# pragma omp for schedule(auto)
	for (unsigned int i = 0; i < num_cells; ++i)
	{
		vertex_coordinates[i].resize(sdim);
		cell_dofs[i].resize(sdim);
		Cell cell(*mesh, i);
		cells[i] = cell;
		cell.get_vertex_coordinates(vertex_coordinates[i]);
		for (unsigned int j = 0; j < sdim; ++j)
			cell_dofs[i][j] = Q->dofmap()->cell_dofs(i)[j];
	}

	// scalars :
	h_grid.resize(dofs);             // cell diameter at node
	m_grid.resize(dofs);             // mass
	V_grid.resize(dofs);             // volume

	// vectors :
	u_x_grid.resize(dofs);           // velocity vector
	u_x_grid_new.resize(dofs);       // next velocity vector
	a_x_grid.resize(dofs);           // acceleration vector
	a_x_grid_new.resize(dofs);       // next acceleration vector
	f_int_x_grid.resize(dofs);       // internal force vector

	// in two or three dimensions, need y components :
	if (gdim == 2 or gdim == 3)
	{
		u_y_grid.resize(dofs);           // velocity vector
		u_y_grid_new.resize(dofs);       // next velocity vector
		a_y_grid.resize(dofs);           // acceleration vector
		a_y_grid_new.resize(dofs);       // next acceleration vector
		f_int_y_grid.resize(dofs);       // internal force vector
	}

	// in three dimensions, need z components :
	if (gdim == 2 or gdim == 3)
	{
		u_z_grid.resize(dofs);           // velocity vector
		u_z_grid_new.resize(dofs);       // next velocity vector
		a_z_grid.resize(dofs);           // acceleration vector
		a_z_grid_new.resize(dofs);       // next acceleration vector
		f_int_z_grid.resize(dofs);       // internal force vector
	}
}

// zero the vector in parallel :
void MPMModel::init_vector(std::vector<double>& vec)
{
	#pragma omp parallel
	{
		auto tid       = omp_get_thread_num();
		auto chunksize = vec.size() / omp_get_num_threads();
		auto begin     = vec.begin() + chunksize * tid;
		auto end       = (tid == omp_get_num_threads()-1 ?
		                  vec.end() : begin + chunksize);
		std::fill(begin, end, 0);
	}
}

void MPMModel::set_h(const std::vector<double>& h_a)
{
	# pragma omp parallel for simd schedule(auto)
	for (std::size_t i = 0; i < h_a.size(); ++i)
		h_grid[i] = h_a[i];
}

void MPMModel::set_V(const std::vector<double>& V_a)
{
	# pragma omp parallel for simd schedule(auto)
	for (std::size_t i = 0; i < V_a.size(); ++i)
		V_grid[i] = V_a[i];
}

void MPMModel::add_material(MPMMaterial& M)
{
	materials.push_back(&M);
}

void MPMModel::set_boundary_conditions(const std::vector<int>& vertices,
                                       const std::vector<double>& values)
{
	for (unsigned int i = 0; i < vertices.size(); ++i)
		bc_vrt.push_back(vertices[i]);
	for (unsigned int i = 0; i < values.size(); ++i)
		bc_val.push_back(values[i]);
}

void MPMModel::update_points()
{
	// iterate through all materials :
	for (unsigned int i = 0; i < materials.size(); ++i)
	{
		std::vector<Point*>& pt_i = materials[i]->get_x_pt(); // the particle Points

		// iterate through particle positions :
		# pragma omp parallel for simd schedule(auto)
		for (unsigned int j = 0; j < materials[i]->get_num_particles(); ++j)
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

	// node index :
	std::vector<unsigned int>& vrt_1 = M->get_vrt_1();
	std::vector<unsigned int>& vrt_2 = M->get_vrt_2();
	std::vector<unsigned int>& vrt_3 = M->get_vrt_3();
	std::vector<unsigned int>& vrt_4 = M->get_vrt_4();

	// basis functions :
	std::vector<double>& phi_1       = M->get_phi_1();
	std::vector<double>& phi_2       = M->get_phi_2();
	std::vector<double>& phi_3       = M->get_phi_3();
	std::vector<double>& phi_4       = M->get_phi_4();

	// basis function derivatives :
	std::vector<double>& grad_phi_1x = M->get_grad_phi_1x();
	std::vector<double>& grad_phi_1y = M->get_grad_phi_1y();
	std::vector<double>& grad_phi_1z = M->get_grad_phi_1z();
	std::vector<double>& grad_phi_2x = M->get_grad_phi_2x();
	std::vector<double>& grad_phi_2y = M->get_grad_phi_2y();
	std::vector<double>& grad_phi_2z = M->get_grad_phi_2z();
	std::vector<double>& grad_phi_3x = M->get_grad_phi_3x();
	std::vector<double>& grad_phi_3y = M->get_grad_phi_3y();
	std::vector<double>& grad_phi_3z = M->get_grad_phi_3z();
	std::vector<double>& grad_phi_4x = M->get_grad_phi_4x();
	std::vector<double>& grad_phi_4y = M->get_grad_phi_4y();
	std::vector<double>& grad_phi_4z = M->get_grad_phi_4z();

	// vector of Dolfin Points :
	std::vector<Point*>& x_pt        = M->get_x_pt();

	// temporary variables :
	unsigned int          c_id = 0;                  // cell index of point
	std::vector<double>   phi_temp(sdim);            // basis values for point
	std::vector<double>   grad_phi_temp(gdim*sdim);  // basis grad. vals. for pt.

	// iterate through particle positions and update the
	// grid node indices, basis values, and basis gradient values :
	# pragma omp parallel for simd schedule(auto) \
	  firstprivate(c_id, phi_temp, grad_phi_temp)
	for (unsigned int i = 0; i < M->get_num_particles(); ++i)
	{
		// first find the cell that the particle is in :
		c_id = bbt->compute_first_entity_collision(*x_pt[i]);

		// compute the basis values at the point :
		element->evaluate_basis_all(&phi_temp[0],
		                            x_pt[i]->coordinates(),
		                            vertex_coordinates[c_id].data(),
		                            cell_orientation);

		// compute the basis gradient values at the point :
		element->evaluate_basis_derivatives_all(deriv_order,
		                                        &grad_phi_temp[0],
		                                        x_pt[i]->coordinates(),
		                                        vertex_coordinates[c_id].data(),
		                                        cell_orientation);

		// all cells have at least two vertices :
		vrt_1[i] = cell_dofs[c_id][0];
		vrt_2[i] = cell_dofs[c_id][1];

		// two basis functions :
		phi_1[i] = phi_temp[0];
		phi_2[i] = phi_temp[1];

		// two or three dimensions have a third basis function :
		if (gdim == 2 or gdim == 3)
		{
			vrt_3[i] = cell_dofs[c_id][2];
			phi_3[i] = phi_temp[2];
		}

		// three dimensions another :
		if (gdim == 3)
		{
			vrt_4[i] = cell_dofs[c_id][3];
			phi_4[i] = phi_temp[3];
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
	for (unsigned int i = 0; i < materials.size(); ++i)
	{
		update_particle_basis_functions(materials[i]);
	}
}

void MPMModel::interpolate_material_mass_to_grid()
{
	if (verbose == true)
		printf("--- C++ interpolate_material_mass_to_grid() ---\n");

	// first reset the mass to zero :
	init_vector(m_grid);

	// iterate through all materials :
	for (unsigned int i = 0; i < materials.size(); ++i)
	{
		// node index :
		std::vector<unsigned int>& vrt_1 = materials[i]->get_vrt_1();
		std::vector<unsigned int>& vrt_2 = materials[i]->get_vrt_2();
		std::vector<unsigned int>& vrt_3 = materials[i]->get_vrt_3();
		std::vector<unsigned int>& vrt_4 = materials[i]->get_vrt_4();

		// basis functions :
		std::vector<double>& phi_1       = materials[i]->get_phi_1();
		std::vector<double>& phi_2       = materials[i]->get_phi_2();
		std::vector<double>& phi_3       = materials[i]->get_phi_3();
		std::vector<double>& phi_4       = materials[i]->get_phi_4();

		// mass :
		std::vector<double>& m           = materials[i]->get_m();

		# pragma omp parallel
		{
			std::vector<double> m_grid_lcl(dofs);

			// iterate through particles and interpolate the
			// particle mass to each node of its cell :
			# pragma omp for simd schedule(auto) nowait
			for (unsigned int j = 0; j < materials[i]->get_num_particles(); ++j)
			{
				// in one dimension, two vertices :
				m_grid_lcl[vrt_1[j]] += phi_1[j] * m[j];
				m_grid_lcl[vrt_2[j]] += phi_2[j] * m[j];

				// in two or three dimensions, one more :
				if (gdim == 2 or gdim == 3)
					m_grid_lcl[vrt_3[j]] += phi_3[j] * m[j];

				// in three dimensions, one more :
				if (gdim == 3)
					m_grid_lcl[vrt_4[j]] += phi_4[j] * m[j];
			}

			# pragma omp critical
			for (unsigned int j = 0; j < dofs; ++j)
				m_grid[j] += m_grid_lcl[j];
		}
	}
}

void MPMModel::interpolate_material_velocity_to_grid()
{
	if (verbose == true)
		printf("--- C++ interpolate_material_velocity_to_grid() ---\n");

	// first reset the velocity to zero :
	init_vector(u_x_grid);

	// y-component :
	if (gdim == 2 or gdim == 3)
		init_vector(u_y_grid);

	// z-component :
	if (gdim == 3)
		init_vector(u_z_grid);

	// iterate through all materials :
	for (unsigned int i = 0; i < materials.size(); ++i)
	{
		// node index :
		std::vector<unsigned int>& vrt_1 = materials[i]->get_vrt_1();
		std::vector<unsigned int>& vrt_2 = materials[i]->get_vrt_2();
		std::vector<unsigned int>& vrt_3 = materials[i]->get_vrt_3();
		std::vector<unsigned int>& vrt_4 = materials[i]->get_vrt_4();

		// basis functions :
		std::vector<double>& phi_1       = materials[i]->get_phi_1();
		std::vector<double>& phi_2       = materials[i]->get_phi_2();
		std::vector<double>& phi_3       = materials[i]->get_phi_3();
		std::vector<double>& phi_4       = materials[i]->get_phi_4();

		// velocity :
		std::vector<double>& u_x         = materials[i]->get_u_x();
		std::vector<double>& u_y         = materials[i]->get_u_y();
		std::vector<double>& u_z         = materials[i]->get_u_z();

		// mass :
		std::vector<double>& m           = materials[i]->get_m();

		# pragma omp parallel
		{
			std::vector<double> u_x_grid_lcl(dofs);
			std::vector<double> u_y_grid_lcl(dofs);
			std::vector<double> u_z_grid_lcl(dofs);

			// iterate through particles :
			# pragma omp for simd schedule(auto) nowait
			for (unsigned int j = 0; j < materials[i]->get_num_particles(); ++j)
			{
				// in one dimension, two vertices, one component of velocity :
				u_x_grid_lcl[vrt_1[j]] += u_x[j] * phi_1[j] * m[j] / m_grid[vrt_1[j]];
				u_x_grid_lcl[vrt_2[j]] += u_x[j] * phi_2[j] * m[j] / m_grid[vrt_2[j]];

				// in two or three dimensions, one more component and vertex :
				if (gdim == 2 or gdim == 3)
				{
					// extra node for x-velocity component :
					u_x_grid_lcl[vrt_3[j]] += u_x[j] * phi_3[j] * m[j] / m_grid[vrt_3[j]];

					// new y-component of velocity :
					u_y_grid_lcl[vrt_1[j]] += u_y[j] * phi_1[j] * m[j] / m_grid[vrt_1[j]];
					u_y_grid_lcl[vrt_2[j]] += u_y[j] * phi_2[j] * m[j] / m_grid[vrt_2[j]];
					u_y_grid_lcl[vrt_3[j]] += u_y[j] * phi_3[j] * m[j] / m_grid[vrt_3[j]];
				}

				// in three dimensions, one more component and vertex :
				if (gdim == 3)
				{
					// extra node for x- and y-velocity components :
					u_x_grid_lcl[vrt_4[j]] += u_x[j] * phi_4[j] * m[j] / m_grid[vrt_4[j]];
					u_y_grid_lcl[vrt_4[j]] += u_y[j] * phi_4[j] * m[j] / m_grid[vrt_4[j]];

					// new z-component of velocity :
					u_z_grid_lcl[vrt_1[j]] += u_z[j] * phi_1[j] * m[j] / m_grid[vrt_1[j]];
					u_z_grid_lcl[vrt_2[j]] += u_z[j] * phi_2[j] * m[j] / m_grid[vrt_2[j]];
					u_z_grid_lcl[vrt_3[j]] += u_z[j] * phi_3[j] * m[j] / m_grid[vrt_3[j]];
					u_z_grid_lcl[vrt_4[j]] += u_z[j] * phi_4[j] * m[j] / m_grid[vrt_4[j]];
				}
			}

			# pragma omp critical
			for (unsigned int j = 0; j < dofs; ++j)
			{
				// always an x-component :
				u_x_grid[j] += u_x_grid_lcl[j];

				// y-component :
				if (gdim == 2 or gdim == 3)
					u_y_grid[j] += u_y_grid_lcl[j];

				// z-component :
				if (gdim == 3)
					u_z_grid[j] += u_z_grid_lcl[j];
			}
		}
	}
}

void MPMModel::calculate_grid_volume()
{
	if (verbose == true)
		printf("--- C++ calculate_grid_volume() ---\n");

	# pragma omp parallel for simd schedule(auto)
	for (std::size_t i = 0; i < dofs; ++i)
		V_grid[i] = 4.0/3.0 * M_PI * pow(h_grid[i]/2.0, 3);
}


void MPMModel::calculate_material_initial_density()
{
	printf("--- C++ calculate_material_initial_density() ---\n");

	// iterate through all materials :
	for (unsigned int i = 0; i < materials.size(); ++i)
	{
		if (materials[i]->get_mass_init() == true)
		{
			printf("    - material `%s` has not been initialized with density -\n",
			       materials[i]->get_name());

			// node index :
			std::vector<unsigned int>& vrt_1 = materials[i]->get_vrt_1();
			std::vector<unsigned int>& vrt_2 = materials[i]->get_vrt_2();
			std::vector<unsigned int>& vrt_3 = materials[i]->get_vrt_3();
			std::vector<unsigned int>& vrt_4 = materials[i]->get_vrt_4();

			// basis functions :
			std::vector<double>& phi_1       = materials[i]->get_phi_1();
			std::vector<double>& phi_2       = materials[i]->get_phi_2();
			std::vector<double>& phi_3       = materials[i]->get_phi_3();
			std::vector<double>& phi_4       = materials[i]->get_phi_4();

			// mass and density :
			std::vector<double>& rho         = materials[i]->get_rho();
			std::vector<double>& rho0        = materials[i]->get_rho0();

			// iterate through particles :
			# pragma omp parallel for simd schedule(auto)
			for (unsigned int j = 0; j < materials[i]->get_num_particles(); ++j)
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
	for (unsigned int i = 0; i < materials.size(); ++i)
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

	// iterate through all materials :
	for (unsigned int i = 0; i < materials.size(); ++i)
	{
		materials[i]->initialize_tensors(dt);
	}
}

void MPMModel::calculate_grid_internal_forces()
{
	if (verbose == true)
		printf("--- C++ update_grid_internal_forces() ---\n");

	// first reset the forces to zero :
	init_vector(f_int_x_grid);

	// y-component :
	if (gdim == 2 or gdim == 3)
		init_vector(f_int_y_grid);

	// z-component :
	if (gdim == 3)
		init_vector(f_int_z_grid);

	// iterate through all materials :
	for (unsigned int i = 0; i < materials.size(); ++i)
	{
		// node index :
		std::vector<unsigned int>& vrt_1 = materials[i]->get_vrt_1();
		std::vector<unsigned int>& vrt_2 = materials[i]->get_vrt_2();
		std::vector<unsigned int>& vrt_3 = materials[i]->get_vrt_3();
		std::vector<unsigned int>& vrt_4 = materials[i]->get_vrt_4();

		// basis function derivatives :
		std::vector<double>& grad_phi_1x = materials[i]->get_grad_phi_1x();
		std::vector<double>& grad_phi_1y = materials[i]->get_grad_phi_1y();
		std::vector<double>& grad_phi_1z = materials[i]->get_grad_phi_1z();
		std::vector<double>& grad_phi_2x = materials[i]->get_grad_phi_2x();
		std::vector<double>& grad_phi_2y = materials[i]->get_grad_phi_2y();
		std::vector<double>& grad_phi_2z = materials[i]->get_grad_phi_2z();
		std::vector<double>& grad_phi_3x = materials[i]->get_grad_phi_3x();
		std::vector<double>& grad_phi_3y = materials[i]->get_grad_phi_3y();
		std::vector<double>& grad_phi_3z = materials[i]->get_grad_phi_3z();
		std::vector<double>& grad_phi_4x = materials[i]->get_grad_phi_4x();
		std::vector<double>& grad_phi_4y = materials[i]->get_grad_phi_4y();
		std::vector<double>& grad_phi_4z = materials[i]->get_grad_phi_4z();

		// stress tensor :
		std::vector<double>& sigma_xx    = materials[i]->get_sigma_xx();
		std::vector<double>& sigma_xy    = materials[i]->get_sigma_xy();
		std::vector<double>& sigma_xz    = materials[i]->get_sigma_xz();
		std::vector<double>& sigma_yy    = materials[i]->get_sigma_yy();
		std::vector<double>& sigma_yz    = materials[i]->get_sigma_yz();
		std::vector<double>& sigma_zz    = materials[i]->get_sigma_zz();

		// volume :
		std::vector<double>& V           = materials[i]->get_V();

		# pragma omp parallel
		{
			std::vector<double> f_int_x_grid_lcl(dofs);
			std::vector<double> f_int_y_grid_lcl(dofs);
			std::vector<double> f_int_z_grid_lcl(dofs);

			// iterate through particles :
			# pragma omp for simd schedule(auto) nowait
			for (unsigned int j = 0; j < materials[i]->get_num_particles(); ++j)
			{
				// in one dimension, two vertices, one component of force :
				f_int_x_grid_lcl[vrt_1[j]] -= sigma_xx[j] * grad_phi_1x[j] * V[j];
				f_int_x_grid_lcl[vrt_2[j]] -= sigma_xx[j] * grad_phi_2x[j] * V[j];

				// in two or three dimensions, one more component and vertex :
				if (gdim == 2 or gdim == 3)
				{
					// extra node for x-force diagonal component :
					f_int_x_grid_lcl[vrt_3[j]] -= sigma_xx[j] * grad_phi_3x[j] * V[j];

					// off-diagonal terms :
					f_int_x_grid_lcl[vrt_1[j]] -= sigma_xy[j] * grad_phi_1y[j] * V[j];
					f_int_x_grid_lcl[vrt_2[j]] -= sigma_xy[j] * grad_phi_2y[j] * V[j];
					f_int_x_grid_lcl[vrt_3[j]] -= sigma_xy[j] * grad_phi_3y[j] * V[j];

					// new diagonal y-component of velocity :
					f_int_y_grid_lcl[vrt_1[j]] -= sigma_yy[j] * grad_phi_1y[j] * V[j];
					f_int_y_grid_lcl[vrt_2[j]] -= sigma_yy[j] * grad_phi_2y[j] * V[j];
					f_int_y_grid_lcl[vrt_3[j]] -= sigma_yy[j] * grad_phi_3y[j] * V[j];

					// and off-diagonal terms :
					f_int_y_grid_lcl[vrt_1[j]] -= sigma_xy[j] * grad_phi_1x[j] * V[j];
					f_int_y_grid_lcl[vrt_2[j]] -= sigma_xy[j] * grad_phi_2x[j] * V[j];
					f_int_y_grid_lcl[vrt_3[j]] -= sigma_xy[j] * grad_phi_3x[j] * V[j];
				}

				// in three dimensions, one more component and vertex :
				if (gdim == 3)
				{
					// extra node for x- and y-force diagonal components :
					f_int_x_grid_lcl[vrt_4[j]] -= sigma_xx[j] * grad_phi_4x[j] * V[j];
					f_int_y_grid_lcl[vrt_4[j]] -= sigma_yy[j] * grad_phi_4y[j] * V[j];

					// and off-diagonal terms :
					f_int_x_grid_lcl[vrt_1[j]] -= sigma_xz[j] * grad_phi_1z[j] * V[j];
					f_int_x_grid_lcl[vrt_2[j]] -= sigma_xz[j] * grad_phi_2z[j] * V[j];
					f_int_x_grid_lcl[vrt_3[j]] -= sigma_xz[j] * grad_phi_3z[j] * V[j];
					f_int_x_grid_lcl[vrt_4[j]] -= sigma_xz[j] * grad_phi_4z[j] * V[j];

					f_int_y_grid_lcl[vrt_1[j]] -= sigma_yz[j] * grad_phi_1z[j] * V[j];
					f_int_y_grid_lcl[vrt_2[j]] -= sigma_yz[j] * grad_phi_2z[j] * V[j];
					f_int_y_grid_lcl[vrt_3[j]] -= sigma_yz[j] * grad_phi_3z[j] * V[j];
					f_int_y_grid_lcl[vrt_4[j]] -= sigma_yz[j] * grad_phi_4z[j] * V[j];

					// new diagonal z-component of force :
					f_int_z_grid_lcl[vrt_1[j]] -= sigma_zz[j] * grad_phi_1z[j] * V[j];
					f_int_z_grid_lcl[vrt_2[j]] -= sigma_zz[j] * grad_phi_2z[j] * V[j];
					f_int_z_grid_lcl[vrt_3[j]] -= sigma_zz[j] * grad_phi_3z[j] * V[j];
					f_int_z_grid_lcl[vrt_4[j]] -= sigma_zz[j] * grad_phi_4z[j] * V[j];

					// and two off-diagonal terms :
					f_int_z_grid_lcl[vrt_1[j]] -= sigma_xz[j] * grad_phi_1x[j] * V[j];
					f_int_z_grid_lcl[vrt_2[j]] -= sigma_xz[j] * grad_phi_2x[j] * V[j];
					f_int_z_grid_lcl[vrt_3[j]] -= sigma_xz[j] * grad_phi_3x[j] * V[j];
					f_int_z_grid_lcl[vrt_4[j]] -= sigma_xz[j] * grad_phi_4x[j] * V[j];

					f_int_z_grid_lcl[vrt_1[j]] -= sigma_yz[j] * grad_phi_1y[j] * V[j];
					f_int_z_grid_lcl[vrt_2[j]] -= sigma_yz[j] * grad_phi_2y[j] * V[j];
					f_int_z_grid_lcl[vrt_3[j]] -= sigma_yz[j] * grad_phi_3y[j] * V[j];
					f_int_z_grid_lcl[vrt_4[j]] -= sigma_yz[j] * grad_phi_4y[j] * V[j];
				}
			}

			# pragma omp critical
			for (unsigned int j = 0; j < dofs; ++j)
			{
				// always an x-component :
				f_int_x_grid[j] += f_int_x_grid_lcl[j];

				// y-component :
				if (gdim == 2 or gdim == 3)
					f_int_y_grid[j] += f_int_y_grid_lcl[j];

				// z-component :
				if (gdim == 3)
					f_int_z_grid[j] += f_int_z_grid_lcl[j];
			}
		}
	}
}

void MPMModel::calculate_grid_acceleration(double m_min)
{
	if (verbose == true)
		printf("--- C++ calculate_grid_acceleration() ---\n");

	// iterate through each node :
	# pragma omp parallel for simd schedule(auto)
	for (unsigned int i = 0; i < dofs; ++i)
	{
		a_x_grid[i] = a_x_grid_new[i];

		// in two or three dimensions, one more component :
		if (gdim == 2 or gdim == 3)
			a_y_grid[i] = a_y_grid_new[i];

		// in three dimensions, one more component :
		if (gdim == 3)
			a_z_grid[i] = a_z_grid_new[i];
	}

	// iterate through each node :
	# pragma omp parallel for simd schedule(auto)
	for (unsigned int i = 0; i < dofs; ++i)
	{
		// if there is a mass on the grid node :
		if (m_grid[i] > 0)
		{
			// set the minimum mass :
			if (m_grid[i] < m_min)
				m_grid[i] = m_min;

			// update acceleration :
			a_x_grid_new[i] = f_int_x_grid[i] / m_grid[i];

			// in two or three dimensions, one more component :
			if (gdim == 2 or gdim == 3)
				a_y_grid_new[i] = f_int_y_grid[i] / m_grid[i];

			// in three dimensions, one more component :
			if (gdim == 3)
				a_y_grid_new[i] = f_int_y_grid[i] / m_grid[i];
			}
		// otherwise set the grid values to zero :
		else
		{
			f_int_x_grid[i] = 0.0;
			a_x_grid_new[i] = 0.0;

			// in two or three dimensions, one more component :
			if (gdim == 2 or gdim == 3)
			{
				f_int_y_grid[i] = 0.0;
				a_y_grid_new[i] = 0.0;
			}

			// in three dimensions, one more component :
			if (gdim == 3)
			{
				f_int_z_grid[i] = 0.0;
				a_z_grid_new[i] = 0.0;
			}
		}
	}

}

void MPMModel::update_grid_velocity()
{
	if (verbose == true)
		printf("--- C++ update_grid_velocity() ---\n");

	// iterate through each node :
	# pragma omp parallel for simd schedule(auto)
	for (unsigned int i = 0; i < dofs; ++i)
	{
		// always at least one component of velocity :
		u_x_grid_new[i]   = u_x_grid[i] + 0.5*(a_x_grid[i] + a_x_grid_new[i]) * dt;

		// in two or three dimensions, one more component :
		if (gdim == 2 or gdim == 3)
			u_y_grid_new[i] = u_y_grid[i] + 0.5*(a_y_grid[i] + a_y_grid_new[i]) * dt;

		// in three dimensions, one more component :
		if (gdim == 3)
			u_z_grid_new[i] = u_z_grid[i] + 0.5*(a_z_grid[i] + a_z_grid_new[i]) * dt;
	}

	/*
	// TODO: set boolean flag for using Dirichlet or not :
	// apply boundary conditions if present :
	for (unsigned int i = 0; i < bc_val.size(); ++i)
	{
		for (unsigned int j = 0; j < bc_vrt.size(); ++j)
		{
			u_x_grid[bc_vrt[j]] = 0.0;
			u_y_grid[bc_vrt[j]] = 0.0;
			u_z_grid[bc_vrt[j]] = 0.0;
		}
	}
	*/
}

void MPMModel::calculate_material_velocity_gradient()
{
	if (verbose == true)
		printf("--- C++ calculate_material_velocity_gradient() ---\n");

	// iterate through all materials :
	for (unsigned int i = 0; i < materials.size(); ++i)
	{
		// node index :
		std::vector<unsigned int>& vrt_1    = materials[i]->get_vrt_1();
		std::vector<unsigned int>& vrt_2    = materials[i]->get_vrt_2();
		std::vector<unsigned int>& vrt_3    = materials[i]->get_vrt_3();
		std::vector<unsigned int>& vrt_4    = materials[i]->get_vrt_4();

		// basis function derivatives :
		std::vector<double>& grad_phi_1x    = materials[i]->get_grad_phi_1x();
		std::vector<double>& grad_phi_1y    = materials[i]->get_grad_phi_1y();
		std::vector<double>& grad_phi_1z    = materials[i]->get_grad_phi_1z();
		std::vector<double>& grad_phi_2x    = materials[i]->get_grad_phi_2x();
		std::vector<double>& grad_phi_2y    = materials[i]->get_grad_phi_2y();
		std::vector<double>& grad_phi_2z    = materials[i]->get_grad_phi_2z();
		std::vector<double>& grad_phi_3x    = materials[i]->get_grad_phi_3x();
		std::vector<double>& grad_phi_3y    = materials[i]->get_grad_phi_3y();
		std::vector<double>& grad_phi_3z    = materials[i]->get_grad_phi_3z();
		std::vector<double>& grad_phi_4x    = materials[i]->get_grad_phi_4x();
		std::vector<double>& grad_phi_4y    = materials[i]->get_grad_phi_4y();
		//std::vector<double>& grad_phi_4z    = materials[i]->get_grad_phi_4z();

		// current velocity gradient tensor :
		std::vector<double>& grad_u_xx      = materials[i]->get_grad_u_xx();
		std::vector<double>& grad_u_xy      = materials[i]->get_grad_u_xy();
		std::vector<double>& grad_u_xz      = materials[i]->get_grad_u_xz();
		std::vector<double>& grad_u_yx      = materials[i]->get_grad_u_yx();
		std::vector<double>& grad_u_yy      = materials[i]->get_grad_u_yy();
		std::vector<double>& grad_u_yz      = materials[i]->get_grad_u_yz();
		std::vector<double>& grad_u_zx      = materials[i]->get_grad_u_zx();
		std::vector<double>& grad_u_zy      = materials[i]->get_grad_u_zy();
		std::vector<double>& grad_u_zz      = materials[i]->get_grad_u_zz();

		// new velocity gradient tensor :
		std::vector<double>& grad_u_xx_star = materials[i]->get_grad_u_xx_star();
		std::vector<double>& grad_u_xy_star = materials[i]->get_grad_u_xy_star();
		std::vector<double>& grad_u_xz_star = materials[i]->get_grad_u_xz_star();
		std::vector<double>& grad_u_yx_star = materials[i]->get_grad_u_yx_star();
		std::vector<double>& grad_u_yy_star = materials[i]->get_grad_u_yy_star();
		std::vector<double>& grad_u_yz_star = materials[i]->get_grad_u_yz_star();
		std::vector<double>& grad_u_zx_star = materials[i]->get_grad_u_zx_star();
		std::vector<double>& grad_u_zy_star = materials[i]->get_grad_u_zy_star();
		std::vector<double>& grad_u_zz_star = materials[i]->get_grad_u_zz_star();

		// iterate through particles :
		# pragma omp parallel for simd schedule(auto)
		for (unsigned int j = 0; j < materials[i]->get_num_particles(); ++j)
		{
			// first, set the previous velocity gradient :
			grad_u_xx[j]      = grad_u_xx_star[j];
			grad_u_xx_star[j] = 0.0;

			// in one dimension, two vertices, one derivative :
			grad_u_xx_star[j] += grad_phi_1x[j] * u_x_grid_new[vrt_1[j]];
			grad_u_xx_star[j] += grad_phi_2x[j] * u_x_grid_new[vrt_2[j]];

			// in two or three dimensions, one more component and vertex :
			if (gdim == 2 or gdim == 3)
			{
				// first, set the previous velocity gradient :
				grad_u_xy[j]      = grad_u_xy_star[j];
				grad_u_yx[j]      = grad_u_yx_star[j];
				grad_u_yy[j]      = grad_u_yy_star[j];
				grad_u_xy_star[j] = 0.0;
				grad_u_yx_star[j] = 0.0;
				grad_u_yy_star[j] = 0.0;

				// extra node for x-velocity gradient diagonal component :
				grad_u_xx_star[j] += grad_phi_3x[j] * u_x_grid_new[vrt_3[j]];

				// off-diagonal terms :
				grad_u_xy_star[j] += grad_phi_1y[j] * u_x_grid_new[vrt_1[j]];
				grad_u_xy_star[j] += grad_phi_2y[j] * u_x_grid_new[vrt_2[j]];
				grad_u_xy_star[j] += grad_phi_3y[j] * u_x_grid_new[vrt_3[j]];

				// new diagonal y-component of velocity gradient :
				grad_u_yy_star[j] += grad_phi_1y[j] * u_y_grid_new[vrt_1[j]];
				grad_u_yy_star[j] += grad_phi_2y[j] * u_y_grid_new[vrt_2[j]];
				grad_u_yy_star[j] += grad_phi_3y[j] * u_y_grid_new[vrt_3[j]];

				// and off-diagonal terms :
				grad_u_yx_star[j] += grad_phi_1x[j] * u_y_grid_new[vrt_1[j]];
				grad_u_yx_star[j] += grad_phi_2x[j] * u_y_grid_new[vrt_2[j]];
				grad_u_yx_star[j] += grad_phi_3x[j] * u_y_grid_new[vrt_3[j]];
			}

			// in three dimensions, one more component and vertex :
			if (gdim == 3)
			{
				// first, set the previous velocity gradient :
				grad_u_xz[j]      = grad_u_xz_star[j];
				grad_u_yz[j]      = grad_u_yz_star[j];
				grad_u_zx[j]      = grad_u_zx_star[j];
				grad_u_zy[j]      = grad_u_zy_star[j];
				grad_u_zz[j]      = grad_u_zz_star[j];
				grad_u_xz_star[j] = 0.0;
				grad_u_yz_star[j] = 0.0;
				grad_u_zx_star[j] = 0.0;
				grad_u_zy_star[j] = 0.0;
				grad_u_zz_star[j] = 0.0;

				// extra node for x- and y-velocity gradient diagonal components :
				grad_u_xx_star[j] += grad_phi_4x[j] * u_x_grid_new[vrt_4[j]];
				grad_u_yy_star[j] += grad_phi_4y[j] * u_y_grid_new[vrt_4[j]];

				// and off-diagonal terms :
				grad_u_xz_star[j] += grad_phi_1z[j] * u_x_grid_new[vrt_1[j]];
				grad_u_xz_star[j] += grad_phi_2z[j] * u_x_grid_new[vrt_2[j]];
				grad_u_xz_star[j] += grad_phi_3z[j] * u_x_grid_new[vrt_3[j]];

				grad_u_yz_star[j] += grad_phi_1z[j] * u_y_grid_new[vrt_1[j]];
				grad_u_yz_star[j] += grad_phi_2z[j] * u_y_grid_new[vrt_2[j]];
				grad_u_yz_star[j] += grad_phi_3z[j] * u_y_grid_new[vrt_3[j]];

				// new diagonal z-component of velocity gradient :
				grad_u_zz_star[j] += grad_phi_1z[j] * u_z_grid_new[vrt_1[j]];
				grad_u_zz_star[j] += grad_phi_2z[j] * u_z_grid_new[vrt_2[j]];
				grad_u_zz_star[j] += grad_phi_3z[j] * u_z_grid_new[vrt_3[j]];

				// and two off-diagonal terms :
				grad_u_zx_star[j] += grad_phi_1x[j] * u_z_grid_new[vrt_1[j]];
				grad_u_zx_star[j] += grad_phi_2x[j] * u_z_grid_new[vrt_2[j]];
				grad_u_zx_star[j] += grad_phi_3x[j] * u_z_grid_new[vrt_3[j]];

				grad_u_zy_star[j] += grad_phi_1y[j] * u_z_grid_new[vrt_1[j]];
				grad_u_zy_star[j] += grad_phi_2y[j] * u_z_grid_new[vrt_2[j]];
				grad_u_zy_star[j] += grad_phi_3y[j] * u_z_grid_new[vrt_3[j]];
			}
		}
	}
}

void MPMModel::update_material_deformation_gradient()
{
	if (verbose == true)
		printf("--- C++ update_material_deformation_gradient() ---\n");

	// iterate through all materials :
	for (unsigned int i = 0; i < materials.size(); ++i)
	{
		materials[i]->update_deformation_gradient(dt);
	}
}

void MPMModel::update_material_density()
{
	if (verbose == true)
		printf("--- C++ update_material_density() ---\n");

	// iterate through all materials :
	for (unsigned int i = 0; i < materials.size(); ++i)
	{
		materials[i]->update_density();
	}
}

void MPMModel::update_material_volume()
{
	if (verbose == true)
		printf("--- C++ update_material_volume() ---\n");

	// iterate through all materials :
	for (unsigned int i = 0; i < materials.size(); ++i)
	{
		materials[i]->update_volume();
	}
}

void MPMModel::update_material_stress()
{
	if (verbose == true)
		printf("--- C++ update_material_stress() ---\n");

	// iterate through all materials :
	for (unsigned int i = 0; i < materials.size(); ++i)
	{
		materials[i]->update_stress(dt);
	}
}

void MPMModel::interpolate_grid_velocity_to_material()
{
	if (verbose == true)
		printf("--- C++ interpolate_grid_velocity_to_material() ---\n");

	// iterate through all materials :
	for (unsigned int i = 0; i < materials.size(); ++i)
	{
		// node index :
		std::vector<unsigned int>& vrt_1 = materials[i]->get_vrt_1();
		std::vector<unsigned int>& vrt_2 = materials[i]->get_vrt_2();
		std::vector<unsigned int>& vrt_3 = materials[i]->get_vrt_3();
		std::vector<unsigned int>& vrt_4 = materials[i]->get_vrt_4();

		// basis functions :
		std::vector<double>& phi_1       = materials[i]->get_phi_1();
		std::vector<double>& phi_2       = materials[i]->get_phi_2();
		std::vector<double>& phi_3       = materials[i]->get_phi_3();
		std::vector<double>& phi_4       = materials[i]->get_phi_4();

		// new velocity :
		std::vector<double>& u_x_star    = materials[i]->get_u_x_star();
		std::vector<double>& u_y_star    = materials[i]->get_u_y_star();
		std::vector<double>& u_z_star    = materials[i]->get_u_z_star();

		// iterate through particles and interpolate grid velocity
		// back to particles from each node :
		# pragma omp parallel for simd schedule(auto)
		for (unsigned int j = 0; j < materials[i]->get_num_particles(); ++j)
		{
			// first reset the velocity :
			u_x_star[j] = 0.0;

			// always at least one component of velocity and two nodes :
			u_x_star[j] += phi_1[j] * u_x_grid_new[vrt_1[j]];
			u_x_star[j] += phi_2[j] * u_x_grid_new[vrt_2[j]];

			// in two or three dimensions, one more component and vertex :
			if (gdim == 2 or gdim == 3)
			{
				// first reset the velocity :
				u_y_star[j] = 0.0;

				// extra node for x-velocity component :
				u_x_star[j] += phi_3[j] * u_x_grid_new[vrt_3[j]];

				// new y-component of velocity :
				u_y_star[j] += phi_1[j] * u_y_grid_new[vrt_1[j]];
				u_y_star[j] += phi_2[j] * u_y_grid_new[vrt_2[j]];
				u_y_star[j] += phi_3[j] * u_y_grid_new[vrt_3[j]];
			}

			// in three dimensions, one more component and vertex :
			if (gdim == 3)
			{
				// first reset the velocity :
				u_z_star[j] = 0.0;

				// extra node for x- and y-velocity components :
				u_x_star[j] += phi_4[j] * u_x_grid_new[vrt_4[j]];
				u_y_star[j] += phi_4[j] * u_y_grid_new[vrt_4[j]];

				// new z-component of velocity :
				u_z_star[j] += phi_1[j] * u_z_grid_new[vrt_1[j]];
				u_z_star[j] += phi_2[j] * u_z_grid_new[vrt_2[j]];
				u_z_star[j] += phi_3[j] * u_z_grid_new[vrt_3[j]];
				u_z_star[j] += phi_4[j] * u_z_grid_new[vrt_4[j]];
			}
		}
	}
}

void MPMModel::interpolate_grid_acceleration_to_material()
{
	if (verbose == true)
		printf("--- C++ interpolate_grid_acceleration_to_material() ---\n");

	// iterate through all materials :
	for (unsigned int i = 0; i < materials.size(); ++i)
	{
		// node index :
		std::vector<unsigned int>& vrt_1 = materials[i]->get_vrt_1();
		std::vector<unsigned int>& vrt_2 = materials[i]->get_vrt_2();
		std::vector<unsigned int>& vrt_3 = materials[i]->get_vrt_3();
		std::vector<unsigned int>& vrt_4 = materials[i]->get_vrt_4();

		// basis functions :
		std::vector<double>& phi_1       = materials[i]->get_phi_1();
		std::vector<double>& phi_2       = materials[i]->get_phi_2();
		std::vector<double>& phi_3       = materials[i]->get_phi_3();
		std::vector<double>& phi_4       = materials[i]->get_phi_4();

		// current acceleration :
		std::vector<double>& a_x         = materials[i]->get_a_x();
		std::vector<double>& a_y         = materials[i]->get_a_y();
		std::vector<double>& a_z         = materials[i]->get_a_z();

		// new acceleration :
		std::vector<double>& a_x_star    = materials[i]->get_a_x_star();
		std::vector<double>& a_y_star    = materials[i]->get_a_y_star();
		std::vector<double>& a_z_star    = materials[i]->get_a_z_star();

		// iterate through particles and interpolate grid accleration
		// to particles at each node :
		# pragma omp parallel for simd schedule(auto)
		for (unsigned int j = 0; j < materials[i]->get_num_particles(); ++j)
		{
			// first, set the previous acceleration :
			a_x[j]      = a_x_star[j];
			a_x_star[j] = 0.0;

			// always at least one component of acceleration and two nodes :
			a_x_star[j] += phi_1[j] * a_x_grid_new[vrt_1[j]];
			a_x_star[j] += phi_2[j] * a_x_grid_new[vrt_2[j]];

			// in two or three dimensions, one more component and vertex :
			if (gdim == 2 or gdim == 3)
			{
				// first, set the previous acceleration :
				a_y[j]      = a_y_star[j];
				a_y_star[j] = 0.0;

				// extra node for x-acceleration component :
				a_x_star[j] += phi_3[j] * a_x_grid_new[vrt_3[j]];

				// new y-component of acceleration :
				a_y_star[j] += phi_1[j] * a_y_grid_new[vrt_1[j]];
				a_y_star[j] += phi_2[j] * a_y_grid_new[vrt_2[j]];
				a_y_star[j] += phi_3[j] * a_y_grid_new[vrt_3[j]];
			}

			// in three dimensions, one more component and vertex :
			if (gdim == 3)
			{
				// first, set the previous acceleration :
				a_z[j]      = a_z_star[j];
				a_z_star[j] = 0.0;

				// extra node for x- and y-acceleration components :
				a_x_star[j] += phi_4[j] * a_x_grid_new[vrt_4[j]];
				a_y_star[j] += phi_4[j] * a_y_grid_new[vrt_4[j]];

				// new z-component of acceleration :
				a_z_star[j] += phi_1[j] * a_z_grid_new[vrt_1[j]];
				a_z_star[j] += phi_2[j] * a_z_grid_new[vrt_2[j]];
				a_z_star[j] += phi_3[j] * a_z_grid_new[vrt_3[j]];
				a_z_star[j] += phi_4[j] * a_z_grid_new[vrt_4[j]];
			}
		}
		//materials[i]->calc_pi();
	}
}

void MPMModel::advect_material_particles()
{
	if (verbose == true)
		printf("--- C++ advect_material_particles() ---\n");

	// iterate through all materials :
	for (unsigned int i = 0; i < materials.size(); ++i)
	{
		materials[i]->advect_particles(dt);
	}
}

void MPMModel::mpm(bool initialize)
{
	if (verbose == true)
		printf("--- C++ mpm() ---\n");

	// get basis function values at material point locations :
	formulate_material_basis_functions();

	// interpolation from particle stage :
	interpolate_material_mass_to_grid();
	interpolate_material_velocity_to_grid();

	// initialization step :
	if (initialize)
	{
		initialize_material_tensors();
		calculate_material_initial_density();
		calculate_material_initial_volume();
	}

	// grid calculation stage :
	calculate_grid_internal_forces();
	calculate_grid_acceleration();
	update_grid_velocity();

	// particle calculation stage :
	calculate_material_velocity_gradient();
	update_material_deformation_gradient();
	update_material_volume();
	update_material_stress();
	interpolate_grid_acceleration_to_material();
	interpolate_grid_velocity_to_material();
	advect_material_particles();
}



