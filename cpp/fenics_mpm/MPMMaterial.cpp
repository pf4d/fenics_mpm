#include "MPMMaterial.h"

using namespace fenics_mpm;

// no destructor yet :
MPMMaterial::~MPMMaterial() { }

// initialize the MPMMaterial variables :
MPMMaterial::MPMMaterial(const std::string&   name,
                         const int            n,
                         const std::vector<double>& x_a,
                         const std::vector<double>& u_a,
                         const dolfin::FiniteElement& element) :
                           name(name),
                           n_p(n),
                           gdim(element.geometric_dimension()),
                           sdim(element.space_dimension())
{
	printf("::: initializing MPMMaterialcpp `%s` with n_p = %u,  gdim = %u,"
	       "  sdim = %u :::\n", name.c_str(), n_p, gdim, sdim);

	// TODO: alignment for SIMD vectorize help ?
	# pragma omp parallel for
	for (unsigned int i = 0; i < n_p; i += 1024)
		n_p_end = min(i+1024, n_p);

	// one component for each particle :
	m.resize(n_p);
	rho0.resize(n_p);
	rho.resize(n_p);
	V0.resize(n_p);
	V.resize(n_p);
	x_pt.resize(n_p);
	det_dF.resize(n_p);

	// these are vectors in topological dimension :
	x.resize(n_p);
	u_x.resize(n_p);
	u_x_star.resize(n_p);
	a_x_star.resize(n_p);
	a_x.resize(n_p);

	// these are components of rank-two flattened tensors defined over each
	// topological dimension :
	grad_u_xx.resize(n_p);
	grad_u_xx_star.resize(n_p);
	dF_xx.resize(n_p);
	F_xx.resize(n_p);
	sigma_xx.resize(n_p);
	epsilon_xx.resize(n_p);
	depsilon_xx.resize(n_p);

	// these are vectors in element dimension;
	// always at least two nodes per cell :
	// TODO: allow higher-order function spaces
	vrt_1.resize(n_p);
	vrt_2.resize(n_p);
	phi_1.resize(n_p);
	phi_2.resize(n_p);

	// this is a flattened tensor defined with columns over each
	//   topological dim. and rows over each element dimension :
	grad_phi_1x.resize(n_p);
	grad_phi_2x.resize(n_p);

	// if this is a two- or three-dimensional problem,
	// allocate space for the y componets :
	if (gdim == 2 or gdim == 3)
	{
		// one extra spatial coordinate :
		y.resize(n_p);

		// one extra vector component :
		u_y.resize(n_p);
		u_y_star.resize(n_p);
		a_y_star.resize(n_p);
		a_y.resize(n_p);

		// the velocity gradient is not symetric :
		grad_u_xy.resize(n_p);
		grad_u_yx.resize(n_p);
		grad_u_yy.resize(n_p);
		grad_u_xy_star.resize(n_p);
		grad_u_yx_star.resize(n_p);
		grad_u_yy_star.resize(n_p);
		dF_xy.resize(n_p);
		dF_yx.resize(n_p);
		dF_yy.resize(n_p);
		F_xy.resize(n_p);
		F_yx.resize(n_p);
		F_yy.resize(n_p);

		// two extra symmetric tensor components :
		sigma_xy.resize(n_p);
		sigma_yy.resize(n_p);
		epsilon_xy.resize(n_p);
		epsilon_yy.resize(n_p);
		depsilon_xy.resize(n_p);
		depsilon_yy.resize(n_p);

		// basis functions have an additional vertex with triangle :
		vrt_3.resize(n_p);
		phi_3.resize(n_p);

		// basis function gradient have four more components :
		grad_phi_1y.resize(n_p);
		grad_phi_2y.resize(n_p);
		grad_phi_3x.resize(n_p);
		grad_phi_3y.resize(n_p);
	}

	// if this is a three-dimensional problem,
	// allocate space for the z components :
	if (gdim == 3)
	{
		// one extra spatial coordinate :
		z.resize(n_p);

		// one extra vector component :
		u_z.resize(n_p);
		u_z_star.resize(n_p);
		a_z_star.resize(n_p);
		a_z.resize(n_p);

		// the velocity gradient is not symetric :
		grad_u_xz.resize(n_p);
		grad_u_yz.resize(n_p);
		grad_u_zx.resize(n_p);
		grad_u_zy.resize(n_p);
		grad_u_zz.resize(n_p);
		grad_u_xz_star.resize(n_p);
		grad_u_yz_star.resize(n_p);
		grad_u_zx_star.resize(n_p);
		grad_u_zy_star.resize(n_p);
		grad_u_zz_star.resize(n_p);
		dF_xz.resize(n_p);
		dF_yz.resize(n_p);
		dF_zx.resize(n_p);
		dF_zy.resize(n_p);
		dF_zz.resize(n_p);
		F_xz.resize(n_p);
		F_yz.resize(n_p);
		F_zx.resize(n_p);
		F_zy.resize(n_p);
		F_zz.resize(n_p);

		// three extra tensor components :
		sigma_xz.resize(n_p);
		sigma_yz.resize(n_p);
		sigma_zz.resize(n_p);
		epsilon_xz.resize(n_p);
		epsilon_yz.resize(n_p);
		epsilon_zz.resize(n_p);
		depsilon_xz.resize(n_p);
		depsilon_yz.resize(n_p);
		depsilon_zz.resize(n_p);

		// basis functions have one more vertex for tetrahedra :
		vrt_4.resize(n_p);
		phi_4.resize(n_p);

		// basis function gradient has six more components :
		grad_phi_1z.resize(n_p);
		grad_phi_2z.resize(n_p);
		grad_phi_3z.resize(n_p);
		grad_phi_4x.resize(n_p);
		grad_phi_4y.resize(n_p);
		grad_phi_4z.resize(n_p);
	}

	// initialize the positions, Points, and velocities :
	# pragma omp parallel for simd schedule(auto)
	for (unsigned int i = 0; i < n_p; ++i)
	{
		std::vector<double> x_t = {0.0, 0.0, 0.0};   // the vector to make a Point

		// we always have one dimension :
		x[i]        = x_a[i*gdim];
		u_x[i]      = u_a[i*gdim];
		u_x_star[i] = u_a[i*gdim];
		x_t[0]      = x_a[i*gdim];

		// tack on another :
		if (gdim == 2 or gdim == 3)
		{
			y[i]        = x_a[i*gdim + 1];
			u_y[i]      = u_a[i*gdim + 1];
			u_y_star[i] = u_a[i*gdim + 1];
			x_t[1]      = x_a[i*gdim + 1];
		}

		// and another :
		if (gdim == 3)
		{
			z[i]        = x_a[i*gdim + 2];
			u_z[i]      = u_a[i*gdim + 2];
			u_z_star[i] = u_a[i*gdim + 2];
			x_t[2]      = x_a[i*gdim + 2];
		}

		// create a new Point and put it in the vector x_pt :
		dolfin::Point* x_point = new dolfin::Point(3, x_t.data());
		x_pt[i]                = x_point;
	}
	printf("    - done -\n");
}

void MPMMaterial::set_initialized_by_mass(const bool val)
{
	printf("--- C++ set_initialized_by_mass(%s) ---\n", val ? "true" : "false");
	mass_init = val;
}

void MPMMaterial::initialize_mass(const std::vector<double>& m_a)
{
	printf("--- C++ initialize_mass() ---\n");
	// resize each of the vectors :
	# pragma omp parallel for simd schedule(auto)
	for (unsigned int i = 0; i < n_p; ++i)
	{
		m[i] = m_a[i];  // initalize the mass
	}
}

void MPMMaterial::initialize_volume(const std::vector<double>& V_a)
{
	printf("--- C++ initialize_volume() ---\n");
	// resize each of the vectors :
	# pragma omp parallel for simd schedule(auto)
	for (unsigned int i = 0; i < n_p; ++i)
	{
		V0[i] = V_a[i];  // initalize the initial volume
		V[i]  = V_a[i];  // initalize the current volume
	}
}

void MPMMaterial::initialize_mass_from_density(const double rho_a)
{
	printf("--- C++ initialize_mass_from_density(%g) ---\n", rho_a);
	// resize each of the vectors :
	# pragma omp parallel for simd schedule(auto)
	for (unsigned int i = 0; i < n_p; ++i)
	{
		m[i]    = rho_a * V0[i];;    // initialize the mass
		rho[i]  = rho_a;             // initialize the current denisty
		rho0[i] = rho_a;             // initialize the initial density
	}
}

void MPMMaterial::calculate_strain_rate()
{
	// calculate particle strain-rate tensor commponents :
	# pragma omp parallel for simd schedule(auto)
	for (unsigned int i = 0; i < n_p; ++i)
	{
		// we always have at least one component :
		depsilon_xx[i] = grad_u_xx[i];

		// if this is a two- or three-dimensional problem,
		// allocate space for the y componets :
		if (gdim == 2 or gdim == 3)
		{
			// two extra tensor components :
			depsilon_xy[i] = 0.5 * (grad_u_xy[i] + grad_u_yx[i]);
			depsilon_yy[i] = grad_u_yy[i];
		}

		// if this is a three-dimensional problem,
		// allocate space for the z components :
		if (gdim == 3)
		{
			// three extra tensor components :
			depsilon_xz[i] = 0.5 * (grad_u_xz[i] + grad_u_zx[i]);
			depsilon_yz[i] = 0.5 * (grad_u_yz[i] + grad_u_zy[i]);
			depsilon_zz[i] = grad_u_zz[i];
		}
	}
}

void MPMMaterial::initialize_tensors(double dt)
{
	printf("--- C++ initialize_tensors() ---\n");

	// iterate through particles :
	# pragma omp parallel for simd schedule(auto)
	for (unsigned int i = 0; i < n_p; ++i)
	{
		// we always have at least one component :
		dF_xx[i] = 1.0 + grad_u_xx[i] * dt;
		F_xx[i]  = dF_xx[i];

		// if this is a two- or three-dimensional problem, compute the y componets :
		if (gdim == 2 or gdim == 3)
		{
			// three extra tensor components :
			dF_xy[i] =       grad_u_xy[i] * dt;
			dF_yx[i] =       grad_u_yx[i] * dt;
			dF_yy[i] = 1.0 + grad_u_yy[i] * dt;
			F_xy[i]  = dF_xy[i];
			F_yx[i]  = dF_yx[i];
			F_yy[i]  = dF_yy[i];
		}

		// if this is a three-dimensional problem, compute the z componets :
		if (gdim == 3)
		{
			// five extra tensor components :
			dF_xz[i] =       grad_u_xz[i] * dt;
			dF_yz[i] =       grad_u_yz[i] * dt;
			dF_zx[i] =       grad_u_zx[i] * dt;
			dF_zy[i] =       grad_u_zy[i] * dt;
			dF_zz[i] = 1.0 + grad_u_zz[i] * dt;
			F_xz[i]  = dF_xz[i];
			F_yz[i]  = dF_yz[i];
			F_zx[i]  = dF_zx[i];
			F_zy[i]  = dF_zy[i];
			F_zz[i]  = dF_zz[i];
		}
	}
	calculate_strain_rate();
	calculate_stress();
}

void MPMMaterial::calculate_initial_volume()
{
	printf("--- C++ calculate_initial_volume() ---\n");
	// iterate through particles :
	# pragma omp parallel for simd schedule(auto)
	for (unsigned int i = 0; i < n_p; ++i)
	{
		// calculate inital volume from particle mass and density :
		V0[i] = m[i] / rho[i];
		V[i]  = V0[i];
	}
}

void MPMMaterial::update_deformation_gradient(double dt)
{
	// iterate through particles :
	# pragma omp parallel for simd schedule(auto)
	for (unsigned int i = 0; i < n_p; ++i)
	{
		// we always have at least one component :
		dF_xx[i]  = 1.0 + 0.5 * (grad_u_xx[i] + grad_u_xx_star[i]) * dt;
		F_xx[i]  *= dF_xx[i];

		// if this is a two- or three-dimensional problem, compute the y componets :
		if (gdim == 2 or gdim == 3)
		{
			// three extra tensor components :
			dF_xy[i]  =       0.5 * (grad_u_xy[i] + grad_u_xy_star[i]) * dt;
			dF_yx[i]  =       0.5 * (grad_u_yx[i] + grad_u_yx_star[i]) * dt;
			dF_yy[i]  = 1.0 + 0.5 * (grad_u_yy[i] + grad_u_yy_star[i]) * dt;
			F_xy[i]  *= dF_xy[i];
			F_yx[i]  *= dF_yx[i];
			F_yy[i]  *= dF_yy[i];
		}

		// if this is a three-dimensional problem, compute the z componets :
		if (gdim == 3)
		{
			// five extra tensor components :
			dF_xz[i]  =       0.5 * (grad_u_xz[i] + grad_u_xz_star[i]) * dt;
			dF_yz[i]  =       0.5 * (grad_u_yz[i] + grad_u_yz_star[i]) * dt;
			dF_zx[i]  =       0.5 * (grad_u_zx[i] + grad_u_zx_star[i]) * dt;
			dF_zy[i]  =       0.5 * (grad_u_zy[i] + grad_u_zy_star[i]) * dt;
			dF_zz[i]  = 1.0 + 0.5 * (grad_u_zz[i] + grad_u_zz_star[i]) * dt;
			F_xz[i]  *= dF_xz[i];
			F_yz[i]  *= dF_yz[i];
			F_zx[i]  *= dF_zx[i];
			F_zy[i]  *= dF_zy[i];
			F_zz[i]  *= dF_zz[i];
		}
	}
}

void MPMMaterial::calculate_determinant_dF()
{
	// iterate through particles :
	# pragma omp parallel for simd schedule(auto)
	for (unsigned int i = 0; i < n_p; ++i)
	{
		// first calculate the determinant :
		// one dimension :
		if (gdim == 1)
			det_dF[i] = dF_xx[i];

		// two dimensions :
		if (gdim == 2)
			det_dF[i] = dF_xx[i] * dF_yy[i] - dF_yx[i] * dF_xy[i];

		// three dimensions :
		if (gdim == 3)
			det_dF[i] = + dF_xx[i] * dF_yy[i] * dF_zz[i]
			            + dF_xy[i] * dF_yz[i] * dF_zx[i]
			            + dF_xz[i] * dF_yx[i] * dF_zy[i]
			            - dF_xz[i] * dF_yy[i] * dF_zx[i]
			            - dF_xy[i] * dF_yx[i] * dF_zz[i]
			            - dF_xx[i] * dF_yz[i] * dF_zy[i];
	}
}

void MPMMaterial::update_density()
{
	double det_dF = 0;
	// iterate through particles :
	# pragma omp parallel for simd schedule(auto)
	for (unsigned int i = 0; i < n_p; ++i)
	{
		// first calculate the determinant :
		// one dimension :
		if (gdim == 1)
			det_dF = dF_xx[i];

		// two dimensions :
		if (gdim == 2)
			det_dF = dF_xx[i] * dF_yy[i] - dF_yx[i] * dF_xy[i];

		// three dimensions :
		if (gdim == 3)
			det_dF = + dF_xx[i] * dF_yy[i] * dF_zz[i]
			         + dF_xy[i] * dF_yz[i] * dF_zx[i]
			         + dF_xz[i] * dF_yx[i] * dF_zy[i]
			         - dF_xz[i] * dF_yy[i] * dF_zx[i]
			         - dF_xy[i] * dF_yx[i] * dF_zz[i]
			         - dF_xx[i] * dF_yz[i] * dF_zy[i];

		rho[i] /= det_dF;
	}
}

void MPMMaterial::update_volume()
{
	double det_dF = 0;
	// iterate through particles :
	# pragma omp parallel for simd schedule(auto)
	for (unsigned int i = 0; i < n_p; ++i)
	{
		// first calculate the determinant :
		// one dimension :
		if (gdim == 1)
			det_dF = dF_xx[i];

		// two dimensions :
		if (gdim == 2)
			det_dF = dF_xx[i] * dF_yy[i] - dF_yx[i] * dF_xy[i];

		// three dimensions :
		if (gdim == 3)
			det_dF = + dF_xx[i] * dF_yy[i] * dF_zz[i]
			         + dF_xy[i] * dF_yz[i] * dF_zx[i]
			         + dF_xz[i] * dF_yx[i] * dF_zy[i]
			         - dF_xz[i] * dF_yy[i] * dF_zx[i]
			         - dF_xy[i] * dF_yx[i] * dF_zz[i]
			         - dF_xx[i] * dF_yz[i] * dF_zy[i];

		V[i] *= det_dF;
	}
}

void MPMMaterial::update_stress(double dt)
{
	calculate_strain_rate();  // calculate depsilon

	// calculate particle strain-rate tensor commponents :
	# pragma omp parallel for simd schedule(auto)
	for (unsigned int i = 0; i < n_p; ++i)
	{
		// we always have at least one component :
		epsilon_xx[i] += depsilon_xx[i] * dt;

		// if this is a two- or three-dimensional problem,
		// allocate space for the y componets :
		if (gdim == 2 or gdim == 3)
		{
			// two extra tensor components :
			epsilon_xy[i] += depsilon_xy[i] * dt;
			epsilon_yy[i] += depsilon_yy[i] * dt;
		}

		// if this is a three-dimensional problem,
		// allocate space for the z components :
		if (gdim == 3)
		{
			// three extra tensor components :
			epsilon_xz[i] += depsilon_xz[i] * dt;
			epsilon_yz[i] += depsilon_yz[i] * dt;
			epsilon_zz[i] += depsilon_zz[i] * dt;
		}
	}
	calculate_stress();  // calculate sigma
}

void MPMMaterial::advect_particles(double dt)
{
	// iterate through particles :
	# pragma omp parallel for simd schedule(auto)
	for (unsigned int i = 0; i < n_p; ++i)
	{
		// we always have at least one component :
		u_x[i]   += 0.5 * (a_x[i] + a_x_star[i]) * dt;
		x[i]     += u_x_star[i] * dt + 0.5 * a_x_star[i] * dt*dt;

		// if this is two- or three-dimensional :
		if (gdim == 2 or gdim == 3)
		{
			u_y[i] += 0.5 * (a_y[i] + a_y_star[i]) * dt;
			y[i]   += u_y_star[i] * dt + 0.5 * a_y_star[i] * dt*dt;
		}

		// if this is three-dimensional :
		if (gdim == 3)
		{
			u_z[i] += 0.5 * (a_z[i] + a_z_star[i]) * dt;
			z[i]   += u_z_star[i] * dt + 0.5 * a_z_star[i] * dt*dt;
		}
	}
}

static long n_stp     = (int) 1e10;  // number of discretizations
double t0, dt, pi, dx = 0.0;         // variables
int i;

void MPMMaterial::calc_pi()
{
	pi = 0.0;
	dx = 1.0 / (double) n_stp;         // the size of the step to take
	t0 = omp_get_wtime();              // start the timer

	# pragma omp parallel
	{
		double x = 0.0;                  // the part for this thread

		# pragma omp for reduction (+:pi) schedule(guided) nowait
		for (i = 0; i < n_stp; ++i)
		{
			x   = (i + 0.5) * dx;          // midpoint rule
			pi += 4.0 / (1.0 + x*x);       // increment the thread sum
		}
	}
	pi *= dx;                          // pull dx multiplication ouside sum
	dt  = omp_get_wtime() - t0;        // time to compute

	//printf("pi = %.2e \t pi - pi_true = %.2e \t dt = %.2e\n",
	//        pi,          pi - M_PI,             dt);
}



