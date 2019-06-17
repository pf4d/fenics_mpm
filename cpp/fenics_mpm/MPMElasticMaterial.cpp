#include "MPMElasticMaterial.h"

using namespace fenics_mpm;

/*
trace_eps  = epsilon_p[0,0] + epsilon_p[1,1]
c1         = 2.0 * self.mu
c2         = self.lmbda * trace_eps
sig_xx     = c1 * epsilon_p[0,0] + c2
sig_xy     = c1 * epsilon_p[0,1]
sig_yy     = c1 * epsilon_p[1,1] + c2
sigma_p    = np.array( [[sig_xx, sig_xy], [sig_xy, sig_yy]], dtype=float )
sigma.append(sigma_p)
*/

MPMElasticMaterial::MPMElasticMaterial(const std::string&           name,
                                       const int                    n,
                                       const std::vector<double>&   x_a,
                                       const std::vector<double>&   u_a,
                                       const dolfin::FiniteElement& element,
                                       const double young_modulus,
                                       const double poisson_ratio) :
                    MPMMaterial(name, n, x_a, u_a, element),
                    E(young_modulus),
                    nu(poisson_ratio)
{
	// Lamé parameters :
	mu       = E / (2.0*(1.0 + nu));
	lmbda    = E*nu / ((1.0 + nu)*(1.0 - 2.0*nu));
	c1       = 2.0 * mu;
}

void MPMElasticMaterial::calculate_stress()
{
	double c2, trace_eps;

	// calculate particle stress tensors :
	# pragma omp parallel for schedule(auto)
	for (unsigned int i = 0; i < n_p; i++)
	{

		// there is always one component of strain :
		trace_eps = epsilon_xx[i];

		// two or three dimensions there is one more :
		if (gdim == 2 or gdim == 3)
			trace_eps += epsilon_yy[i];

		// three dimensions there is yet one more :
		if (gdim == 3)
			trace_eps += epsilon_zz[i];

		c2 = lmbda * trace_eps;  // the coefficient

		// update the stress tensor :
		sigma_xx[i] = c1 * epsilon_xx[i] + c2;

		// two or three dimensions there are two more components :
		if (gdim == 2 or gdim == 3)
		{
			sigma_xy[i] = c1 * epsilon_xy[i];
			sigma_yy[i] = c1 * epsilon_yy[i] + c2;
		}

		// three dimensions there are yet three more :
		if (gdim == 3)
		{
			sigma_xz[i] = c1 * epsilon_xz[i];
			sigma_yz[i] = c1 * epsilon_yz[i];
			sigma_zz[i] = c1 * epsilon_zz[i] + c2;
		}
	}
}



