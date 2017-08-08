#include "MPMElasticMaterial.h"

using namespace dolfin;

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

MPMElasticMaterial::MPMElasticMaterial(const std::string&   name,
                                       const int            n,
                                       const Array<double>& x_a,
                                       const Array<double>& u_a,
                                       const FiniteElement& element,
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
  unsigned int idx;
  double       c2, trace_eps;

  // calculate particle strain-rate tensors :
  for (unsigned int i = 0; i < n_p; i++)
  {
    // calculate the trace of epsilon :
    trace_eps = 0;
    for (unsigned int p = 0; p < gdim; p++)
      trace_eps += epsilon[i][p*gdim + p];
    c2 = lmbda * trace_eps;

    // update the stress tensor :
    for (unsigned int j = 0; j < gdim; j++)
    {
      for (unsigned int k = 0; k < gdim; k++)
      {
        idx   = j*gdim + k;
        if (k == j)
          sigma[i][idx] = c1 * epsilon[i][idx] + c2;
        else
          sigma[i][idx] = c1 * epsilon[i][idx];
      }
    }
  }
}


