#include "MPMMaterial.h"

using namespace dolfin;

MPMMaterial::MPMMaterial(Array<double>& m,
                         Array<double>& x,
                         Array<double>& u,
                         const unsigned int topological_dimension,
                         const unsigned int element_dimension)
            : tDim(topological_dimension), eDim(element_dimension)
{
  n_p = m.size();
  printf("::: created material with n_p = %u \t" +
                                    "tDim = %u \t" +
                                    "eDim = %u :::\n", n_p, tDim, eDim);

  // one scalar or vector for each particle :
  m.resize(n_p);
  rho.resize(n_p);
  V0.resize(n_p);
  V.resize(n_p);
  x.resize(n_p);
  u.resize(n_p);
  u_star.resize(n_p);
  a.resize(n_p);
  grad_u.resize(n_p);
  vrt.resize(n_p);
  phi.resize(n_p);
  grad_phi.resize(n_p);
  F.resize(n_p);
  sigma.resize(n_p);
  epsilon.resize(n_p);
 
  // the flattened identity tensor :
  I[i].resize(tDim*eDim);

  // resize each of the vectors :
  for (unsigned int i = 0; i < n_p; i++)
  {
    // these are vectors in topological dimension :
    x[i].resize(tDim);
    u[i].resize(tDim);
    u_star[i].resize(tDim);
    a[i].resize(tDim);
    
    // these are vectors in element dimension :
    vrt[i].resize(eDim);
    phi[i].resize(eDim);

    // these are flattened tensors defined over each 
    //   topological and element dimension :
    grad_u[i].resize(tDim*eDim);
    grad_phi[i].resize(tDim*eDim);
    F[i].resize(tDim*eDim);
    sigma[i].resize(tDim*eDim);
    epsilon[i].resize(tDim*eDim);
  }
}

void MPMMaterial::update_strain_rate()
{
  // calculate particle strain-rate tensors :
  for (unsigned int i = 0; i < n_p; i++)
  {
    for (unsigned int j = 0; j < tDim; j++)
    {
      if (i == j)
        epsilon[i][j] = grad_u[i][j];
      else if (j > i)
      {
        eps_temp      = 0.5 * (grad_u[i][j] + grad_u[j][i]);
        epsilon[i][j] = eps_temp;
        epsilon[j][i] = eps_temp;
      }
    }
  }
}



