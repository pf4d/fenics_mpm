#include "MPMElasticMaterial.h"

using namespace dolfin;

void MPMElasticMaterial::calculate_stress()
{
  // calculate particle strain-rate tensors :
  for (unsigned int i = 0; i < n_p; i++)
  {
    for (unsigned int j = 0; j < gdim; j++)
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



