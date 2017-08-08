#include "MPMImpenetrableMaterial.h"

using namespace dolfin;

MPMImpenetrableMaterial::MPMImpenetrableMaterial(const std::string&   name,
                                                 const int            n,
                                                 const Array<double>& x_a,
                                                 const Array<double>& u_a,
                                                 const FiniteElement& element) :
                    MPMMaterial(name, n, x_a, u_a, element)
{
  unsigned int idx, idx_T;
  
  for (unsigned int i = 0; i < n_p; i++)
  {
    for (unsigned int j = 0; j < gdim; j++)
    {
      for (unsigned int k = 0; k < gdim; k++)
      {
        epsilon[i][idx]  = 0.0;
        depsilon[i][idx] = 0.0;
        sigma[i][idx]    = 0.0;
      }
    }
  }
}



