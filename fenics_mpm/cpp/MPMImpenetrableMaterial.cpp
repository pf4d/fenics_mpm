#include "MPMImpenetrableMaterial.h"

using namespace dolfin;

MPMImpenetrableMaterial::MPMImpenetrableMaterial(const std::string&   name,
                                                 const int            n,
                                                 const Array<double>& x_a,
                                                 const Array<double>& u_a,
                                                 const FiniteElement& element) :
                    MPMMaterial(name, n, x_a, u_a, element)
{
  # pragma omp parallel for schedule(auto)
  for (unsigned int i = 0; i < n_p; ++i)
  {
    // we always have at least one component :
    epsilon_xx[i]  = 0.0;
    depsilon_xx[i] = 0.0;
    sigma_xx[i]    = 0.0;
  
    // if this is a two- or three-dimensional problem, 
    // allocate space for the y componets :
    if (gdim == 2 or gdim == 3)
    {
      // two extra tensor components :
      epsilon_xy[i]  = 0.0;
      epsilon_yy[i]  = 0.0;
      depsilon_xy[i] = 0.0;
      depsilon_yy[i] = 0.0;
      sigma_xy[i]    = 0.0;
      sigma_yy[i]    = 0.0;
    }

    // if this is a three-dimensional problem, 
    // allocate space for the z components :
    if (gdim == 3)
    {
      // three extra tensor components :
      epsilon_xz[i]  = 0.0;
      epsilon_yz[i]  = 0.0;
      epsilon_zz[i]  = 0.0;
      depsilon_xz[i] = 0.0;
      depsilon_yz[i] = 0.0;
      depsilon_zz[i] = 0.0;
      sigma_xz[i]    = 0.0;
      sigma_yz[i]    = 0.0;
      sigma_zz[i]    = 0.0;
    }
  }
}



