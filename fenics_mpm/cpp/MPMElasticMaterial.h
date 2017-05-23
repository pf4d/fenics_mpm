#ifndef __MPMELASTICMATERIAL_H
#define __MPMELASTICMATERIAL_H

#include "MPMMaterial.h"

namespace dolfin
{
  class MPMElasticMaterial : public MPMMaterial
  {
    public:
      MPMElasticMaterial(const Array<double>& m_a,
                         const Array<double>& x_a,
                         const Array<double>& u_a,
                         const FiniteElement& element,
                         double young_modulus,
                         double poisson_ratio);
     ~MPMElasticMaterial() {};

      void calculate_stress();

    private:
      double E;
      double nu;

      double mu;
      double lmbda;
  };
}
#endif
