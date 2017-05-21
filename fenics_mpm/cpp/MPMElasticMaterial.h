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
                         const FiniteElement& element) :
      MPMMaterial(m_a, x_a, u_a, element) { };

      ~MPMElasticMaterial() {};

      void calculate_stress();

  };
}
#endif
