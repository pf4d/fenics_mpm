#ifndef __MPMIMPENETRABLEMATERIAL_H
#define __MPMIMPENETRABLEMATERIAL_H

#include "MPMMaterial.h"

namespace dolfin
{
  class MPMImpenetrableMaterial : public MPMMaterial
  {
    public:
      MPMImpenetrableMaterial(const Array<double>& m_a,
                              const Array<double>& x_a,
                              const Array<double>& u_a,
                              const FiniteElement& element);
     ~MPMImpenetrableMaterial() {};

      virtual void calculate_stress() {};
      virtual void calculate_strain_rate() {};
      virtual void calculate_incremental_strain_rate() {};
      
      virtual void initialize_tensors(double dt) {};
      virtual void calculate_initial_volume() {};
      virtual void update_deformation_gradient(double dt) {};
      virtual void update_volume() {};
      virtual void update_stress(double dt) {};
      virtual void advect_particles(double dt) {};

  };
}
#endif
