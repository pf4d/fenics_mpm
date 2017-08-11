#ifndef __MPMIMPENETRABLEMATERIAL_H
#define __MPMIMPENETRABLEMATERIAL_H

#include "MPMMaterial.h"

namespace dolfin
{
  class MPMImpenetrableMaterial : public MPMMaterial
  {
    public:
      MPMImpenetrableMaterial(const std::string&   name,
                              const int            n,
                              const Array<double>& x_a,
                              const Array<double>& u_a,
                              const FiniteElement& element);
     ~MPMImpenetrableMaterial() {};

      void calculate_stress() {};
      void calculate_strain_rate() {};
      void calculate_incremental_strain_rate() {};
      
      void initialize_tensors(double dt) {};
      void calculate_initial_volume() {};
      void update_deformation_gradient(double dt) {};
      void update_volume() {};
      void update_stress(double dt) {};
      void advect_particles(double dt) {};

  };
}
#endif
