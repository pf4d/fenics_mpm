#ifndef __MPMIMPENETRABLEMATERIAL_H
#define __MPMIMPENETRABLEMATERIAL_H

#include <dolfin/fem/FiniteElement.h>  // for FiniteElement
#include "MPMMaterial.h"               // for MPMMaterial

namespace fenics_mpm
{
	class MPMImpenetrableMaterial : public MPMMaterial
	{
		public:
			MPMImpenetrableMaterial(const std::string&           name,
			                        const int                    n,
			                        const std::vector<double>&   x_a,
			                        const std::vector<double>&   u_a,
			                        const dolfin::FiniteElement& element);
		 ~MPMImpenetrableMaterial() {};

			void calculate_stress() override {};
			void calculate_strain_rate() {};
			void calculate_incremental_strain_rate() {};

			void initialize_tensors(double) {};
			void calculate_initial_volume() {};
			void update_deformation_gradient(double) {};
			void update_volume() {};
			void update_stress(double) {};
			void advect_particles(double) {};

	};
}
#endif
