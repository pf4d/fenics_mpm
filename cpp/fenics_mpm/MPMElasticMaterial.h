#ifndef __MPMELASTICMATERIAL_H
#define __MPMELASTICMATERIAL_H

#include <dolfin/fem/FiniteElement.h>  // for FiniteElement
#include "MPMMaterial.h"               // for MPMMaterial

namespace fenics_mpm
{
	class MPMElasticMaterial : public MPMMaterial
	{
		public:
			MPMElasticMaterial(const std::string&           name,
			                   const int                    n,
			                   const std::vector<double>&   x_a,
			                   const std::vector<double>&   u_a,
			                   const dolfin::FiniteElement& element,
			                   double young_modulus,
			                   double poisson_ratio);
		 ~MPMElasticMaterial() {};

			void calculate_stress() override;

		private:
			double E;
			double nu;

			double mu;
			double lmbda;
			double c1;
	};
}
#endif
