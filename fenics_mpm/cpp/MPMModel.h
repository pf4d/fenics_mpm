#ifndef __MPMMODEL_H
#define __MPMMODEL_H

#include <dolfin/function/FunctionSpace.h>
#include <dolfin/geometry/BoundingBoxTree.h>
#include <dolfin/fem/GenericDofMap.h>
#include "MPMMaterial.h"

namespace dolfin
{
  class MPMModel
  {
    public:
      MPMModel(const FunctionSpace& V);
      void add_material(MPMMaterial& M);
      void update_particle_basis_functions(MPMMaterial* M);
      void formulate_material_basis_functions();

    private:
      const unsigned int cell_orientation = 0;
      const unsigned int deriv_order = 1; 
      const FunctionSpace* Q;
      std::shared_ptr<const FiniteElement> element;
      std::unique_ptr<Cell> cell;
      unsigned int cell_id;
      std::size_t gdim;
      std::size_t sdim;
      std::shared_ptr<const dolfin::Mesh> mesh;
      std::vector<double> vertex_coordinates;
      std::vector<MPMMaterial*> materials;
  };
}
#endif
