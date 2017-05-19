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
      void eval(const Array<double>& x);
      std::vector<double>       get_phi()      {return phi;};
      std::vector<unsigned int> get_vrt()      {return vrt;};
      std::vector<double>       get_grad_phi() {return grad_phi;};
      void add_material(const MPMMaterial& M);

    private:
      const unsigned int cell_orientation = 0;
      const unsigned int deriv_order = 1; 
      const FunctionSpace* Q;
      std::vector<double> phi;
      std::vector<unsigned int> vrt;
      std::vector<double> grad_phi;
      std::shared_ptr<const FiniteElement> element;
      std::unique_ptr<Cell> cell;
      unsigned int cell_id;
      std::size_t gdim;
      std::size_t sdim;
      std::shared_ptr<const dolfin::Mesh> mesh;
      std::vector<double> vertex_coordinates;
      std::vector<const MPMMaterial*> materials;
  };
}
#endif
