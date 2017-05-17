#ifndef __MPMMODEL_H
#define __MPMMODEL_H

#include <dolfin/function/FunctionSpace.h>
#include <dolfin/geometry/BoundingBoxTree.h>
#include <dolfin/fem/GenericDofMap.h>

namespace dolfin
{
  class MPMModel
  {
    public:
      MPMModel(const FunctionSpace& V, const unsigned int N_p);
      void eval(const Array<double>& x);
      std::vector<std::vector<double>>       get_phi()      {return phi;};
      std::vector<std::vector<unsigned int>> get_vrt()      {return vrt;};
      std::vector<std::vector<double>>       get_grad_phi() {return grad_phi;};

    private:
      const unsigned int n_p;
      const unsigned int cell_orientation = 0;
      const unsigned int deriv_order = 1; 
      const FunctionSpace* Q;
      std::vector<std::vector<double>>       phi;
      std::vector<std::vector<unsigned int>> vrt;
      std::vector<std::vector<double>>       grad_phi;
      std::shared_ptr<const FiniteElement> element;
      std::unique_ptr<Cell> cell;
      unsigned int cell_id;
      std::size_t gdim;
      std::size_t sdim;
      std::shared_ptr<const dolfin::Mesh> mesh;
      std::vector<double> vertex_coordinates;
  };
}
#endif
