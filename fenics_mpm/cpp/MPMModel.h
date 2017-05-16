#ifndef __MPMMODEL_H
#define __MPMMODEL_H

#include <dolfin/function/FunctionSpace.h>

namespace dolfin
{
  class MPMModel
  {
    public:
      MPMModel(const FunctionSpace& V);
      void eval(const Function& u);
      std::size_t num_components() {return value_size_loc;};
      const std::vector<dolfin::la_index> get_vrt() {return vrt;};
      std::vector<double> get_phi() {return phi;};
      std::vector<std::vector<double> > get_grad_phi() {return grad_grad_phi;};

    private:
      std::vector<double> phi;
      std::vector<std::vector<double> > grad_phi;
      boost::shared_ptr<const FiniteElement> element;
      Cell* cell;
      unsigned int cell_id;
      GenericDofMap* dofmap;
      const std::vector<dolfin::la_index> &vrt;
      std::size_t value_size_loc;
  };
}
#endif
