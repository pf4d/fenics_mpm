#ifndef __MPMMODEL_H
#define __MPMMODEL_H

#include <dolfin/function/FunctionSpace.h>
#include <dolfin/geometry/BoundingBoxTree.h>
#include <dolfin/common/Array.h>
#include <dolfin/fem/GenericDofMap.h>
#include "MPMMaterial.h"
#include <math.h>

namespace dolfin
{
  class MPMModel
  {
    public:
      MPMModel(const FunctionSpace& V, 
               const unsigned int num_dofs,
               const Array<int>& coords);
      void add_material(MPMMaterial& M);
      void update_particle_basis_functions(MPMMaterial* M);
      void formulate_material_basis_functions();
      void interpolate_material_velocity_to_grid(); 
      void interpolate_material_mass_to_grid();
      void calculate_grid_volume();
      void calculate_material_density();
      void calculate_material_initial_volume();
      void calculate_material_velocity_gradient();
      
      void set_h(const Array<double>& h);

      std::vector<double>       get_m() {return m_grid;};
      
      std::vector<double>       get_U3(unsigned int index) const;
      void set_U3(unsigned int index, std::vector<double>& value);

    private:
      const unsigned int dofs = 0;
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

      // grid variables :
      std::vector<unsigned int> coord_arr = {0,0,0};
      std::vector<double> h_grid;
      std::vector<double> m_grid;
      std::vector<double> V_grid;
      std::vector<std::vector<double>> U3_grid;
      std::vector<std::vector<double>> a3_grid;
      std::vector<std::vector<double>> f_int_grid;
  };
}
#endif
