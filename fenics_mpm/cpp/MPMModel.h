#ifndef __MPMMODEL_H
#define __MPMMODEL_H

#include <dolfin/function/FunctionSpace.h>
#include <numpy/arrayobject.h>
#include <dolfin/geometry/BoundingBoxTree.h>
#include <dolfin/common/Array.h>
#include <dolfin/fem/GenericDofMap.h>
#include "MPMMaterial.h"
#include <math.h>
#include <omp.h>

namespace dolfin
{
  class MPMModel
  {
    public:
      MPMModel(const FunctionSpace& V, 
               const unsigned int num_dofs,
               const Array<int>& coords,
               double time_step,
               bool verbosity);
      void add_material(MPMMaterial& M);
      void set_boundary_conditions(const Array<int>& vertices,
                                   const Array<double>& values);
      void update_points();
      void update_particle_basis_functions(MPMMaterial* M);
      void formulate_material_basis_functions();
      void interpolate_material_velocity_to_grid(); 
      void interpolate_material_mass_to_grid();
      void calculate_grid_volume();
      void calculate_material_density();
      void calculate_material_initial_volume();
      void calculate_material_velocity_gradient();
      void initialize_material_tensors();
      void interpolate_grid_velocity_to_material();
      void interpolate_grid_acceleration_to_material();
      void update_material_volume();
      void update_material_deformation_gradient();
      void update_material_stress();
      void calculate_grid_internal_forces();
      void update_grid_velocity();
      void calculate_grid_acceleration(double m_min = 1e-2);
      void advect_material_particles();
      
      void set_h(const Array<double>& h);

      std::vector<double>       get_m() {return m_grid;};
      
      std::vector<double>       get_U3(unsigned int index) const;
      void set_U3(unsigned int index, std::vector<double>& value);
      
      std::vector<double>       get_a3(unsigned int index) const;
      void set_a3(unsigned int index, std::vector<double>& value);
      
      std::vector<double>       get_f_int(unsigned int index) const;
      void set_f_int(unsigned int index, std::vector<double>& value);

    private:
      double             dt               = 0;
      bool               verbose          = true;
      const unsigned int dofs             = 0;
      const unsigned int cell_orientation = 0;
      const unsigned int deriv_order      = 1; 
      std::vector<unsigned int> bc_vrt;
      std::vector<double>       bc_val;
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
      std::vector<unsigned int> coord_arr;
      std::vector<double> h_grid;
      std::vector<double> m_grid;
      std::vector<double> V_grid;
      std::vector<std::vector<double>> U3_grid;
      std::vector<std::vector<double>> a3_grid;
      std::vector<std::vector<double>> U3_grid_new;
      std::vector<std::vector<double>> a3_grid_new;
      std::vector<std::vector<double>> f_int_grid;
  };
}
#endif
