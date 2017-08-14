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
      void init_vector(std::vector<double> vec);
      void update_points();
      void update_particle_basis_functions(MPMMaterial* M);
      void formulate_material_basis_functions();
      void interpolate_material_velocity_to_grid(); 
      void interpolate_material_mass_to_grid();
      void calculate_grid_volume();
      void calculate_material_initial_density();
      void calculate_material_initial_volume();
      void calculate_material_velocity_gradient();
      void initialize_material_tensors();
      void interpolate_grid_velocity_to_material();
      void interpolate_grid_acceleration_to_material();
      void update_material_density();
      void update_material_volume();
      void update_material_deformation_gradient();
      void update_material_stress();
      void calculate_grid_internal_forces();
      void update_grid_velocity();
      void calculate_grid_acceleration(double m_min = 1e-2);
      void advect_material_particles();
      void mpm(bool initialize);
      
      void set_h(const Array<double>& h_a);
      void set_V(const Array<double>& V_a);

      std::vector<double> get_m()       {return m_grid;};
      std::vector<double> get_u_x()     {return u_x_grid;};
      std::vector<double> get_u_y()     {return u_y_grid;};
      std::vector<double> get_u_z()     {return u_z_grid;};
      std::vector<double> get_a_x()     {return a_x_grid;};
      std::vector<double> get_a_y()     {return a_y_grid;};
      std::vector<double> get_a_z()     {return a_z_grid;};
      std::vector<double> get_f_int_x() {return f_int_x_grid;};
      std::vector<double> get_f_int_y() {return f_int_y_grid;};
      std::vector<double> get_f_int_z() {return f_int_z_grid;};

    private:
      const double                         dt               = 0;
      const unsigned int                   gdim             = 0;
      const unsigned int                   sdim             = 0;
      const bool                           verbose          = true;
      const unsigned int                   dofs             = 0;
      const unsigned int                   num_cells        = 0;
      const unsigned int                   cell_orientation = 0;
      const unsigned int                   deriv_order      = 1; 
      const FunctionSpace*                 Q;
      const std::shared_ptr<const FiniteElement>            element;
      const std::shared_ptr<const dolfin::Mesh>             mesh;
      const std::shared_ptr<const dolfin::BoundingBoxTree>  bbt;
      std::vector<unsigned int>            bc_vrt;
      std::vector<double>                  bc_val;
      std::vector<MPMMaterial*>            materials;
    
      // cell information for update_particle_basis_functions() :
      std::vector<std::vector<double>>     vertex_coordinates;
      std::vector<std::vector<double>>     cell_dofs;
      std::vector<Cell>                    cells;
 
      // grid variables :
      std::vector<double>       h_grid;
      std::vector<double>       m_grid;
      std::vector<double>       V_grid;

      // velocity :
      std::vector<double>       u_x_grid;
      std::vector<double>       u_y_grid;
      std::vector<double>       u_z_grid;
      std::vector<double>       u_x_grid_new;
      std::vector<double>       u_y_grid_new;
      std::vector<double>       u_z_grid_new;

      // acceleration :
      std::vector<double>       a_x_grid;
      std::vector<double>       a_y_grid;
      std::vector<double>       a_z_grid;
      std::vector<double>       a_x_grid_new;
      std::vector<double>       a_y_grid_new;
      std::vector<double>       a_z_grid_new;

      // internal forces :
      std::vector<double>       f_int_x_grid;
      std::vector<double>       f_int_y_grid;
      std::vector<double>       f_int_z_grid;
  };
}
#endif
