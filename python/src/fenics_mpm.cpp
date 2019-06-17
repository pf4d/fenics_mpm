#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <dolfin/function/FunctionSpace.h>
#include <dolfin/fem/FiniteElement.h>
#include <fenics_mpm/MPMModel.h>
#include <fenics_mpm/MPMMaterial.h>
#include <fenics_mpm/MPMElasticMaterial.h>

namespace py = pybind11;

using namespace fenics_mpm;

class PyMPMMaterial : public MPMMaterial
{
	public:
		// inherit the constructors :
		using MPMMaterial::MPMMaterial;

		// trampoline (need one for each virtual function) :
		void calculate_stress() override
		{
			PYBIND11_OVERLOAD_PURE(void, MPMMaterial, calculate_stress);
		}
};

PYBIND11_MODULE(cpp, m)
{
	py::class_<MPMModel>(m, "MPMModel")
		.def(py::init<std::shared_ptr<const dolfin::FunctionSpace>,
		              const unsigned int,
		              double,
		              bool>())
		.def("init_vector",             &MPMModel::init_vector)
		.def("add_material",            &MPMModel::add_material)
		.def("set_boundary_conditions", &MPMModel::set_boundary_conditions)
		.def("update_points",           &MPMModel::update_points)
		.def("update_particle_basis_functions", &MPMModel::update_particle_basis_functions)
		.def("formulate_material_basis_functions", &MPMModel::formulate_material_basis_functions)
		.def("interpolate_material_velocity_to_grid", &MPMModel::interpolate_material_velocity_to_grid)
		.def("interpolate_material_mass_to_grid", &MPMModel::interpolate_material_mass_to_grid)
		.def("calculate_grid_volume", &MPMModel::calculate_grid_volume)
		.def("calculate_material_initial_density", &MPMModel::calculate_material_initial_density)
		.def("calculate_material_initial_volume", &MPMModel::calculate_material_initial_volume)
		.def("calculate_material_velocity_gradient", &MPMModel::calculate_material_velocity_gradient)
		.def("initialize_material_tensors", &MPMModel::initialize_material_tensors)
		.def("interpolate_grid_velocity_to_material", &MPMModel::interpolate_grid_velocity_to_material)
		.def("interpolate_grid_acceleration_to_material", &MPMModel::interpolate_grid_acceleration_to_material)
		.def("update_material_density", &MPMModel::update_material_density)
		.def("update_material_volume", &MPMModel::update_material_volume)
		.def("update_material_deformation_gradient", &MPMModel::update_material_deformation_gradient)
		.def("update_material_stress", &MPMModel::update_material_stress)
		.def("calculate_grid_internal_forces", &MPMModel::calculate_grid_internal_forces)
		.def("update_grid_velocity", &MPMModel::update_grid_velocity)
		.def("calculate_grid_acceleration", &MPMModel::calculate_grid_acceleration)
		.def("advect_material_particles", &MPMModel::advect_material_particles)
		.def("mpm", &MPMModel::mpm)
		.def("set_h", &MPMModel::set_h)
		.def("set_V", &MPMModel::set_V)
		.def("get_m", &MPMModel::get_m)
		.def("get_u_x", &MPMModel::get_u_x)
		.def("get_u_y", &MPMModel::get_u_y)
		.def("get_u_z", &MPMModel::get_u_z)
		.def("get_a_x", &MPMModel::get_a_x)
		.def("get_a_y", &MPMModel::get_a_y)
		.def("get_a_z", &MPMModel::get_a_z)
		.def("get_f_int_x", &MPMModel::get_f_int_x)
		.def("get_f_int_y", &MPMModel::get_f_int_y)
		.def("get_f_int_z", &MPMModel::get_f_int_z)
		.def("__repr__",
			[](const MPMModel &a)
			{
				return "<MPMModel>";
			});

	py::class_<MPMMaterial, PyMPMMaterial>mpmmaterial(m, "MPMMaterial");
		mpmmaterial
		.def(py::init<const std::string &,
		              const int,
		              const std::vector<double> &,
		              const std::vector<double> &,
		              const dolfin::FiniteElement &>())
		.def("calculate_stress", & MPMMaterial::calculate_stress);

	py::class_<MPMElasticMaterial>(m, "MPMElasticMaterial", mpmmaterial)
		.def(py::init<const std::string &,
		              const int,
		              const std::vector<double> &,
		              const std::vector<double> &,
		              const dolfin::FiniteElement &,
		              double,
		              double>())
		.def("get_mass_init", &MPMElasticMaterial::get_mass_init)
		.def("get_name", &MPMElasticMaterial::get_name)
		.def("get_m", &MPMElasticMaterial::get_m)
		.def("get_x_pt", &MPMElasticMaterial::get_x_pt)
		.def("get_rho0", &MPMElasticMaterial::get_rho0)
		.def("get_rho", &MPMElasticMaterial::get_rho)
		.def("get_V0", &MPMElasticMaterial::get_V0)
		.def("get_V", &MPMElasticMaterial::get_V)
		.def("get_det_dF", &MPMElasticMaterial::get_det_dF)
		.def("get_x", &MPMElasticMaterial::get_x)
		.def("get_y", &MPMElasticMaterial::get_y)
		.def("get_z", &MPMElasticMaterial::get_z)
		.def("get_u_x", &MPMElasticMaterial::get_u_x)
		.def("get_u_y", &MPMElasticMaterial::get_u_y)
		.def("get_u_z", &MPMElasticMaterial::get_u_z)
		.def("get_a_x", &MPMElasticMaterial::get_a_x)
		.def("get_a_y", &MPMElasticMaterial::get_a_y)
		.def("get_a_z", &MPMElasticMaterial::get_a_z)
		.def("get_u_x_star", &MPMElasticMaterial::get_u_x_star)
		.def("get_u_y_star", &MPMElasticMaterial::get_u_y_star)
		.def("get_u_z_star", &MPMElasticMaterial::get_u_z_star)
		.def("get_a_x_star", &MPMElasticMaterial::get_a_x_star)
		.def("get_a_y_star", &MPMElasticMaterial::get_a_y_star)
		.def("get_a_z_star", &MPMElasticMaterial::get_a_z_star)
		.def("get_grad_u_xx", &MPMElasticMaterial::get_grad_u_xx)
		.def("get_grad_u_xy", &MPMElasticMaterial::get_grad_u_xy)
		.def("get_grad_u_xz", &MPMElasticMaterial::get_grad_u_xz)
		.def("get_grad_u_yx", &MPMElasticMaterial::get_grad_u_yx)
		.def("get_grad_u_yy", &MPMElasticMaterial::get_grad_u_yy)
		.def("get_grad_u_yz", &MPMElasticMaterial::get_grad_u_yz)
		.def("get_grad_u_zx", &MPMElasticMaterial::get_grad_u_zx)
		.def("get_grad_u_zy", &MPMElasticMaterial::get_grad_u_zy)
		.def("get_grad_u_zz", &MPMElasticMaterial::get_grad_u_zz)
		.def("get_grad_u_xx_star", &MPMElasticMaterial::get_grad_u_xx_star)
		.def("get_grad_u_xy_star", &MPMElasticMaterial::get_grad_u_xy_star)
		.def("get_grad_u_xz_star", &MPMElasticMaterial::get_grad_u_xz_star)
		.def("get_grad_u_yx_star", &MPMElasticMaterial::get_grad_u_yx_star)
		.def("get_grad_u_yy_star", &MPMElasticMaterial::get_grad_u_yy_star)
		.def("get_grad_u_yz_star", &MPMElasticMaterial::get_grad_u_yz_star)
		.def("get_grad_u_zx_star", &MPMElasticMaterial::get_grad_u_zx_star)
		.def("get_grad_u_zy_star", &MPMElasticMaterial::get_grad_u_zy_star)
		.def("get_grad_u_zz_star", &MPMElasticMaterial::get_grad_u_zz_star)
		.def("get_dF_xx", &MPMElasticMaterial::get_dF_xx)
		.def("get_dF_xy", &MPMElasticMaterial::get_dF_xy)
		.def("get_dF_xz", &MPMElasticMaterial::get_dF_xz)
		.def("get_dF_yx", &MPMElasticMaterial::get_dF_yx)
		.def("get_dF_yy", &MPMElasticMaterial::get_dF_yy)
		.def("get_dF_yz", &MPMElasticMaterial::get_dF_yz)
		.def("get_dF_zx", &MPMElasticMaterial::get_dF_zx)
		.def("get_dF_zy", &MPMElasticMaterial::get_dF_zy)
		.def("get_dF_zz", &MPMElasticMaterial::get_dF_zz)
		.def("get_F_xx", &MPMElasticMaterial::get_F_xx)
		.def("get_F_xy", &MPMElasticMaterial::get_F_xy)
		.def("get_F_xz", &MPMElasticMaterial::get_F_xz)
		.def("get_F_yx", &MPMElasticMaterial::get_F_yx)
		.def("get_F_yy", &MPMElasticMaterial::get_F_yy)
		.def("get_F_yz", &MPMElasticMaterial::get_F_yz)
		.def("get_F_zx", &MPMElasticMaterial::get_F_zx)
		.def("get_F_zy", &MPMElasticMaterial::get_F_zy)
		.def("get_F_zz", &MPMElasticMaterial::get_F_zz)
		.def("get_sigma_xx", &MPMElasticMaterial::get_sigma_xx)
		.def("get_sigma_xy", &MPMElasticMaterial::get_sigma_xy)
		.def("get_sigma_xz", &MPMElasticMaterial::get_sigma_xz)
		.def("get_sigma_yy", &MPMElasticMaterial::get_sigma_yy)
		.def("get_sigma_yz", &MPMElasticMaterial::get_sigma_yz)
		.def("get_sigma_zz", &MPMElasticMaterial::get_sigma_zz)
		.def("get_epsilon_xx", &MPMElasticMaterial::get_epsilon_xx)
		.def("get_epsilon_xy", &MPMElasticMaterial::get_epsilon_xy)
		.def("get_epsilon_xz", &MPMElasticMaterial::get_epsilon_xz)
		.def("get_epsilon_yy", &MPMElasticMaterial::get_epsilon_yy)
		.def("get_epsilon_yz", &MPMElasticMaterial::get_epsilon_yz)
		.def("get_epsilon_zz", &MPMElasticMaterial::get_epsilon_zz)
		.def("get_depsilon_xx", &MPMElasticMaterial::get_depsilon_xx)
		.def("get_depsilon_xy", &MPMElasticMaterial::get_depsilon_xy)
		.def("get_depsilon_xz", &MPMElasticMaterial::get_depsilon_xz)
		.def("get_depsilon_yy", &MPMElasticMaterial::get_depsilon_yy)
		.def("get_depsilon_yz", &MPMElasticMaterial::get_depsilon_yz)
		.def("get_depsilon_zz", &MPMElasticMaterial::get_depsilon_zz)
		.def("get_vrt_1", &MPMElasticMaterial::get_vrt_1)
		.def("get_vrt_2", &MPMElasticMaterial::get_vrt_2)
		.def("get_vrt_3", &MPMElasticMaterial::get_vrt_3)
		.def("get_vrt_4", &MPMElasticMaterial::get_vrt_4)
		.def("get_phi_1", &MPMElasticMaterial::get_phi_1)
		.def("get_phi_2", &MPMElasticMaterial::get_phi_2)
		.def("get_phi_3", &MPMElasticMaterial::get_phi_3)
		.def("get_phi_4", &MPMElasticMaterial::get_phi_4)
		.def("get_grad_phi_1x", &MPMElasticMaterial::get_grad_phi_1x)
		.def("get_grad_phi_1y", &MPMElasticMaterial::get_grad_phi_1y)
		.def("get_grad_phi_1z", &MPMElasticMaterial::get_grad_phi_1z)
		.def("get_grad_phi_2x", &MPMElasticMaterial::get_grad_phi_2x)
		.def("get_grad_phi_2y", &MPMElasticMaterial::get_grad_phi_2y)
		.def("get_grad_phi_2z", &MPMElasticMaterial::get_grad_phi_2z)
		.def("get_grad_phi_3x", &MPMElasticMaterial::get_grad_phi_3x)
		.def("get_grad_phi_3y", &MPMElasticMaterial::get_grad_phi_3y)
		.def("get_grad_phi_3z", &MPMElasticMaterial::get_grad_phi_3z)
		.def("get_grad_phi_4x", &MPMElasticMaterial::get_grad_phi_4x)
		.def("get_grad_phi_4y", &MPMElasticMaterial::get_grad_phi_4y)
		.def("get_grad_phi_4z", &MPMElasticMaterial::get_grad_phi_4z)
		.def("set_initialized_by_mass", &MPMElasticMaterial::set_initialized_by_mass)
		.def("initialize_mass", &MPMElasticMaterial::initialize_mass)
		.def("initialize_volume", &MPMElasticMaterial::initialize_volume)
		.def("initialize_mass_from_density", &MPMElasticMaterial::initialize_mass_from_density)
		.def("get_num_particles", &MPMElasticMaterial::get_num_particles)
		.def("calculate_strain_rate", &MPMElasticMaterial::calculate_strain_rate)
		.def("calculate_stress", &MPMElasticMaterial::calculate_stress)
		.def("initialize_tensors", &MPMElasticMaterial::initialize_tensors)
		.def("calculate_initial_volume", &MPMElasticMaterial::calculate_initial_volume)
		.def("calculate_determinant_dF", &MPMElasticMaterial::calculate_determinant_dF)
		.def("update_deformation_gradient", &MPMElasticMaterial::update_deformation_gradient)
		.def("update_density", &MPMElasticMaterial::update_density)
		.def("update_volume", &MPMElasticMaterial::update_volume)
		.def("update_stress", &MPMElasticMaterial::update_stress)
		.def("advect_particles", &MPMElasticMaterial::advect_particles)
		.def("calc_pi", &MPMElasticMaterial::calc_pi);
}



