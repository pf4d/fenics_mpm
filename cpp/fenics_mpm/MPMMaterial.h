#ifndef __MPMMATERIAL_H
#define __MPMMATERIAL_H

#include <dolfin/geometry/Point.h>
#include <dolfin/fem/FiniteElement.h>
#include <vector>
#include <omp.h>
#include <math.h>

using namespace std;

inline unsigned int min( int a, int b ) { return a < b ? a : b; }
inline unsigned int max( int a, int b ) { return a > b ? a : b; }

namespace fenics_mpm
{
	class MPMMaterial
	{
		public:
			virtual ~MPMMaterial() = 0;
			MPMMaterial(const string&   name,
			            const int            n,
			            const vector<double>& x_a,
			            const vector<double>& u_a,
			            const dolfin::FiniteElement& element);

			bool                    get_mass_init()      {return mass_init;};
			const char *            get_name()           {return name.c_str();};
			vector<double>&         get_m()              {return m;};
			vector<dolfin::Point*>& get_x_pt()           {return x_pt;};
			vector<double>&         get_rho0()           {return rho0;};
			vector<double>&         get_rho()            {return rho;};
			vector<double>&         get_V0()             {return V0;};
			vector<double>&         get_V()              {return V;};
			vector<double>&         get_det_dF()         {return det_dF;};
			vector<double>&         get_x()              {return x;};
			vector<double>&         get_y()              {return y;};
			vector<double>&         get_z()              {return z;};
			vector<double>&         get_u_x()            {return u_x;};
			vector<double>&         get_u_y()            {return u_y;};
			vector<double>&         get_u_z()            {return u_z;};
			vector<double>&         get_a_x()            {return a_x;};
			vector<double>&         get_a_y()            {return a_y;};
			vector<double>&         get_a_z()            {return a_z;};
			vector<double>&         get_u_x_star()       {return u_x_star;};
			vector<double>&         get_u_y_star()       {return u_y_star;};
			vector<double>&         get_u_z_star()       {return u_z_star;};
			vector<double>&         get_a_x_star()       {return a_x_star;};
			vector<double>&         get_a_y_star()       {return a_y_star;};
			vector<double>&         get_a_z_star()       {return a_z_star;};
			vector<double>&         get_grad_u_xx()      {return grad_u_xx;};
			vector<double>&         get_grad_u_xy()      {return grad_u_xy;};
			vector<double>&         get_grad_u_xz()      {return grad_u_xz;};
			vector<double>&         get_grad_u_yx()      {return grad_u_yx;};
			vector<double>&         get_grad_u_yy()      {return grad_u_yy;};
			vector<double>&         get_grad_u_yz()      {return grad_u_yz;};
			vector<double>&         get_grad_u_zx()      {return grad_u_zx;};
			vector<double>&         get_grad_u_zy()      {return grad_u_zy;};
			vector<double>&         get_grad_u_zz()      {return grad_u_zz;};
			vector<double>&         get_grad_u_xx_star() {return grad_u_xx_star;};
			vector<double>&         get_grad_u_xy_star() {return grad_u_xy_star;};
			vector<double>&         get_grad_u_xz_star() {return grad_u_xz_star;};
			vector<double>&         get_grad_u_yx_star() {return grad_u_yx_star;};
			vector<double>&         get_grad_u_yy_star() {return grad_u_yy_star;};
			vector<double>&         get_grad_u_yz_star() {return grad_u_yz_star;};
			vector<double>&         get_grad_u_zx_star() {return grad_u_zx_star;};
			vector<double>&         get_grad_u_zy_star() {return grad_u_zy_star;};
			vector<double>&         get_grad_u_zz_star() {return grad_u_zz_star;};
			vector<double>&         get_dF_xx()          {return dF_xx;};
			vector<double>&         get_dF_xy()          {return dF_xy;};
			vector<double>&         get_dF_xz()          {return dF_xz;};
			vector<double>&         get_dF_yx()          {return dF_yx;};
			vector<double>&         get_dF_yy()          {return dF_yy;};
			vector<double>&         get_dF_yz()          {return dF_yz;};
			vector<double>&         get_dF_zx()          {return dF_zx;};
			vector<double>&         get_dF_zy()          {return dF_zy;};
			vector<double>&         get_dF_zz()          {return dF_zz;};
			vector<double>&         get_F_xx()           {return F_xx;};
			vector<double>&         get_F_xy()           {return F_xy;};
			vector<double>&         get_F_xz()           {return F_xz;};
			vector<double>&         get_F_yx()           {return F_yx;};
			vector<double>&         get_F_yy()           {return F_yy;};
			vector<double>&         get_F_yz()           {return F_yz;};
			vector<double>&         get_F_zx()           {return F_zx;};
			vector<double>&         get_F_zy()           {return F_zy;};
			vector<double>&         get_F_zz()           {return F_zz;};
			vector<double>&         get_sigma_xx()       {return sigma_xx;};
			vector<double>&         get_sigma_xy()       {return sigma_xy;};
			vector<double>&         get_sigma_xz()       {return sigma_xz;};
			vector<double>&         get_sigma_yy()       {return sigma_yy;};
			vector<double>&         get_sigma_yz()       {return sigma_yz;};
			vector<double>&         get_sigma_zz()       {return sigma_zz;};
			vector<double>&         get_epsilon_xx()     {return epsilon_xx;};
			vector<double>&         get_epsilon_xy()     {return epsilon_xy;};
			vector<double>&         get_epsilon_xz()     {return epsilon_xz;};
			vector<double>&         get_epsilon_yy()     {return epsilon_yy;};
			vector<double>&         get_epsilon_yz()     {return epsilon_yz;};
			vector<double>&         get_epsilon_zz()     {return epsilon_zz;};
			vector<double>&         get_depsilon_xx()    {return depsilon_xx;};
			vector<double>&         get_depsilon_xy()    {return depsilon_xy;};
			vector<double>&         get_depsilon_xz()    {return depsilon_xz;};
			vector<double>&         get_depsilon_yy()    {return depsilon_yy;};
			vector<double>&         get_depsilon_yz()    {return depsilon_yz;};
			vector<double>&         get_depsilon_zz()    {return depsilon_zz;};
			vector<unsigned int>&   get_vrt_1()          {return vrt_1;};
			vector<unsigned int>&   get_vrt_2()          {return vrt_2;};
			vector<unsigned int>&   get_vrt_3()          {return vrt_3;};
			vector<unsigned int>&   get_vrt_4()          {return vrt_4;};
			vector<double>&         get_phi_1()          {return phi_1;};
			vector<double>&         get_phi_2()          {return phi_2;};
			vector<double>&         get_phi_3()          {return phi_3;};
			vector<double>&         get_phi_4()          {return phi_4;};
			vector<double>&         get_grad_phi_1x()    {return grad_phi_1x;};
			vector<double>&         get_grad_phi_1y()    {return grad_phi_1y;};
			vector<double>&         get_grad_phi_1z()    {return grad_phi_1z;};
			vector<double>&         get_grad_phi_2x()    {return grad_phi_2x;};
			vector<double>&         get_grad_phi_2y()    {return grad_phi_2y;};
			vector<double>&         get_grad_phi_2z()    {return grad_phi_2z;};
			vector<double>&         get_grad_phi_3x()    {return grad_phi_3x;};
			vector<double>&         get_grad_phi_3y()    {return grad_phi_3y;};
			vector<double>&         get_grad_phi_3z()    {return grad_phi_3z;};
			vector<double>&         get_grad_phi_4x()    {return grad_phi_4x;};
			vector<double>&         get_grad_phi_4y()    {return grad_phi_4y;};
			vector<double>&         get_grad_phi_4z()    {return grad_phi_4z;};

			void         set_initialized_by_mass(const bool val);
			void         initialize_mass(const vector<double>& m_a);
			void         initialize_volume(const vector<double>& V_a);
			void         initialize_mass_from_density(const double rho_a);
			unsigned int get_num_particles() const {return n_p;};
			void         calculate_strain_rate();
			virtual void calculate_stress() = 0;

			void         initialize_tensors(double dt);
			void         calculate_initial_volume();
			void         calculate_determinant_dF();
			void         update_deformation_gradient(double dt);
			void         update_density();
			void         update_volume();
			void         update_stress(double dt);
			void         advect_particles(double dt);
			void         calc_pi();

		protected:
			unsigned int           n_p_end;
			string                 name;        // name of material
			bool                   mass_init;   // initialized via mass
			unsigned int           n_p;         // number of particles
			const unsigned int     gdim;        // topological dimension
			const unsigned int     sdim;        // element dimension
			vector<double>         m;           // mass vector
			vector<double>         rho0;        // initial density vector
			vector<double>         rho;         // density vector
			vector<double>         V0;          // initial volume vector
			vector<double>         V;           // volume vector
			vector<double>         det_dF;      // det. of incremental def. grad.
			vector<dolfin::Point*> x_pt;        // (x,y,z) position Points
			vector<double>         x;           // x-position vector
			vector<double>         y;           // y-position vector
			vector<double>         z;           // z-position vector
			vector<double>         u_x;         // x-component of velocity vector
			vector<double>         u_y;         // y-component of velocity vector
			vector<double>         u_z;         // z-component of velocity vector
			vector<double>         u_x_star;    // temporary x-velocity interp.
			vector<double>         u_y_star;    // temporary y-velocity interp.
			vector<double>         u_z_star;    // temporary z-velocity interp.
			vector<double>         a_x_star;    // temporary x-accel. interp.
			vector<double>         a_y_star;    // temporary y-accel. interp.
			vector<double>         a_z_star;    // temporary z-accel. interp.
			vector<double>         a_x;         // x-acceleration vector
			vector<double>         a_y;         // y-acceleration vector
			vector<double>         a_z;         // z-acceleration vector
			vector<double>         grad_u_xx;   // velocity grad. tensor xx
			vector<double>         grad_u_xy;   // velocity grad. tensor xy
			vector<double>         grad_u_xz;   // velocity grad. tensor xz
			vector<double>         grad_u_yx;   // velocity grad. tensor yx
			vector<double>         grad_u_yy;   // velocity grad. tensor yy
			vector<double>         grad_u_yz;   // velocity grad. tensor yz
			vector<double>         grad_u_zx;   // velocity grad. tensor zx
			vector<double>         grad_u_zy;   // velocity grad. tensor zy
			vector<double>         grad_u_zz;   // velocity grad. tensor zz
			vector<double>         grad_u_xx_star; // velocity grad. tensor xx
			vector<double>         grad_u_xy_star; // velocity grad. tensor xy
			vector<double>         grad_u_xz_star; // velocity grad. tensor xz
			vector<double>         grad_u_yx_star; // velocity grad. tensor yx
			vector<double>         grad_u_yy_star; // velocity grad. tensor yy
			vector<double>         grad_u_yz_star; // velocity grad. tensor yz
			vector<double>         grad_u_zx_star; // velocity grad. tensor zx
			vector<double>         grad_u_zy_star; // velocity grad. tensor zy
			vector<double>         grad_u_zz_star; // velocity grad. tensor zz
			vector<double>         F_xx;           // deformation grad. tensor xx
			vector<double>         F_xy;           // deformation grad. tensor xy
			vector<double>         F_xz;           // deformation grad. tensor xz
			vector<double>         F_yx;           // deformation grad. tensor yx
			vector<double>         F_yy;           // deformation grad. tensor yy
			vector<double>         F_yz;           // deformation grad. tensor yz
			vector<double>         F_zx;           // deformation grad. tensor zx
			vector<double>         F_zy;           // deformation grad. tensor zy
			vector<double>         F_zz;           // deformation grad. tensor zz
			vector<double>         dF_xx;          // inc. def. grad. tensor xx
			vector<double>         dF_xy;          // inc. def. grad. tensor xy
			vector<double>         dF_xz;          // inc. def. grad. tensor xz
			vector<double>         dF_yx;          // inc. def. grad. tensor yx
			vector<double>         dF_yy;          // inc. def. grad. tensor yy
			vector<double>         dF_yz;          // inc. def. grad. tensor yz
			vector<double>         dF_zx;          // inc. def. grad. tensor zx
			vector<double>         dF_zy;          // inc. def. grad. tensor zy
			vector<double>         dF_zz;          // inc. def. grad. tensor zz
			vector<double>         sigma_xx;       // stress tensor xx
			vector<double>         sigma_xy;       // stress tensor xy, yx
			vector<double>         sigma_xz;       // stress tensor xz, zx
			vector<double>         sigma_yy;       // stress tensor yy
			vector<double>         sigma_yz;       // stress tensor yz, zy
			vector<double>         sigma_zz;       // stress tensor zz
			vector<double>         epsilon_xx;     // strain tensor xx
			vector<double>         epsilon_xy;     // strain tensor xy, yx
			vector<double>         epsilon_xz;     // strain tensor xz, zx
			vector<double>         epsilon_yy;     // strain tensor yy
			vector<double>         epsilon_yz;     // strain tensor yz, zy
			vector<double>         epsilon_zz;     // strain tensor zz
			vector<double>         depsilon_xx;    // strain-rate tensor xx
			vector<double>         depsilon_xy;    // strain-rate tensor xy, yx
			vector<double>         depsilon_xz;    // strain-rate tensor xz, zx
			vector<double>         depsilon_yy;    // strain-rate tensor yy
			vector<double>         depsilon_yz;    // strain-rate tensor yz, zy
			vector<double>         depsilon_zz;    // strain-rate tensor zz
			vector<unsigned int>   vrt_1;       // grid nodal indicies
			vector<unsigned int>   vrt_2;       // grid nodal indicies
			vector<unsigned int>   vrt_3;       // grid nodal indicies
			vector<unsigned int>   vrt_4;       // grid nodal indicies
			vector<double>         phi_1;       // grid basis val's
			vector<double>         phi_2;       // grid basis val's
			vector<double>         phi_3;       // grid basis val's
			vector<double>         phi_4;       // grid basis val's
			vector<double>         grad_phi_1x; // grid basis grad. val's
			vector<double>         grad_phi_1y; // grid basis grad. val's
			vector<double>         grad_phi_1z; // grid basis grad. val's
			vector<double>         grad_phi_2x; // grid basis grad. val's
			vector<double>         grad_phi_2y; // grid basis grad. val's
			vector<double>         grad_phi_2z; // grid basis grad. val's
			vector<double>         grad_phi_3x; // grid basis grad. val's
			vector<double>         grad_phi_3y; // grid basis grad. val's
			vector<double>         grad_phi_3z; // grid basis grad. val's
			vector<double>         grad_phi_4x; // grid basis grad. val's
			vector<double>         grad_phi_4y; // grid basis grad. val's
			vector<double>         grad_phi_4z; // grid basis grad. val's
	};
}
#endif



