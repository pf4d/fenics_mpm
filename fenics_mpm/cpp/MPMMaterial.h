#ifndef __MPMMATERIAL_H
#define __MPMMATERIAL_H

#include <dolfin/common/Array.h>
#include <dolfin/geometry/Point.h>
#include <dolfin/fem/FiniteElement.h>
#include <vector>
#include <omp.h>
#include <math.h>

namespace dolfin
{
  class MPMMaterial
  {
    public:
      virtual ~MPMMaterial() = 0;
      MPMMaterial(const std::string&   name,
                  const int            n,
                  const Array<double>& x_a,
                  const Array<double>& u_a,
                  const FiniteElement& element);

      bool                        get_mass_init()      {return mass_init;};
      const char *                get_name()           {return name.c_str();};
      std::vector<double>&        get_m()              {return m;};
      std::vector<Point*>&        get_x_pt()           {return x_pt;};
      std::vector<double>&        get_rho0()           {return rho0;};
      std::vector<double>&        get_rho()            {return rho;};
      std::vector<double>&        get_V0()             {return V0;};
      std::vector<double>&        get_V()              {return V;};
      std::vector<double>&        get_det_dF()         {return det_dF;};
      std::vector<double>&        get_x()              {return x;};
      std::vector<double>&        get_y()              {return y;};
      std::vector<double>&        get_z()              {return z;};
      std::vector<double>&        get_u_x()            {return u_x;};
      std::vector<double>&        get_u_y()            {return u_y;};
      std::vector<double>&        get_u_z()            {return u_z;};
      std::vector<double>&        get_a_x()            {return a_x;};
      std::vector<double>&        get_a_y()            {return a_y;};
      std::vector<double>&        get_a_z()            {return a_z;};
      std::vector<double>&        get_u_x_star()       {return u_x_star;};
      std::vector<double>&        get_u_y_star()       {return u_y_star;};
      std::vector<double>&        get_u_z_star()       {return u_z_star;};
      std::vector<double>&        get_a_x_star()       {return a_x_star;};
      std::vector<double>&        get_a_y_star()       {return a_y_star;};
      std::vector<double>&        get_a_z_star()       {return a_z_star;};
      std::vector<double>&        get_grad_u_xx()      {return grad_u_xx;};
      std::vector<double>&        get_grad_u_xy()      {return grad_u_xy;};
      std::vector<double>&        get_grad_u_xz()      {return grad_u_xz;};
      std::vector<double>&        get_grad_u_yx()      {return grad_u_yx;};
      std::vector<double>&        get_grad_u_yy()      {return grad_u_yy;};
      std::vector<double>&        get_grad_u_yz()      {return grad_u_yz;};
      std::vector<double>&        get_grad_u_zx()      {return grad_u_zx;};
      std::vector<double>&        get_grad_u_zy()      {return grad_u_zy;};
      std::vector<double>&        get_grad_u_zz()      {return grad_u_zz;};
      std::vector<double>&        get_grad_u_xx_star() {return grad_u_xx_star;};
      std::vector<double>&        get_grad_u_xy_star() {return grad_u_xy_star;};
      std::vector<double>&        get_grad_u_xz_star() {return grad_u_xz_star;};
      std::vector<double>&        get_grad_u_yx_star() {return grad_u_yx_star;};
      std::vector<double>&        get_grad_u_yy_star() {return grad_u_yy_star;};
      std::vector<double>&        get_grad_u_yz_star() {return grad_u_yz_star;};
      std::vector<double>&        get_grad_u_zx_star() {return grad_u_zx_star;};
      std::vector<double>&        get_grad_u_zy_star() {return grad_u_zy_star;};
      std::vector<double>&        get_grad_u_zz_star() {return grad_u_zz_star;};
      std::vector<double>&        get_dF_xx()          {return dF_xx;};
      std::vector<double>&        get_dF_xy()          {return dF_xy;};
      std::vector<double>&        get_dF_xz()          {return dF_xz;};
      std::vector<double>&        get_dF_yx()          {return dF_yx;};
      std::vector<double>&        get_dF_yy()          {return dF_yy;};
      std::vector<double>&        get_dF_yz()          {return dF_yz;};
      std::vector<double>&        get_dF_zx()          {return dF_zx;};
      std::vector<double>&        get_dF_zy()          {return dF_zy;};
      std::vector<double>&        get_dF_zz()          {return dF_zz;};
      std::vector<double>&        get_F_xx()           {return F_xx;};
      std::vector<double>&        get_F_xy()           {return F_xy;};
      std::vector<double>&        get_F_xz()           {return F_xz;};
      std::vector<double>&        get_F_yx()           {return F_yx;};
      std::vector<double>&        get_F_yy()           {return F_yy;};
      std::vector<double>&        get_F_yz()           {return F_yz;};
      std::vector<double>&        get_F_zx()           {return F_zx;};
      std::vector<double>&        get_F_zy()           {return F_zy;};
      std::vector<double>&        get_F_zz()           {return F_zz;};
      std::vector<double>&        get_sigma_xx()       {return sigma_xx;};
      std::vector<double>&        get_sigma_xy()       {return sigma_xy;};
      std::vector<double>&        get_sigma_xz()       {return sigma_xz;};
      std::vector<double>&        get_sigma_yy()       {return sigma_yy;};
      std::vector<double>&        get_sigma_yz()       {return sigma_yz;};
      std::vector<double>&        get_sigma_zz()       {return sigma_zz;};
      std::vector<double>&        get_epsilon_xx()     {return epsilon_xx;};
      std::vector<double>&        get_epsilon_xy()     {return epsilon_xy;};
      std::vector<double>&        get_epsilon_xz()     {return epsilon_xz;};
      std::vector<double>&        get_epsilon_yy()     {return epsilon_yy;};
      std::vector<double>&        get_epsilon_yz()     {return epsilon_yz;};
      std::vector<double>&        get_epsilon_zz()     {return epsilon_zz;};
      std::vector<unsigned int>&  get_vrt_1()          {return vrt_1;};
      std::vector<unsigned int>&  get_vrt_2()          {return vrt_2;};
      std::vector<unsigned int>&  get_vrt_3()          {return vrt_3;};
      std::vector<unsigned int>&  get_vrt_4()          {return vrt_4;};
      std::vector<double>&        get_phi_1()          {return phi_1;};
      std::vector<double>&        get_phi_2()          {return phi_2;};
      std::vector<double>&        get_phi_3()          {return phi_3;};
      std::vector<double>&        get_phi_4()          {return phi_4;};
      std::vector<double>&        get_grad_phi_1x()    {return grad_phi_1x;};
      std::vector<double>&        get_grad_phi_1y()    {return grad_phi_1y;};
      std::vector<double>&        get_grad_phi_1z()    {return grad_phi_1z;};
      std::vector<double>&        get_grad_phi_2x()    {return grad_phi_2x;};
      std::vector<double>&        get_grad_phi_2y()    {return grad_phi_2y;};
      std::vector<double>&        get_grad_phi_2z()    {return grad_phi_2z;};
      std::vector<double>&        get_grad_phi_3x()    {return grad_phi_3x;};
      std::vector<double>&        get_grad_phi_3y()    {return grad_phi_3y;};
      std::vector<double>&        get_grad_phi_3z()    {return grad_phi_3z;};
      std::vector<double>&        get_grad_phi_4x()    {return grad_phi_4x;};
      std::vector<double>&        get_grad_phi_4y()    {return grad_phi_4y;};
      std::vector<double>&        get_grad_phi_4z()    {return grad_phi_4z;};
     
      void         set_initialized_by_mass(const bool val);
      void         initialize_mass(const Array<double>& m_a);
      void         initialize_volume(const Array<double>& V_a);
      void         initialize_mass_from_density(const double rho_a);
      unsigned int get_num_particles() const {return n_p;};
      void         calculate_strain_rate();
      void         calculate_incremental_strain_rate();
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
      const std::string          name;        // name of material
      bool                       mass_init;   // initialized via mass
      unsigned int               n_p;         // number of particles
      const unsigned int         gdim;        // topological dimension
      const unsigned int         sdim;        // element dimension
      std::vector<double>        m;           // mass vector
      std::vector<double>        rho0;        // initial density vector
      std::vector<double>        rho;         // density vector
      std::vector<double>        V0;          // initial volume vector
      std::vector<double>        V;           // volume vector
      std::vector<double>        det_dF;      // det. of incremental def. grad.
      std::vector<Point*>        x_pt;        // (x,y,z) position Points
      std::vector<double>        x;           // x-position vector
      std::vector<double>        y;           // y-position vector
      std::vector<double>        z;           // z-position vector
      std::vector<double>        u_x;         // x-component of velocity vector
      std::vector<double>        u_y;         // y-component of velocity vector
      std::vector<double>        u_z;         // z-component of velocity vector
      std::vector<double>        u_x_star;    // temporary x-velocity interp.
      std::vector<double>        u_y_star;    // temporary y-velocity interp.
      std::vector<double>        u_z_star;    // temporary z-velocity interp.
      std::vector<double>        a_x_star;    // temporary x-accel. interp.
      std::vector<double>        a_y_star;    // temporary y-accel. interp.
      std::vector<double>        a_z_star;    // temporary z-accel. interp.
      std::vector<double>        a_x;         // x-acceleration vector
      std::vector<double>        a_y;         // y-acceleration vector
      std::vector<double>        a_z;         // z-acceleration vector
      std::vector<double>        grad_u_xx;   // velocity grad. tensor xx
      std::vector<double>        grad_u_xy;   // velocity grad. tensor xy
      std::vector<double>        grad_u_xz;   // velocity grad. tensor xz
      std::vector<double>        grad_u_yx;   // velocity grad. tensor yx
      std::vector<double>        grad_u_yy;   // velocity grad. tensor yy
      std::vector<double>        grad_u_yz;   // velocity grad. tensor yz
      std::vector<double>        grad_u_zx;   // velocity grad. tensor zx
      std::vector<double>        grad_u_zy;   // velocity grad. tensor zy
      std::vector<double>        grad_u_zz;   // velocity grad. tensor zz
      std::vector<double>        grad_u_xx_star; // velocity grad. tensor xx
      std::vector<double>        grad_u_xy_star; // velocity grad. tensor xy
      std::vector<double>        grad_u_xz_star; // velocity grad. tensor xz
      std::vector<double>        grad_u_yx_star; // velocity grad. tensor yx
      std::vector<double>        grad_u_yy_star; // velocity grad. tensor yy
      std::vector<double>        grad_u_yz_star; // velocity grad. tensor yz
      std::vector<double>        grad_u_zx_star; // velocity grad. tensor zx
      std::vector<double>        grad_u_zy_star; // velocity grad. tensor zy
      std::vector<double>        grad_u_zz_star; // velocity grad. tensor zz
      std::vector<double>        F_xx;           // deformation grad. tensor xx
      std::vector<double>        F_xy;           // deformation grad. tensor xy
      std::vector<double>        F_xz;           // deformation grad. tensor xz
      std::vector<double>        F_yx;           // deformation grad. tensor yx
      std::vector<double>        F_yy;           // deformation grad. tensor yy
      std::vector<double>        F_yz;           // deformation grad. tensor yz
      std::vector<double>        F_zx;           // deformation grad. tensor zx
      std::vector<double>        F_zy;           // deformation grad. tensor zy
      std::vector<double>        F_zz;           // deformation grad. tensor zz
      std::vector<double>        dF_xx;          // inc. def. grad. tensor xx
      std::vector<double>        dF_xy;          // inc. def. grad. tensor xy
      std::vector<double>        dF_xz;          // inc. def. grad. tensor xz
      std::vector<double>        dF_yx;          // inc. def. grad. tensor yx
      std::vector<double>        dF_yy;          // inc. def. grad. tensor yy
      std::vector<double>        dF_yz;          // inc. def. grad. tensor yz
      std::vector<double>        dF_zx;          // inc. def. grad. tensor zx
      std::vector<double>        dF_zy;          // inc. def. grad. tensor zy
      std::vector<double>        dF_zz;          // inc. def. grad. tensor zz
      std::vector<double>        sigma_xx;       // stress tensor xx
      std::vector<double>        sigma_xy;       // stress tensor xy, yx
      std::vector<double>        sigma_xz;       // stress tensor xz, zx
      std::vector<double>        sigma_yy;       // stress tensor yy
      std::vector<double>        sigma_yz;       // stress tensor yz, zy
      std::vector<double>        sigma_zz;       // stress tensor zz
      std::vector<double>        epsilon_xx;     // strain-rate tensor xx
      std::vector<double>        epsilon_xy;     // strain-rate tensor xy, yx
      std::vector<double>        epsilon_xz;     // strain-rate tensor xz, zx
      std::vector<double>        epsilon_yy;     // strain-rate tensor yy
      std::vector<double>        epsilon_yz;     // strain-rate tensor yz, zy
      std::vector<double>        epsilon_zz;     // strain-rate tensor zz
      std::vector<double>        depsilon_xx;    // inc. st.-rate tensor xx
      std::vector<double>        depsilon_xy;    // inc. st.-rate tensor xy, yx
      std::vector<double>        depsilon_xz;    // inc. st.-rate tensor xz, zx
      std::vector<double>        depsilon_yy;    // inc. st.-rate tensor yy
      std::vector<double>        depsilon_yz;    // inc. st.-rate tensor yz, zy
      std::vector<double>        depsilon_zz;    // inc. st.-rate tensor zz
      std::vector<unsigned int>  vrt_1;       // grid nodal indicies
      std::vector<unsigned int>  vrt_2;       // grid nodal indicies
      std::vector<unsigned int>  vrt_3;       // grid nodal indicies
      std::vector<unsigned int>  vrt_4;       // grid nodal indicies
      std::vector<double>        phi_1;       // grid basis val's
      std::vector<double>        phi_2;       // grid basis val's
      std::vector<double>        phi_3;       // grid basis val's
      std::vector<double>        phi_4;       // grid basis val's
      std::vector<double>        grad_phi_1x; // grid basis grad. val's
      std::vector<double>        grad_phi_1y; // grid basis grad. val's
      std::vector<double>        grad_phi_1z; // grid basis grad. val's
      std::vector<double>        grad_phi_2x; // grid basis grad. val's
      std::vector<double>        grad_phi_2y; // grid basis grad. val's
      std::vector<double>        grad_phi_2z; // grid basis grad. val's
      std::vector<double>        grad_phi_3x; // grid basis grad. val's
      std::vector<double>        grad_phi_3y; // grid basis grad. val's
      std::vector<double>        grad_phi_3z; // grid basis grad. val's
      std::vector<double>        grad_phi_4x; // grid basis grad. val's
      std::vector<double>        grad_phi_4y; // grid basis grad. val's
      std::vector<double>        grad_phi_4z; // grid basis grad. val's
  };
}
#endif



