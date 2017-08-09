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

      bool                 get_mass_init()   {return mass_init;};
      const char *         get_name()        {return name.c_str();};
      std::vector<double>  get_m()           {return m;};
      std::vector<double>  get_x()           {return x;};
      std::vector<Point*>  get_x_pt()        {return x_pt;};
      std::vector<double>  get_rho0()        {return rho0;};
      std::vector<double>  get_rho()         {return rho;};
      std::vector<double>  get_V0()          {return V0;};
      std::vector<double>  get_V()           {return V;};
      std::vector<double>  get_I()           {return I;};
      std::vector<double>  get_u_star()      {return u_star;};
      std::vector<double>  get_a_star()      {return a_star;};
      std::vector<double>  get_a()           {return a;};
      std::vector<double>  get_u()           {return u;};
      std::vector<double>  get_grad_u()      {return grad_u;};
      std::vector<double>  get_grad_u_star() {return grad_u;};
      std::vector<double>  get_F()           {return F;};
      std::vector<double>  get_sigma()       {return sigma;};
      std::vector<double>  get_epsilon()     {return epsilon;};
     
      double       det(std::vector<double>& u);
      void         set_initialized_by_mass(const bool val);
      void         initialize_mass(const Array<double>& m_a);
      void         initialize_volume(const Array<double>& V_a);
      void         initialize_mass_from_density(const double rho_a);
      unsigned int get_num_particles() const {return n_p;};
      virtual void calculate_strain_rate();
      virtual void calculate_incremental_strain_rate();
      virtual void calculate_stress() = 0;

      virtual void initialize_tensors(double dt);
      virtual void calculate_initial_volume();
      virtual void update_deformation_gradient(double dt);
      virtual void update_density();
      virtual void update_volume();
      virtual void update_stress(double dt);
      virtual void advect_particles(double dt);
      void         calc_pi();
      
      Point* get_x_pt(unsigned int index) const {return x_pt.at(index);};
      
    protected:
      std::string                name;        // name of material
      bool                       mass_init;   // initialized via mass
      unsigned int               n_p;         // number of particles
      const unsigned int         gdim;        // topological dimension
      const unsigned int         sdim;        // element dimension
      std::vector<double>        m;           // mass vector
      std::vector<double>        rho0;        // initial density vector
      std::vector<double>        rho;         // density vector
      std::vector<double>        V0;          // initial volume vector
      std::vector<double>        V;           // volume vector
      std::vector<double>        I;           // identity tensor
      std::vector<Point*>        x_pt;        // position Points
      std::vector<double>        x;           // position vector
      std::vector<double>        u;           // velocity vector
      std::vector<double>        u_star;      // grid vel. interp.
      std::vector<double>        a_star;      // grid vel. interp.
      std::vector<double>        a;           // acceleration vector
      std::vector<double>        grad_u;      // velocity grad. tensor
      std::vector<double>        grad_u_star; // vel. grad. tensor
      std::vector<unsigned int>  vrt;         // grid nodal indicies
      std::vector<double>        phi;         // grid basis val's
      std::vector<double>        grad_phi;    // grid basis grad. val's
      std::vector<double>        dF;          // inc. def. grad. ten.
      std::vector<double>        F;           // def. gradient tensor
      std::vector<double>        sigma;       // stress tensor
      std::vector<double>        epsilon;     // strain-rate tensor
      std::vector<double>        depsilon;    // inc. st.-rate tensor
      
      // components of the strain-rate tensor :
      double  eps_xx;
      double  eps_yy;
      double  eps_zz;
      double  eps_xy;
      double  eps_xz;
      double  eps_yz;
      double  eps_temp; // temporary for calculation

  };
}
#endif
