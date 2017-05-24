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
      MPMMaterial(const Array<double>& m_a,
                  const Array<double>& x_a,
                  const Array<double>& u_a,
                  const FiniteElement& element);
      
      std::vector<double>                    get_m()        {return m;};
      std::vector<double>                    get_rho()      {return rho;};
      std::vector<double>                    get_V0()       {return V0;};
      std::vector<double>                    get_V()        {return V;};
      std::vector<double>                    get_I()        {return I;};
      std::vector<std::vector<double>>       get_u_star()   {return u_star;};
      std::vector<std::vector<double>>       get_a()        {return a;};
      std::vector<std::vector<double>>       get_u()        {return u;};
      std::vector<std::vector<double>>       get_grad_u()   {return grad_u;};
      std::vector<std::vector<double>>       get_F()        {return F;};
      std::vector<std::vector<double>>       get_sigma()    {return sigma;};
      std::vector<std::vector<double>>       get_epsilon()  {return epsilon;};

      unsigned int get_num_particles() const {return n_p;};
      void         calculate_strain_rate(std::vector<std::vector<double>> eps);
      void         calculate_incremental_strain_rate();
      virtual void calculate_stress() = 0;
      void         calc_pi();
      
      Point* get_x_pt(unsigned int index) const {return x_pt.at(index);};
      
      double                    get_m(unsigned int index) const;
      void set_m(unsigned int index, double& value);
      
      double                    get_rho(unsigned int index) const;
      void set_rho(unsigned int index, double& value);
      
      double                    get_V0(unsigned int index) const;
      void set_V0(unsigned int index, double& value);
      
      double                    get_V(unsigned int index) const;
      void set_V(unsigned int index, double& value);
      
      std::vector<double>       get_x(unsigned int index) const;
      void set_x(unsigned int index, std::vector<double>& value);
      
      std::vector<double>       get_u(unsigned int index) const;
      void set_u(unsigned int index, std::vector<double>& value);
      
      std::vector<double>       get_a(unsigned int index) const;
      void set_a(unsigned int index, std::vector<double>& value);
      
      std::vector<double>       get_u_star(unsigned int index) const;
      void set_u_star(unsigned int index, std::vector<double>& value);
      
      std::vector<double>       get_phi(unsigned int index) const;
      void  set_phi(unsigned int index, std::vector<double>& value);
      
      std::vector<double>       get_grad_phi(unsigned int index) const;
      void  set_grad_phi(unsigned int index, std::vector<double>& value);
      
      std::vector<unsigned int> get_vrt(unsigned int index) const;
      void  set_vrt(unsigned int index, std::vector<unsigned int>& value);
      
      std::vector<double>       get_grad_u(unsigned int index) const;
      void  set_grad_u(unsigned int index, std::vector<double>& value);
      
      std::vector<double>       get_dF(unsigned int index) const;
      void  set_dF(unsigned int index, std::vector<double>& value);
      
      std::vector<double>       get_F(unsigned int index) const;
      void  set_F(unsigned int index, std::vector<double>& value);
      
      std::vector<double>       get_sigma(unsigned int index) const;
      void  set_sigma(unsigned int index, std::vector<double>& value);
      
      std::vector<double>       get_epsilon(unsigned int index) const;
      void  set_epsilon(unsigned int index, std::vector<double>& value);
      
      std::vector<double>       get_depsilon(unsigned int index) const;
      void  set_depsilon(unsigned int index, std::vector<double>& value);

    protected:
      unsigned int                           n_p;      // number of particles
      const unsigned int                     gdim;     // topological dimension
      const unsigned int                     sdim;     // element dimension
      std::vector<double>                    m;        // mass vector
      std::vector<double>                    rho;      // density vector
      std::vector<double>                    V0;       // initial volume vector
      std::vector<double>                    V;        // volume vector
      std::vector<double>                    I;        // identity tensor
      std::vector<Point*>                    x_pt;     // position Points
      std::vector<std::vector<double>>       x;        // position vector
      std::vector<std::vector<double>>       u;        // velocity vector
      std::vector<std::vector<double>>       u_star;   // grid vel. interp.
      std::vector<std::vector<double>>       a;        // acceleration vector
      std::vector<std::vector<double>>       grad_u;   // velocity grad. tensor
      std::vector<std::vector<unsigned int>> vrt;      // grid nodal indicies
      std::vector<std::vector<double>>       phi;      // grid basis val's
      std::vector<std::vector<double>>       grad_phi; // grid basis grad. val's
      std::vector<std::vector<double>>       dF;       // inc. def. grad. ten.
      std::vector<std::vector<double>>       F;        // def. gradient tensor
      std::vector<std::vector<double>>       sigma;    // stress tensor
      std::vector<std::vector<double>>       epsilon;  // strain-rate tensor
      std::vector<std::vector<double>>       depsilon; // inc. st.-rate tensor
      
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
