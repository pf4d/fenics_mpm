#ifndef __MPMMATERIAL_H
#define __MPMMATERIAL_H

namespace dolfin
{
  class MPMMaterial
  {
    public:
      MPMMaterial(Array<double>& m,
                  Array<double>& x,
                  Array<double>& u,
                  const unsigned int topological_dimension,
                  const unsigned int element_dimension);
      
      std::vector<double>                    get_m()        {return m;};
      std::vector<double>                    get_rho()      {return rho;};
      std::vector<double>                    get_V0()       {return V0;};
      std::vector<double>                    get_V()        {return V;};
      std::vector<double>                    get_I()        {return I;};
      std::vector<std::vector<double>>       get_x()        {return x;};
      std::vector<std::vector<double>>       get_u()        {return u;};
      std::vector<std::vector<double>>       get_u_star()   {return u_star;};
      std::vector<std::vector<double>>       get_a()        {return a;};
      std::vector<std::vector<double>>       get_grad_u()   {return grad_u;};
      std::vector<std::vector<unsigned int>> get_vrt()      {return vrt;};
      std::vector<std::vector<double>>       get_phi()      {return phi;};
      std::vector<std::vector<double>>       get_grad_phi() {return grad_phi;};
      std::vector<std::vector<double>>       get_F()        {return F;};
      std::vector<std::vector<double>>       get_sigma()    {return sigma;};
      std::vector<std::vector<double>>       get_epsilon()  {return epsilon;};

    private:
      unsigned int                           n_p;      // number of particles
      const unsigned int                     tDim;     // topological dimension
      const unsigned int                     eDim;     // element dimension
      std::vector<double>                    m         // mass vector
      std::vector<double>                    rho       // density vector
      std::vector<double>                    V0        // initial volume vector
      std::vector<double>                    V         // volume vector
      std::vector<double>                    I         // identity tensor
      std::vector<std::vector<double>>       x         // position vector
      std::vector<std::vector<double>>       u         // velocity vector
      std::vector<std::vector<double>>       u_star    // grid vel. interp.
      std::vector<std::vector<double>>       a         // acceleration vector
      std::vector<std::vector<double>>       grad_u    // velocity grad. tensor
      std::vector<std::vector<unsigned int>> vrt       // grid nodal indicies
      std::vector<std::vector<double>>       phi       // grid basis val's
      std::vector<std::vector<double>>       grad_phi  // grid basis grad. val's
      std::vector<std::vector<double>>       F         // def. gradient tensor
      std::vector<std::vector<double>>       sigma     // stress tensor
      std::vector<std::vector<double>>       epsilon   // strain-rate tensor

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
