#include "MPMMaterial.h"

using namespace dolfin;

MPMMaterial::MPMMaterial(const Array<double>& m_a,
                         const Array<double>& x_a,
                         const Array<double>& u_a,
                         const unsigned int topological_dimension,
                         const unsigned int element_dimension)
            : tDim(topological_dimension), eDim(element_dimension)
{
  n_p = m_a.size();
  printf("::: MPMMaterialcpp with n_p = %u \t tDim = %u \t eDim = %u :::\n", 
         n_p, tDim, eDim);

  // one scalar or vector for each particle :
  m.resize(n_p);
  rho.resize(n_p);
  V0.resize(n_p);
  V.resize(n_p);
  x.resize(n_p);
  u.resize(n_p);
  u_star.resize(n_p);
  a.resize(n_p);
  grad_u.resize(n_p);
  vrt.resize(n_p);
  phi.resize(n_p);
  grad_phi.resize(n_p);
  F.resize(n_p);
  sigma.resize(n_p);
  epsilon.resize(n_p);
 
  // the flattened identity tensor :
  I.resize(tDim*eDim);
  
  # pragma omp parallel
  {
    // resize each of the vectors :
    # pragma omp for
    for (unsigned int i = 0; i < n_p; i++)
    {
      // these are vectors in topological dimension :
      x[i].resize(tDim);
      u[i].resize(tDim);
      u_star[i].resize(tDim);
      a[i].resize(tDim);
      
      // these are vectors in element dimension :
      vrt[i].resize(eDim);
      phi[i].resize(eDim);
    
      // these are flattened tensors defined with columns over each 
      //   topological dim. and rows over each element dimension :
      grad_u[i].resize(tDim*eDim);
      grad_phi[i].resize(tDim*eDim);
      F[i].resize(tDim*eDim);
      sigma[i].resize(tDim*eDim);
      epsilon[i].resize(tDim*eDim);
      
      m[i] = m_a.data()[i];  // initalize the mass
      unsigned int idx = 0;  // index variable
      for (unsigned int j = 0; j < tDim; j++)
      {
        idx          = i*tDim + j;
        x[i][j]      = x_a.data()[idx];
        u[i][j]      = u_a.data()[idx];
        u_star[i][j] = u[i][j];
      }
    }
  }
}

void MPMMaterial::update_strain_rate()
{
  // calculate particle strain-rate tensors :
  for (unsigned int i = 0; i < n_p; i++)
  {
    for (unsigned int j = 0; j < tDim; j++)
    {
      if (i == j)
        epsilon[i][j] = grad_u[i][j];
      else if (j > i)
      {
        eps_temp      = 0.5 * (grad_u[i][j] + grad_u[j][i]);
        epsilon[i][j] = eps_temp;
        epsilon[j][i] = eps_temp;
      }
    }
  }
}

static long n_stp     = (int) 1e10;  // number of discretizations
double t0, dt, pi, dx = 0.0;         // variables
int i;

void MPMMaterial::calc_pi()
{
  pi = 0.0;
  dx = 1.0 / (double) n_stp;         // the size of the step to take
  t0 = omp_get_wtime();              // start the timer

  # pragma omp parallel
  {
    double x = 0.0;                  // the part for this thread
                                    
    # pragma omp for reduction (+:pi) schedule(guided) nowait
    for (i = 0; i < n_stp; i++)    
    {                               
      x   = (i + 0.5) * dx;          // midpoint rule
      pi += 4.0 / (1.0 + x*x);       // increment the thread sum
    }                               
  }                                 
  pi *= dx;                          // pull dx multiplication ouside sum
  dt  = omp_get_wtime() - t0;        // time to compute

  printf("pi = %.2e \t pi - pi_true = %.2e \t dt = %.2e\n", 
          pi,          pi - M_PI,             dt);
}



