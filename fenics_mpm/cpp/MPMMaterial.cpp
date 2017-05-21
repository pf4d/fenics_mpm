#include "MPMMaterial.h"

using namespace dolfin;

// no destructor yet :
MPMMaterial::~MPMMaterial() { }

// initialize the MPMMaterial variables :
MPMMaterial::MPMMaterial(const Array<double>& m_a,
                         const Array<double>& x_a,
                         const Array<double>& u_a,
                         const FiniteElement& element) :
                           gdim(element.geometric_dimension()),
                           sdim(element.space_dimension())
{
  n_p = m_a.size();
  printf("::: MPMMaterialcpp with n_p = %u \t gdim = %u \t sdim = %u :::\n", 
         n_p, gdim, sdim);

  // one scalar or vector for each particle :
  m.resize(n_p);
  rho.resize(n_p);
  V0.resize(n_p);
  V.resize(n_p);
  x.resize(n_p);
  x_pt.resize(n_p);
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
  I.resize(gdim*sdim);
  
  // resize each of the vectors :
  for (unsigned int i = 0; i < n_p; i++)
  {
    // these are vectors in topological dimension :
    x[i].resize(gdim);
    u[i].resize(gdim);
    u_star[i].resize(gdim);
    a[i].resize(gdim);
    
    // these are vectors in element dimension :
    vrt[i].resize(sdim);
    phi[i].resize(sdim);
  
    // these are flattened tensors defined with columns over each 
    //   topological dim. and rows over each element dimension :
    grad_u[i].resize(gdim*sdim);
    grad_phi[i].resize(gdim*sdim);
    F[i].resize(gdim*sdim);
    sigma[i].resize(gdim*sdim);
    epsilon[i].resize(gdim*sdim);
    
    m[i] = m_a[i];                               // initalize the mass
    unsigned int idx = 0;                        // index variable
    std::vector<double> x_t = {0.0, 0.0, 0.0};   // the vector to make a Point
    for (unsigned int j = 0; j < gdim; j++)
    {
      idx          = i*gdim + j;
      x_t[j]       = x_a[idx];
      x[i][j]      = x_a[idx];
      u[i][j]      = u_a[idx];
      u_star[i][j] = u[i][j];
    }
    Point* x_point = new Point(x_t[0], x_t[1], x_t[2]);  // create a new Point
    x_pt[i]        = x_point;                            // put it in the vector
  }
}

std::vector<double>  MPMMaterial::get_x(unsigned int index) const
{
  return x.at(index);
}

void  MPMMaterial::set_x(unsigned int index, std::vector<double>& value)
{
  x.at(index) = value;
}

std::vector<double>  MPMMaterial::get_phi(unsigned int index) const
{
  return phi.at(index);
}

void  MPMMaterial::set_phi(unsigned int index, std::vector<double>& value)
{
  phi.at(index) = value;
}

std::vector<double>  MPMMaterial::get_grad_phi(unsigned int index) const
{
  return grad_phi.at(index);
}

void  MPMMaterial::set_grad_phi(unsigned int index, std::vector<double>& value)
{
  grad_phi.at(index) = value;
}

std::vector<unsigned int>  MPMMaterial::get_vrt(unsigned int index) const
{
  return vrt.at(index);
}

void  MPMMaterial::set_vrt(unsigned int index, std::vector<unsigned int>& value)
{
  vrt.at(index) = value;
}

void MPMMaterial::update_strain_rate()
{
  // calculate particle strain-rate tensors :
  for (unsigned int i = 0; i < n_p; i++)
  {
    for (unsigned int j = 0; j < gdim; j++)
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



