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
  dF.resize(n_p);
  F.resize(n_p);
  sigma.resize(n_p);
  epsilon.resize(n_p);
  depsilon.resize(n_p);
 
  // the flattened identity tensor :
  I.resize(gdim*gdim);

  // create the identity tensor
  // FIXME: rewrite any code that uses "I" to skip multiplication by zero.
  for (unsigned int i = 0; i < gdim; i++)
  {
    for (unsigned int j = 0; j < gdim; j++)
    {
      if (i == j)
      {
        I[i + j*gdim] = 1.0;
      }
      else
      {
        I[i + j*gdim] = 0.0;
      }
    }
  }
  
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
  
    // this is a flattened tensor defined with columns over each 
    //   topological dim. and rows over each element dimension :
    grad_phi[i].resize(gdim*sdim);
    
    // these are rank-two flattened tensors defined over each 
    //   topological dimension :
    grad_u[i].resize(gdim*gdim);
    dF[i].resize(gdim*gdim);
    F[i].resize(gdim*gdim);
    sigma[i].resize(gdim*gdim);
    epsilon[i].resize(gdim*gdim);
    depsilon[i].resize(gdim*gdim);
    
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

double               MPMMaterial::get_m(unsigned int index) const
{
  return m.at(index);
}

void  MPMMaterial::set_m(unsigned int index, double& value)
{
  m.at(index) = value;
}

double               MPMMaterial::get_rho(unsigned int index) const
{
  return rho.at(index);
}

void  MPMMaterial::set_rho(unsigned int index, double& value)
{
  rho.at(index) = value;
}

double               MPMMaterial::get_V0(unsigned int index) const
{
  return V0.at(index);
}

void  MPMMaterial::set_V0(unsigned int index, double& value)
{
  V0.at(index) = value;
}

double               MPMMaterial::get_V(unsigned int index) const
{
  return V.at(index);
}

void  MPMMaterial::set_V(unsigned int index, double& value)
{
  V.at(index) = value;
}

std::vector<double>  MPMMaterial::get_x(unsigned int index) const
{
  return x.at(index);
}

void  MPMMaterial::set_x(unsigned int index, std::vector<double>& value)
{
  x.at(index) = value;
}

std::vector<double>  MPMMaterial::get_u(unsigned int index) const
{
  return u.at(index);
}

void  MPMMaterial::set_u(unsigned int index, std::vector<double>& value)
{
  u.at(index) = value;
}

std::vector<double>  MPMMaterial::get_a(unsigned int index) const
{
  return a.at(index);
}

void  MPMMaterial::set_a(unsigned int index, std::vector<double>& value)
{
  a.at(index) = value;
}

std::vector<double>  MPMMaterial::get_u_star(unsigned int index) const
{
  return u_star.at(index);
}

void  MPMMaterial::set_u_star(unsigned int index, std::vector<double>& value)
{
  u_star.at(index) = value;
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

std::vector<double>  MPMMaterial::get_grad_u(unsigned int index) const
{
  return grad_u.at(index);
}

void  MPMMaterial::set_grad_u(unsigned int index, std::vector<double>& value)
{
  grad_u.at(index) = value;
}

std::vector<double>  MPMMaterial::get_dF(unsigned int index) const
{
  return dF.at(index);
}

void  MPMMaterial::set_dF(unsigned int index, std::vector<double>& value)
{
  dF.at(index) = value;
}

std::vector<double>  MPMMaterial::get_F(unsigned int index) const
{
  return F.at(index);
}

void  MPMMaterial::set_F(unsigned int index, std::vector<double>& value)
{
  F.at(index) = value;
}

std::vector<double>  MPMMaterial::get_sigma(unsigned int index) const
{
  return sigma.at(index);
}

void  MPMMaterial::set_sigma(unsigned int index, std::vector<double>& value)
{
  sigma.at(index) = value;
}

std::vector<double>  MPMMaterial::get_epsilon(unsigned int index) const
{
  return epsilon.at(index);
}

void  MPMMaterial::set_epsilon(unsigned int index, std::vector<double>& value)
{
  epsilon.at(index) = value;
}

std::vector<double>  MPMMaterial::get_depsilon(unsigned int index) const
{
  return depsilon.at(index);
}

void  MPMMaterial::set_depsilon(unsigned int index, std::vector<double>& value)
{
  depsilon.at(index) = value;
}

void MPMMaterial::calculate_strain_rate()
{
  unsigned int idx, idx_T;

  // calculate particle strain-rate tensors :
  for (unsigned int i = 0; i < n_p; i++)
  {
    for (unsigned int j = 0; j < gdim; j++)
    {
      for (unsigned int k = 0; k < gdim; k++)
      {
        idx   = j + k*gdim;
        idx_T = k + j*gdim;
        if (k == j)
          epsilon[i][idx] = grad_u[i][idx];
        else
        {
          epsilon[i][idx] = 0.5 * (grad_u[i][idx] + grad_u[i][idx_T]);
        }
      }
    }
  }
}

void MPMMaterial::calculate_incremental_strain_rate()
{
  unsigned int idx, idx_T;

  // calculate particle strain-rate tensors :
  for (unsigned int i = 0; i < n_p; i++)
  {
    for (unsigned int j = 0; j < gdim; j++)
    {
      for (unsigned int k = 0; k < gdim; k++)
      {
        idx   = j + k*gdim;
        idx_T = k + j*gdim;
        if (k == j)
          depsilon[i][idx] = grad_u[i][idx];
        else
          depsilon[i][idx] = 0.5 * (grad_u[i][idx] + grad_u[i][idx_T]);
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



