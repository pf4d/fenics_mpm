#include "MPMMaterial.h"

using namespace dolfin;

// no destructor yet :
MPMMaterial::~MPMMaterial() { }

// initialize the MPMMaterial variables :
MPMMaterial::MPMMaterial(const std::string&   name,
                         const int            n,
                         const Array<double>& x_a,
                         const Array<double>& u_a,
                         const FiniteElement& element) :
                           gdim(element.geometric_dimension()),
                           sdim(element.space_dimension()),
                           name(name),
                           n_p(n)
{
  printf("::: initializing MPMMaterialcpp `%s` with n_p = %u,  gdim = %u,"
         "  sdim = %u :::\n", name.c_str(), n_p, gdim, sdim);

  // one scalar or vector for each particle :
  m.resize(n_p);
  rho0.resize(n_p);
  rho.resize(n_p);
  V0.resize(n_p);
  V.resize(n_p);
  x_pt.resize(n_p);
  
  // these are vectors in topological dimension :
  x.resize(n_p*gdim);
  u.resize(n_p*gdim);
  u_star.resize(n_p*gdim);
  a_star.resize(n_p*gdim);
  a.resize(n_p*gdim);
    
  // these are vectors in element dimension :
  vrt.resize(n_p*sdim);
  phi.resize(n_p*sdim);
  
  // this is a flattened tensor defined with columns over each 
  //   topological dim. and rows over each element dimension :
  grad_phi.resize(n_p*gdim*sdim);
  
  // these are rank-two flattened tensors defined over each 
  //   topological dimension :
  grad_u.resize(n_p*gdim*gdim);
  grad_u_star.resize(n_p*gdim*gdim);
  dF.resize(n_p*gdim*gdim);
  F.resize(n_p*gdim*gdim);
  sigma.resize(n_p*gdim*gdim);
  epsilon.resize(n_p*gdim*gdim);
  depsilon.resize(n_p*gdim*gdim);
 
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
  
  // initialize the positions, Points, and velocities :
  for (unsigned int i = 0; i < n_p; i++)
  {
    unsigned int idx = 0;                        // index variable
    std::vector<double> x_t = {0.0, 0.0, 0.0};   // the vector to make a Point
    for (unsigned int j = 0; j < gdim; j++)
    {
      idx         = i*gdim + j;
      x_t[j]      = x_a[idx];
      x[idx]      = x_a[idx];
      u[idx]      = u_a[idx];
      u_star[idx] = u[idx];
    }
    Point* x_point = new Point(x_t[0], x_t[1], x_t[2]);  // create a new Point
    x_pt[i]        = x_point;                            // put it in the vector
  }
}

void MPMMaterial::set_initialized_by_mass(const bool val)
{
  printf("--- C++ set_initialized_by_mass(%s) ---\n", val ? "true" : "false");
  mass_init = val;
}

void MPMMaterial::initialize_mass(const Array<double>& m_a)
{
  printf("--- C++ initialize_mass() ---\n");
  // resize each of the vectors :
  for (unsigned int i = 0; i < n_p; i++)
  {
    m[i] = m_a[i];  // initalize the mass
  }
}

void MPMMaterial::initialize_volume(const Array<double>& V_a)
{
  printf("--- C++ initialize_volume() ---\n");
  // resize each of the vectors :
  for (unsigned int i = 0; i < n_p; i++)
  {
    V0[i] = V_a[i];  // initalize the initial volume
    V[i]  = V_a[i];  // initalize the current volume
  }
}

void MPMMaterial::initialize_mass_from_density(const double rho_a)
{
  printf("--- C++ initialize_mass_from_density(%g) ---\n", rho_a);
  // resize each of the vectors :
  for (unsigned int i = 0; i < n_p; i++)
  {
    double m_i = rho_a * V0[i];
    m[i]       = m_i;    // initialize the mass
    rho[i]     = rho_a;  // initialize the current denisty
    rho0[i]    = rho_a;  // initialize the initial density
  }
}

double MPMMaterial::det(std::vector<double>& u)
{
  unsigned int n = u.size(); // n x n tensor of rank n - 1
  double det;                // the determinant

  if      (n == 1) det = u[0];
  else if (n == 4) det = u[0] * u[3] - u[2] * u[1];
  else if (n == 9) det = + u[0] * u[4] * u[8]
                         + u[1] * u[5] * u[6]
                         + u[2] * u[3] * u[7]
                         - u[2] * u[4] * u[6]
                         - u[1] * u[3] * u[8]
                         - u[0] * u[5] * u[7];
  return det;
}

void MPMMaterial::calculate_strain_rate()
{
  unsigned int idn, idx, idx_T;

  // calculate particle strain-rate tensors :
  for (unsigned int i = 0; i < n_p; i++)
  {
    idn = i*gdim*gdim;
    for (unsigned int j = 0; j < gdim; j++)
    {
      for (unsigned int k = 0; k < gdim; k++)
      {
        idx   = j + k*gdim + idn;
        idx_T = k + j*gdim + idn;
        if (k == j)
          epsilon[idx] = grad_u[idx];
        else
        {
          epsilon[idx] = 0.5 * (grad_u[idx] + grad_u[idx_T]);
        }
      }
    }
  }
}

void MPMMaterial::calculate_incremental_strain_rate()
{
  unsigned int idn, idx, idx_T;

  // calculate particle strain-rate tensors :
  for (unsigned int i = 0; i < n_p; i++)
  {
    idn = i*gdim*gdim;
    for (unsigned int j = 0; j < gdim; j++)
    {
      for (unsigned int k = 0; k < gdim; k++)
      {
        idx   = j + k*gdim + idn;
        idx_T = k + j*gdim + idn;
        if (k == j)
          depsilon[idx] = grad_u[idx];
        else
          depsilon[idx] = 0.5 * (grad_u[idx] + grad_u[idx_T]);
      }
    }
  }
}

void MPMMaterial::initialize_tensors(double dt)
{
  printf("--- C++ initialize_tensors() ---\n");
  
  unsigned int idn, idx;
 
  // iterate through particles :
  for (unsigned int i = 0; i < n_p; i++) 
  {
    idn = i*gdim*gdim;

    // iterate through each element of the tensor :
    for (unsigned int j = 0; j < gdim*gdim; j++)
    {
      idx     = j + idn;
      dF[idx] = I[j] + grad_u[idx] * dt;
      F[idx]  = dF[idx];
    }
  }
  calculate_strain_rate();
  calculate_stress();
}

void MPMMaterial::calculate_initial_volume()
{
  printf("--- C++ calculate_initial_volume() ---\n");
  // iterate through particles :
  for (unsigned int i = 0; i < n_p; i++) 
  {
    // calculate inital volume from particle mass and density :
    double V0_i = m[i] / rho[i];
    V0[i]       = V0_i;
    V[i]        = V0_i;
  }
}

void MPMMaterial::update_deformation_gradient(double dt)
{
  unsigned int idn, idx;
  
  // iterate through particles :
  for (unsigned int i = 0; i < n_p; i++) 
  {
    idn = i*gdim*gdim;

    // iterate through each component of the tensor :
    for (unsigned int j = 0; j < gdim*gdim; j++)
    {
      idx      = j + idn;
      dF[idx]  = I[j] + 0.5 * (grad_u[idx] + grad_u_star[idx]) * dt;
      F[idx]  *= dF[idx];
    }
  }
}

void MPMMaterial::update_density()
{
  // iterate through particles :
  for (unsigned int i = 0; i < n_p; i++) 
  {
    rho[i] /= det(dF[i]);
  }
}

void MPMMaterial::update_volume()
{
  // iterate through particles :
  for (unsigned int i = 0; i < n_p; i++) 
  {
    V[i] *= det(dF[i]);
  }
}

void MPMMaterial::update_stress(double dt)
{
  unsigned int idn, idx;

  calculate_incremental_strain_rate();

  // iterate through particles :
  for (unsigned int i = 0; i < n_p; i++) 
  {
    idn = i*gdim*gdim;
    // iterate through each component of the tensor :
    for (unsigned int j = 0; j < gdim*gdim; j++)
    {
      idx           = j + idn;
      epsilon[idx] += depsilon[idx] * dt;
    }
  }
  calculate_stress();
}

void MPMMaterial::advect_particles(double dt)
{
  unsigned int idn, idx;

  // iterate through particles :
  for (unsigned int i = 0; i < n_p; i++) 
  {
    idn = i*gdim;
    for (unsigned int j = 0; j < gdim; j++)
    {
      idx      = j + idn;
      u[idx] += 0.5 * (a[idx] + a_star[idx])* dt;
      x[idx] += u_star[idx] * dt + 0.5 * a_star[idx] * dt*dt;
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

  //printf("pi = %.2e \t pi - pi_true = %.2e \t dt = %.2e\n", 
  //        pi,          pi - M_PI,             dt);
}



