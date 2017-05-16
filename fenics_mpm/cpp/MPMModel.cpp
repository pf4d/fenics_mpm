#include "MPMModel.h"

using namespace dolfin;

Probe::Probe(const FunctionSpace& V)
{

  const Mesh& mesh = *V.mesh();
  std::size_t gdim = mesh.geometry().dim();

  element  = V.element();
  dofmap   = V.dofmap();

  // Compute in tensor (one for scalar function, . . .)
  value_size_loc = 1;
  for (unsigned int i = 0; i < element->value_rank(); i++)
    value_size_loc *= element->value_dimension(i);

  // Create work vector for basis
  phi.resize(value_size_loc);

  // Create work vector for basis
  grad_phi.resize(value_size_loc);
  for (unsigned int i = 0; i < value_size_loc; ++i)
    grad_phi[i].resize(element->space_dimension());
                               
}                              
                              _
/*                             
    # find the cell with point  :
    x_pt       = Point(x)
    cell_id    = mesh.bounding_box_tree().compute_first_entity_collision(x_pt)
    cell       = Cell(mesh, cell_id)
    coord_dofs = cell.get_vertex_coordinates()       # local coordinates
    
    # array for all basis functions of the cell :
    phi = np.zeros(element.space_dimension(), dtype=float)
    
    # array for values with derivatives of all 
    # basis functions, 2 * element dim :
    grad_phi = np.zeros(2*element.space_dimension(), dtype=float)
    
    # compute basis function values :
    element.evaluate_basis_all(phi, x, coord_dofs, cell.orientation())
    
    # compute 1st order derivatives :
    element.evaluate_basis_derivatives_all(1, grad_phi, x, 
                                           coord_dofs, cell.orientation())

    # reshape such that rows are [d/dx, d/dy] :
    grad_phi = grad_phi.reshape((-1, 2))

    # get corresponding vertex indices, in dof indicies : 
    vrt = self.dofmap.cell_dofs(cell.index())

    return vrt, phi, grad_phi
*/

void Probe::eval(const Array<double>& x)
{

  // Find the cell that contains probe
  const Point point(gdim, x.data());
  cell_id = mesh.bounding_box_tree()->compute_first_entity_collision(x_pt);
  cell    = new Cell(mesh, cell_id);

  element->evaluate_basis_all(&phi, &x,
                              cell.get_vertex_coordinates(),
                              cell.orientation());
  
  element->evaluate_basis_derivatives_all(1, &grad_phi, &x,
                                          cell.get_vertex_coordinates(),
                                          cell.orientation());
  grad_phi.resize(value_size_loc);
  for (unsigned int i = 0; i < value_size_loc; ++i)
    grad_phi[i].resize(element->space_dimension());

  vrt = dofmap.cell_dofs(cell.index());
}



