#include "Probe.h"

using namespace dolfin;

Probe::Probe(const FunctionSpace& V)
{

  const Mesh& mesh = *V.mesh();
  std::size_t gdim = mesh.geometry().dim();

  _element  = V.element();
  _dofmap   = V.dofmap();

  // Compute in tensor (one for scalar function, . . .)
  value_size_loc = 1;
  for (uint i = 0; i < _element->value_rank(); i++)
    value_size_loc *= _element->value_dimension(i);

  _probes.resize(value_size_loc);
    
  // Create work vector for basis
  std::vector<double> basis(value_size_loc);

  coefficients.resize(_element->space_dimension());

  // Create work vector for basis
  basis_matrix.resize(value_size_loc);
  for (uint i = 0; i < value_size_loc; ++i)
    basis_matrix[i].resize(_element->space_dimension());

}

/*
    # find the cell with point :
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
  int id = mesh.intersected_cell(point);

  // If the cell is on this process, then create an instance
  // of the Probe class. Otherwise raise a dolfin_error.
  if (id != -1)
  {
    // Create cell that contains point
    dolfin_cell = new Cell(mesh, id);
    ufc_cell    = new UFCCell(*dolfin_cell);

    for (uint i = 0; i < _element->space_dimension(); ++i)
    {
      _element->evaluate_basis(i, &basis[0], &x[0],
                               dolfin_cell.get_vertex_coordinates(),
                               dolfin_cell.orientation());
      for (uint j = 0; j < value_size_loc; ++j)
        basis_matrix[j][i] = basis[j];
    }
    vrt = dofmap.cell_dofs(dolfin_cell.index());
  }
  else
  {
    dolfin_error("MPMModel.cpp","eval probe","Probe is not found on processor");
  }
  return _probes[i];
}



