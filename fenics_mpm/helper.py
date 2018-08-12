# -*- coding: iso-8859-15 -*-

import inspect
from   dolfin   import *
from   colored  import fg, attr
import numpy        as np


def raiseNotDefined():
  fileName = inspect.stack()[1][1]
  line     = inspect.stack()[1][2]
  method   = inspect.stack()[1][3]
       
  text = "*** Method not implemented: %s at line %s of %s"
  print text % (method, line, fileName)
  sys.exit(1)

def print_min_max(u, title, color='97'):
  r"""
  Print the minimum and maximum values of ``u``, a Vector, Function, or array.

  :param u: the variable to print the min and max of
  :param title: the name of the function to print
  :param color: the color of printed text
  :type u: :class:`~fenics.GenericVector`, :class:`~numpy.ndarray`, :class:`~fenics.Function`, int, float, :class:`~fenics.Constant`
  :type title: string
  :type color: string
  """
  if isinstance(u, GenericVector):
    uMin = MPI.min(mpi_comm_world(), u.min())
    uMax = MPI.max(mpi_comm_world(), u.max())
    s    = title + ' <min, max> : <%.3e, %.3e>' % (uMin, uMax)
    print_text(s, color)
  elif isinstance(u, np.ndarray):
    if u.dtype != np.float64:
      u = u.astype(float64)
    uMin = MPI.min(mpi_comm_world(), u.min())
    uMax = MPI.max(mpi_comm_world(), u.max())
    s    = title + ' <min, max> : <%.3e, %.3e>' % (uMin, uMax)
    print_text(s, color)
  elif isinstance(u, Function):# \
    #   or isinstance(u, dolfin.functions.function.Function):
    uMin = MPI.min(mpi_comm_world(), u.vector().min())
    uMax = MPI.max(mpi_comm_world(), u.vector().max())
    s    = title + ' <min, max> : <%.3e, %.3e>' % (uMin, uMax)
    print_text(s, color)
  elif isinstance(u, int) or isinstance(u, float):
    s    = title + ' : %.3e' % u
    print_text(s, color)
  elif isinstance(u, Constant):
    s    = title + ' : %.3e' % u(0)
    print_text(s, color)
  else:
    er = title + ": print_min_max function requires a Vector, Function" \
         + ", array, int or float, not %s." % type(u)
    print_text(er, 'red', 1)


def get_text(text, color='white', atrb=0, cls=None):
  r"""
  Returns text ``text`` from calling class ``cls`` for printing at a later time.

  :param text: the text to print
  :param color: the color of the text to print
  :param atrb: attributes to send use by ``colored`` package
  :param cls: the calling class
  :type text: string
  :type color: string
  :type atrb: int
  :type cls: object
  """
  if cls is not None:
    color = cls.color()
  if MPI.rank(mpi_comm_world())==0:
    if atrb != 0:
      text = ('%s%s' + text + '%s') % (fg(color), attr(atrb), attr(0))
    else:
      text = ('%s' + text + '%s') % (fg(color), attr(0))
    return text


def print_text(text, color='white', atrb=0, cls=None):
  r"""
  Print text ``text`` from calling class ``cls`` to the screen.

  :param text: the text to print
  :param color: the color of the text to print
  :param atrb: attributes to send use by ``colored`` package
  :param cls: the calling class
  :type text: string
  :type color: string
  :type atrb: int
  :type cls: object
  """
  if cls is not None:
    color = cls.color()
  if MPI.rank(mpi_comm_world())==0:
    if atrb != 0:
      text = ('%s%s' + text + '%s') % (fg(color), attr(atrb), attr(0))
    else:
      text = ('%s' + text + '%s') % (fg(color), attr(0))
    print text

def calculate_mesh_midpoints_and_volumes(mesh):
  """
  """
  s = "::: CALCULATING CELL MIDPOINTS AND VOLUMES :::"
  print_text(s)

  dim   = mesh.ufl_cell().topological_dimension()
  s = "    - iterating through %i cells of %i-dimensional mesh - "
  print_text(s % (mesh.num_cells(), dim))

  x,y,z,V = [],[],[],[]
  if dim == 2:
    for c in cells(mesh):
      x.append(c.midpoint().x())
      y.append(c.midpoint().y())
      V.append(c.volume())
    x = np.array(x)
    y = np.array(y)
    V = np.array(V)
    X = np.ascontiguousarray(np.array([x, y]).T)
  elif dim == 3:
    for c in cells(mesh):
      x.append(c.midpoint().x())
      y.append(c.midpoint().y())
      z.append(c.midpoint().z())
      V.append(c.volume())
    x = np.array(x)
    y = np.array(y)
    z = np.array(z)
    V = np.array(V)
    X = np.ascontiguousarray(np.array([x, y, z]).T)
  s = "    - done - "
  print_text(s)
  print_min_max(x, 'x')
  print_min_max(y, 'y')
  if dim == 3: print_min_max(z, 'z')
  print_min_max(V, 'V')
  return (X, V)

