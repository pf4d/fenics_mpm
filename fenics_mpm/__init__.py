__version__    = '1.0'
__author__     = 'Evan M. Cummings'
__license__    = 'LGPL-3'
__maintainer__ = 'Evan M. Cummings'
__email__      = 'evan.cummings@aalto.fi'

__all__ = []

import pkgutil
import inspect
import matplotlib as mpl
#mpl.use('Agg')
mpl.rcParams['font.family']          = 'serif'
mpl.rcParams['legend.fontsize']      = 'medium'
mpl.rcParams['text.usetex']          = True
mpl.rcParams['text.latex.preamble']  = ['\usepackage[mathscr]{euscript}']
#mpl.rcParams['contour.negative_linestyle']   = 'solid'

# open the cpp code :
import os
from   fenics  import compile_extension_module

cpp_src_dir     = os.path.dirname(os.path.abspath(__file__)) + "/cpp/"
headers         = ["MPMMaterial.h",
                   "MPMElasticMaterial.h",
                   "MPMImpenetrableMaterial.h",
                   "MPMModel.h"]
code            = ''
for header_file in headers:
  header_file   = open(cpp_src_dir + header_file, "r")
  code         += header_file.read()
  header_file.close()

system_headers  = ['numpy/arrayobject.h',
                   'dolfin/geometry/BoundingBoxTree.h',
                   'dolfin/fem/GenericDofMap.h',
                   'dolfin/function/FunctionSpace.h']
swigargs        = ['-c++', '-fcompact', '-O', '-I.', '-small']
cmake_packages  = ['DOLFIN']
module_name     = "MPMModelcpp"
sources         = ["MPMMaterial.cpp",
                   "MPMElasticMaterial.cpp",
                   "MPMImpenetrableMaterial.cpp",
                   "MPMModel.cpp"]
source_dir      = cpp_src_dir
include_dirs    = [".", cpp_src_dir, 
                   '/usr/lib/petscdir/3.7.3/x86_64-linux-gnu-real/include/']
additional_decl = """
%init%{
  import_array();
  %}

  // Include global SWIG interface files:
  // Typemaps, shared_ptr declarations, exceptions, version
  %include <boost_shared_ptr.i>

  // Global typemaps and forward declarations
  %include "dolfin/swig/typemaps/includes.i"
  %include "dolfin/swig/forwarddeclarations.i"

  // Global exceptions
  %include <exception.i>

  // Local shared_ptr declarations
  %shared_ptr(dolfin::Function)
  %shared_ptr(dolfin::FunctionSpace)

  // %import types from submodule function of SWIG module function
  %import(module="dolfin.cpp.function") "dolfin/function/Function.h"
  %import(module="dolfin.cpp.function") "dolfin/function/FunctionSpace.h"

  %feature("autodoc", "1");
"""
#compiled_module = instant.build_module(
#    modulename = module_name,
#    code=code,
#    source_directory=source_dir,
#    additional_declarations=additional_decl,
#    system_headers=system_headers,
#    include_dirs=include_dirs,
#    swigargs=swigargs,
#    sources=sources,
#    cmake_packages=cmake_packages)

# compile this with Instant JIT compiler :
inst_params = {'code'                      : code,
               'module_name'               : module_name,
               'source_directory'          : cpp_src_dir,
               'sources'                   : sources,
               'additional_system_headers' : [],
               'include_dirs'              : include_dirs}
mpm_module = compile_extension_module(**inst_params)

# conditional fix (issue #107) :
import ufl
ufl.algorithms.apply_derivatives.CONDITIONAL_WORKAROUND = True

for loader, name, is_pkg in pkgutil.walk_packages(__path__):
  module = loader.find_module(name).load_module(name)
  for name, value in inspect.getmembers(module):
    if name.startswith('__'):
      continue

    globals()[name] = value
    __all__.append(name)

from helper      import *
from model       import *
from material    import *
from gridmodel   import *

#mpm_module = MPMModelcpp.get_compile_cpp_code()



