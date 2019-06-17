import pkgutil
import inspect
import ufl
import matplotlib as mpl
#mpl.use('Agg')
mpl.rcParams['font.family']          = 'serif'
mpl.rcParams['legend.fontsize']      = 'medium'
mpl.rcParams['text.usetex']          = True
mpl.rcParams['text.latex.preamble']  = ['\\usepackage[mathscr]{euscript}']
#mpl.rcParams['contour.negative_linestyle']   = 'solid'

## open the cpp code :
#import os
#from   dolfin  import compile_cpp_code, parameters
#
##parameters['form_compiler']['cpp_optimize_flags'] = "-O3"
#
#cpp_src_dir     = os.path.dirname(os.path.abspath(__file__)) + "/cpp/"
#headers         = ["MPMMaterial.h",
#                   "MPMElasticMaterial.h",
#                   "MPMImpenetrableMaterial.h",
#                   "MPMModel.h"]
#code            = ''
#for header_file in headers:
#	header_file   = open(cpp_src_dir + header_file, "r")
#	code         += header_file.read()
#	header_file.close()
#
#module_name     = "MPMModelcpp"
#sources         = ["MPMMaterial.cpp",
#                   "MPMElasticMaterial.cpp",
#                   "MPMImpenetrableMaterial.cpp",
#                   "MPMModel.cpp"]
#include_dirs    = [".", cpp_src_dir]
#
## NOTE: what about:
##extra_libraries = kwargs.get("libraries", [])
##extra_library_dirs = kwargs.get("library_dirs", [])
#
## compile this with Instant JIT compiler :
#kwargs = {'cppargs'                   : '-O3 -fopenmp',
#          'include_dirs'              : include_dirs}
#mpm_module = compile_cpp_code(code, **kwargs)

# conditional fix (issue #107) :
ufl.algorithms.apply_derivatives.CONDITIONAL_WORKAROUND = True

from .helper      import *
from .model       import *
from .material    import *
from .gridmodel   import *



