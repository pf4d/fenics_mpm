# open the cpp code :
import os
from   dolfin  import compile_extension_module

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
                   "MsdfPMImpenetrableMaterial.cpp",
                   "MPMModel.cpp"]
source_dir      = cpp_src_dir
include_dirs    = [".", cpp_src_dir] 
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

def get_compile_cpp_code():
  return compile_extension_module(**inst_params)


