cmake_minimum_required(VERSION 3.12...3.14)

project(fenics_mpm_pybind11)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(DOLFIN REQUIRED)
include(${DOLFIN_USE_FILE})

find_package(pybind11 REQUIRED CONFIG HINTS ${PYBIND11_DIR} ${PYBIND11_ROOT}
	$ENV{PYBIND11_DIR} $ENV{PYBIND11_ROOT})

find_package(fenics_mpm REQUIRED CONFIG HINTS ${FENICS_MPM_DIR}
	$ENV{FENICS_MPM_DIR})

pybind11_add_module(cpp MODULE src/fenics_mpm.cpp)

target_link_libraries(cpp PRIVATE ${fenics_mpm_LIBRARIES})
target_link_libraries(cpp PRIVATE dolfin)

target_include_directories(cpp PRIVATE ${EIGEN3_INCLUDE_DIR})
target_include_directories(cpp PRIVATE ${fenics_mpm_INCLUDE_DIR})

set(CMAKE_VERBOSE_MAKEFILE ON)



