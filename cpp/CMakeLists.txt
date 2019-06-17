cmake_minimum_required(VERSION 3.12...3.14)

# require out-of-source builds :
file(TO_CMAKE_PATH "${PROJECT_BINARY_DIR}/CMakeLists.txt" LOC_PATH)
if(EXISTS "${LOC_PATH}")
	message(FATAL_ERROR "You cannot build in a source directory (or any directory with a CMakeLists.txt file). Please make a build subdirectory. Feel free to remove CMakeCache.txt and CMakeFiles.")
endif()

# packages used by this script :
include(GNUInstallDirs)            # for ${CMAKE_INSTALL_<some>DIR}
include(CMakePackageConfigHelpers) # for generating config files
include(CheckIPOSupported)         # for enabling inter-procedural opt
include(FeatureSummary)            # for printing target package info

find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
	set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
endif()

#if(CMAKE_VERSION VERSION_GREATER 3.6)
#	# Add clang-tidy if available
#	option(CLANG_TIDY_FIX "Perform fixes for Clang-Tidy" OFF)
#	find_program(
#		CLANG_TIDY_EXE
#		NAMES "clang-tidy"
#		DOC "Path to clang-tidy executable"
#	)
#
#	if(CLANG_TIDY_EXE)
#		if(CLANG_TIDY_FIX)
#			set(CMAKE_CXX_CLANG_TIDY "${CLANG_TIDY_EXE}" "-fix")
#		else()
#			set(CMAKE_CXX_CLANG_TIDY "${CLANG_TIDY_EXE}")
#		endif()
#	endif()
#endif()

# guard against bad build-type strings
if (NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE "Release")
endif()

string(TOLOWER "${CMAKE_BUILD_TYPE}" cmake_build_type_tolower)
if(    NOT cmake_build_type_tolower STREQUAL "debug"
   AND NOT cmake_build_type_tolower STREQUAL "release"
   AND NOT cmake_build_type_tolower STREQUAL "relwithdebinfo")
	message(FATAL_ERROR "Unknown build type \"${CMAKE_BUILD_TYPE}\". Allowed values are Debug, Release, RelWithDebInfo (case-insensitive).")
endif()

# this will set ${PROJECT_NAME}, among others :
project(
	FENICS_MPM
	VERSION 2019.0.1
	DESCRIPTION "material-point method"
	LANGUAGES CXX
)

# create a version header file :
configure_file(
	"${PROJECT_SOURCE_DIR}/Version.h.in"
	"${PROJECT_BINARY_DIR}/Version.h"
)

# get the list of sources and headers from the file system :
file(GLOB SOURCES CONFIGURE_DEPENDS "fenics_mpm/*.cpp")
file(GLOB HEADERS CONFIGURE_DEPENDS "fenics_mpm/*.h")

# create a new shared library called "FENICS_MPM" :
add_library(fenics_mpm
	SHARED
		${SOURCES} ${HEADERS}
)

# set project properties which will be useful later :
set_target_properties(
	fenics_mpm
	PROPERTIES
		VERSION   ${PROJECT_VERSION}
		SOVERSION ${PROJECT_VERSION}
		PUBLIC_HEADER "${HEADERS}"
)

check_ipo_supported(RESULT result)
if(result)
	set_target_properties(fenics_mpm
		PROPERTIES
			INTERPROCEDURAL_OPTIMIZATION TRUE
	)
endif()

target_compile_options(fenics_mpm PRIVATE -O3 -Wall -Wextra -Wpedantic)

target_compile_features(fenics_mpm PRIVATE cxx_std_11)


# ensure Eigen is in the path :
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
target_include_directories(fenics_mpm PRIVATE ${EIGEN3_INCLUDE_DIR})

ENABLE_LANGUAGE(C)

find_package(DOLFIN REQUIRED)
include(${DOLFIN_USE_FILE})

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
	target_link_libraries(fenics_mpm PRIVATE OpenMP::OpenMP_CXX)
endif()

set_package_properties(OpenMP
	PROPERTIES
		URL "http://www.openmp.org"
		DESCRIPTION "Parallel compiler directives"
		PURPOSE "partitions the particle operations between threads"
)

set_package_properties(DOLFIN
	PROPERTIES
		URL "https://fenicsproject.org/"
		DESCRIPTION "Finite element software libraray"
		PURPOSE "used to communicate particle information between material points"
)

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
	feature_summary(WHAT ENABLED_FEATURES DISABLED_FEATURES PACKAGES_FOUND)
	feature_summary(FILENAME ${CMAKE_CURRENT_BINARY_DIR}/features.log WHAT ALL)
endif()

install(
	TARGETS fenics_mpm
	EXPORT fenics_mpmTargets
	LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
	ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
	RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
	INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
	PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/fenics_mpm
)

install(
	EXPORT fenics_mpmTargets
	FILE fenics_mpmTargets.cmake
	NAMESPACE fenics_mpm::
	DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/cmake/fenics_mpm
)

set(FENICS_MPM_INCLUDE_DIR
	"${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_INCLUDEDIR}")
set(FENICS_MPM_LIBRARIES
	"${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}/libfenics_mpm.so")
set(FENICS_MPM_ROOT_DIR "${CMAKE_INSTALL_PREFIX}")

# generate the version file for the config file
write_basic_package_version_file(
	"${CMAKE_CURRENT_BINARY_DIR}/fenics_mpmConfigVersion.cmake"
	VERSION "${PROJECT_VERSION}"
	COMPATIBILITY AnyNewerVersion
)

# generate the config file that includes the exports
configure_package_config_file(
	${CMAKE_CURRENT_SOURCE_DIR}/fenics_mpmConfig.cmake.in
	${CMAKE_CURRENT_BINARY_DIR}/fenics_mpmConfig.cmake
	PATH_VARS FENICS_MPM_INCLUDE_DIR FENICS_MPM_ROOT_DIR FENICS_MPM_LIBRARIES
	INSTALL_DESTINATION "${CMAKE_INSTALL_DATAROOTDIR}/cmake/fenics_mpm"
	NO_SET_AND_CHECK_MACRO
	NO_CHECK_REQUIRED_COMPONENTS_MACRO
)

# install the configuration file
install(
	FILES
		"${CMAKE_CURRENT_BINARY_DIR}/fenics_mpmConfig.cmake"
		"${CMAKE_CURRENT_BINARY_DIR}/fenics_mpmConfigVersion.cmake"
	DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/cmake/fenics_mpm
)

# generate the export targets for the build tree
# needs to be after the install(TARGETS ) command
export(
	TARGETS fenics_mpm
	NAMESPACE fenics_mpm::
	FILE "${CMAKE_CURRENT_BINARY_DIR}/fenics_mpmTargets.cmake"
)
#export (PACKAGE fenics_mpm)



