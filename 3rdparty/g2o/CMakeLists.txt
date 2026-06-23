# Vendored g2o — graph-based non-linear least-squares optimizer.
# Builds as a static library linked into any OpenCV module that needs it.
# Requires Eigen3 (found by the parent OpenCV build via EIGEN_INCLUDE_PATH).

if(NOT HAVE_EIGEN)
  message(STATUS "G2O: Eigen not found — g2o will not be built")
  set(HAVE_G2O 0 CACHE INTERNAL "")
  return()
endif()

set(G2O_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/g2o")

# Generate config.h into the source directory; g2o sources reference it as
# "../../config.h" (relative from g2o/core/ etc.), so it must live there.
set(G2O_CXX_COMPILER "${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER}")
configure_file(config.h.in "${CMAKE_CURRENT_SOURCE_DIR}/config.h")

set(g2o_srcs
  # types
  ${G2O_SOURCE_DIR}/types/types_sba.cpp
  ${G2O_SOURCE_DIR}/types/types_six_dof_expmap.cpp
  ${G2O_SOURCE_DIR}/types/types_seven_dof_expmap.cpp
  # core
  ${G2O_SOURCE_DIR}/core/hyper_graph_action.cpp
  ${G2O_SOURCE_DIR}/core/hyper_graph.cpp
  ${G2O_SOURCE_DIR}/core/marginal_covariance_cholesky.cpp
  ${G2O_SOURCE_DIR}/core/matrix_structure.cpp
  ${G2O_SOURCE_DIR}/core/batch_stats.cpp
  ${G2O_SOURCE_DIR}/core/parameter.cpp
  ${G2O_SOURCE_DIR}/core/cache.cpp
  ${G2O_SOURCE_DIR}/core/optimizable_graph.cpp
  ${G2O_SOURCE_DIR}/core/solver.cpp
  ${G2O_SOURCE_DIR}/core/optimization_algorithm_factory.cpp
  ${G2O_SOURCE_DIR}/core/estimate_propagator.cpp
  ${G2O_SOURCE_DIR}/core/factory.cpp
  ${G2O_SOURCE_DIR}/core/sparse_optimizer.cpp
  ${G2O_SOURCE_DIR}/core/hyper_dijkstra.cpp
  ${G2O_SOURCE_DIR}/core/parameter_container.cpp
  ${G2O_SOURCE_DIR}/core/optimization_algorithm.cpp
  ${G2O_SOURCE_DIR}/core/optimization_algorithm_with_hessian.cpp
  ${G2O_SOURCE_DIR}/core/optimization_algorithm_levenberg.cpp
  ${G2O_SOURCE_DIR}/core/optimization_algorithm_gauss_newton.cpp
  ${G2O_SOURCE_DIR}/core/jacobian_workspace.cpp
  ${G2O_SOURCE_DIR}/core/robust_kernel.cpp
  ${G2O_SOURCE_DIR}/core/robust_kernel_factory.cpp
  ${G2O_SOURCE_DIR}/core/robust_kernel_impl.cpp
  # stuff
  ${G2O_SOURCE_DIR}/stuff/timeutil.cpp
  ${G2O_SOURCE_DIR}/stuff/os_specific.c
  ${G2O_SOURCE_DIR}/stuff/string_tools.cpp
  ${G2O_SOURCE_DIR}/stuff/property.cpp
)

add_library(g2o STATIC ${g2o_srcs})

target_include_directories(g2o
  PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${G2O_SOURCE_DIR}/core"
    "${G2O_SOURCE_DIR}/types"
    "${G2O_SOURCE_DIR}/stuff"
)

# Eigen3 may be available as an imported target (conda/modern CMake) or as a path.
if(TARGET Eigen3::Eigen)
  target_link_libraries(g2o PUBLIC Eigen3::Eigen)
elseif(EIGEN_INCLUDE_PATH)
  target_include_directories(g2o PUBLIC "${EIGEN_INCLUDE_PATH}")
endif()

if(UNIX)
  target_compile_definitions(g2o PRIVATE UNIX)
endif()

if(MSVC90)
  target_compile_definitions(g2o PRIVATE EIGEN_DONT_ALIGN_STATICALLY=1)
endif()

set_target_properties(g2o PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  FOLDER "3rdparty"
)

# Suppress all warnings for this 3rdparty target only.
target_compile_options(g2o PRIVATE
  $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-w>
  $<$<CXX_COMPILER_ID:MSVC>:/w>
)



# Export variables for the parent build to consume.
set(G2O_INCLUDE_DIRS
  "${CMAKE_CURRENT_SOURCE_DIR}"
  "${G2O_SOURCE_DIR}/core"
  "${G2O_SOURCE_DIR}/types"
  "${G2O_SOURCE_DIR}/stuff"
  CACHE INTERNAL "g2o include directories"
)
set(G2O_LIBRARIES g2o CACHE INTERNAL "g2o library target")
set(HAVE_G2O 1 CACHE INTERNAL "g2o is available")

if(TARGET Eigen3::Eigen)
  message(STATUS "G2O: build enabled (Eigen: Eigen3::Eigen target)")
else()
  message(STATUS "G2O: build enabled (Eigen: ${EIGEN_INCLUDE_PATH})")
endif()
