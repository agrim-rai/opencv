set(the_description "SLAM and Visual Odometry")

ocv_define_module(slam
  opencv_3d
  opencv_features
  opencv_imgproc
  opencv_imgcodecs
  opencv_video
  OPTIONAL
  opencv_dnn
  WRAP python)

# ---------------------------------------------------------------------------
# Optional: g2o for real pose-only bundle adjustment (PoseOptimization).
# Install via:  conda install -c conda-forge g2o
# CMake picks it up automatically when the conda env is active and
# CMAKE_PREFIX_PATH contains $CONDA_PREFIX (01_build.sh handles this).
# ---------------------------------------------------------------------------
if(BUILD_opencv_slam)
  find_package(g2o QUIET)
  if(g2o_FOUND)
    message(STATUS "opencv_slam: g2o ${g2o_VERSION} found — real PoseOptimization enabled")

    # Collect whichever component targets exist (name varies by g2o version).
    set(_g2o_targets "")
    foreach(_t core stuff types_sba solver_dense solver_eigen)
      if(TARGET g2o::${_t})
        list(APPEND _g2o_targets g2o::${_t})
      endif()
    endforeach()

    if(_g2o_targets)
      target_link_libraries(opencv_slam PRIVATE ${_g2o_targets})
      target_compile_definitions(opencv_slam PRIVATE HAVE_G2O)
      message(STATUS "opencv_slam: linking g2o targets: ${_g2o_targets}")
    else()
      message(WARNING "opencv_slam: g2o found but no expected CMake targets — stub used")
    endif()
    unset(_g2o_targets)
  else()
    message(STATUS "opencv_slam: g2o not found — reprojection stub active")
    message(STATUS "  Enable: conda install -c conda-forge g2o")
    message(STATUS "  Then:   cmake -DCMAKE_PREFIX_PATH=\$CONDA_PREFIX ... (handled by 01_build.sh)")
  endif()
endif()
