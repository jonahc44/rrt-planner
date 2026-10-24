cmake_minimum_required(VERSION 3.16)
project(rrt_planner VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(RRT_BUILD_TESTS "Build unit tests" ON)
option(RRT_BUILD_EXAMPLES "Build example programs" ON)
option(RRT_BUILD_PYTHON "Build the pybind11 Python module" OFF)

include(FetchContent)

# ---- nanoflann (header-only KD-tree) ----
find_package(nanoflann 1.5 QUIET)
if(NOT nanoflann_FOUND)
  FetchContent_Declare(nanoflann
    GIT_REPOSITORY https://github.com/jlblancoc/nanoflann.git
    GIT_TAG v1.7.1
    GIT_SHALLOW TRUE)
  set(NANOFLANN_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
  set(NANOFLANN_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(nanoflann)
endif()

# ---- library ----
add_library(rrt_planner
  src/state_space.cpp
  src/collision_checker.cpp
  src/real_vector_2d_space.cpp
  src/dubins.cpp
  src/dubins_state_space.cpp
  src/geometric_world_2d.cpp
  src/occupancy_grid.cpp
  src/costmap_cost.cpp
  src/nearest_neighbors.cpp
  src/rrt_star.cpp)
add_library(rrt::planner ALIAS rrt_planner)
target_include_directories(rrt_planner PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_link_libraries(rrt_planner PRIVATE nanoflann::nanoflann)
# The Python extension links this static library into a shared object.
set_target_properties(rrt_planner PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_options(rrt_planner PRIVATE
  $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall -Wextra -Wpedantic>)

if(RRT_BUILD_PYTHON)
  find_package(Python 3.9 REQUIRED COMPONENTS Interpreter Development.Module)
  # Ask the active interpreter where pybind11 lives (pip install pybind11).
  if(NOT pybind11_DIR)
    execute_process(
      COMMAND "${Python_EXECUTABLE}" -m pybind11 --cmakedir
      OUTPUT_VARIABLE pybind11_DIR OUTPUT_STRIP_TRAILING_WHITESPACE
      ERROR_QUIET)
  endif()
  find_package(pybind11 3.0 CONFIG REQUIRED)
  pybind11_add_module(_core python/bindings.cpp)
  target_link_libraries(_core PRIVATE rrt::planner)
  if(SKBUILD)
    install(TARGETS _core LIBRARY DESTINATION rrt_planner)
  else()
    # Developer build: drop the module next to the Python sources so
    # `PYTHONPATH=python` picks it up without installing.
    set_target_properties(_core PROPERTIES
      LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/python/rrt_planner)
  endif()
endif()

if(RRT_BUILD_EXAMPLES)
  foreach(ex plan_r2 plan_dubins)
    add_executable(${ex} examples/${ex}.cpp)
    target_link_libraries(${ex} PRIVATE rrt::planner)
  endforeach()
endif()

if(RRT_BUILD_TESTS)
  enable_testing()
  find_package(GTest QUIET)
  if(NOT GTest_FOUND)
    FetchContent_Declare(googletest
      GIT_REPOSITORY https://github.com/google/googletest.git
      GIT_TAG v1.15.2
      GIT_SHALLOW TRUE)
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
  endif()
  add_executable(rrt_tests
    tests/test_collision.cpp
    tests/test_dubins.cpp
    tests/test_instrumentation.cpp
    tests/test_nearest_neighbors.cpp
    tests/test_rrt_star.cpp
    tests/test_steering.cpp)
  target_link_libraries(rrt_tests PRIVATE rrt::planner GTest::gtest_main)
  include(GoogleTest)
  gtest_discover_tests(rrt_tests)
endif()
