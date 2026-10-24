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
target_compile_options(rrt_planner PRIVATE
  $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall -Wextra -Wpedantic>)

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
    tests/test_dubins.cpp
    tests/test_nearest_neighbors.cpp
    tests/test_rrt_star.cpp)
  target_link_libraries(rrt_tests PRIVATE rrt::planner GTest::gtest_main)
  include(GoogleTest)
  gtest_discover_tests(rrt_tests)
endif()
