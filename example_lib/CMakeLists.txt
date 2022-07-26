cmake_minimum_required (VERSION 3.19)

project(example_lib)

# setup the example lib
# http://mariobadr.com/creating-a-header-only-library-with-cmake.html
add_library(EXAMPLE_LIB INTERFACE)
target_include_directories(EXAMPLE_LIB INTERFACE ${PROJECT_SOURCE_DIR}/include)
target_compile_features(EXAMPLE_LIB INTERFACE cxx_std_20)
