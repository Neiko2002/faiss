cmake_minimum_required(VERSION 3.19)

project(benchmark)
include_directories(${PROJECT_SOURCE_DIR}/include)

add_executable(bench_main ${PROJECT_SOURCE_DIR}/src/benchmark_main.cpp)
target_link_libraries(bench_main PUBLIC EXAMPLE_LIB compile-options)