cmake_minimum_required(VERSION 3.19)

project(benchmark)
include_directories(${PROJECT_SOURCE_DIR}/include)

add_executable(faiss_build_and_test EXCLUDE_FROM_ALL ${PROJECT_SOURCE_DIR}/src/faiss_build_and_test.cpp)
target_link_libraries(faiss_build_and_test PRIVATE compile-options)
target_compile_definitions(faiss_build_and_test PRIVATE DATA_PATH="${DATA_PATH}")
