cmake_minimum_required(VERSION 3.19)

project(benchmark)
include_directories(${PROJECT_SOURCE_DIR}/include)

add_executable(faiss_benchmark EXCLUDE_FROM_ALL ${PROJECT_SOURCE_DIR}/src/faiss_benchmark.cpp)
target_link_libraries(faiss_benchmark PRIVATE compile-options)

add_executable(faiss_index EXCLUDE_FROM_ALL ${PROJECT_SOURCE_DIR}/src/faiss_index.cpp)
target_link_libraries(faiss_index PRIVATE compile-options)

add_executable(faiss_flat_index EXCLUDE_FROM_ALL ${PROJECT_SOURCE_DIR}/src/faiss_flat_index.cpp)
target_link_libraries(faiss_flat_index PRIVATE compile-options)

add_executable(faiss_flat_index_compute_gt EXCLUDE_FROM_ALL ${PROJECT_SOURCE_DIR}/src/faiss_flat_index_compute_gt.cpp)
target_link_libraries(faiss_flat_index_compute_gt PRIVATE compile-options)

add_executable(faiss_ivfpq_index EXCLUDE_FROM_ALL ${PROJECT_SOURCE_DIR}/src/faiss_ivfpq_index.cpp)
target_link_libraries(faiss_ivfpq_index PRIVATE compile-options)

add_executable(faiss_fastscan_index EXCLUDE_FROM_ALL ${PROJECT_SOURCE_DIR}/src/faiss_fastscan_index.cpp)
target_link_libraries(faiss_fastscan_index PRIVATE compile-options)