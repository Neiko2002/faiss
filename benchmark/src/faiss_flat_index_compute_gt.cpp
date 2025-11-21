/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 * 
 * https://raw.githubusercontent.com/facebookresearch/faiss/main/demos/demo_sift1M.cpp
 */

#include <filesystem>
#include <iostream>
#include <fstream>
#include <vector>
#include <unordered_set>
#include <algorithm>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <sys/stat.h>
#include <sys/types.h>

#include <faiss/AutoTune.h>
#include <faiss/index_factory.h>

#include "stopwatch.h"

/**
 * To run this demo, please download the ANN_SIFT1M dataset from
 *
 *   http://corpus-texmex.irisa.fr/
 *
 * and unzip it to the sudirectory sift1M.
 **/

/*****************************************************
 * I/O functions for fvecs and ivecs
 *****************************************************/
float* fvecs_read(const char* fname, size_t* d_out, size_t* n_out) {
    std::error_code ec{};
    auto file_size = std::filesystem::file_size(fname, ec);
    if (ec != std::error_code{}) {
        std::cerr << "error when accessing file " << fname << ", size is: " << file_size << " message: " << ec.message() << std::endl;
        perror("");
        abort();
    }

    auto ifstream = std::ifstream(fname, std::ios::binary);
    if (!ifstream.is_open())
    {
        std::cerr << "could not open " << fname << std::endl;
        perror("");
        abort();
    }

    int dims;
    ifstream.read(reinterpret_cast<char*>(&dims), sizeof(int));
    assert((dims > 0 && dims < 1000000) || !"unreasonable dimension");
    assert(file_size % ((dims + 1) * 4) == 0 || !"weird file size");
    size_t n = file_size / ((dims + 1) * 4);

    *d_out = dims;
    *n_out = n;

    float* x = new float[n * (dims + 1)];
    ifstream.seekg(0);
    ifstream.read(reinterpret_cast<char*>(x), n * (dims + 1) * sizeof(float));
    if (!ifstream) assert(ifstream.gcount() == static_cast<int>(n * (dims + 1)) || !"could not read whole file");

    // shift array to remove row headers
    for (size_t i = 0; i < n; i++) std::memmove(&x[i * dims], &x[1 + i * (dims + 1)], dims * sizeof(float));

    ifstream.close();
    return x;
}

// not very clean, but works as long as sizeof(int) == sizeof(float)
int* ivecs_read(const char* fname, size_t* d_out, size_t* n_out) {
    return (int*)fvecs_read(fname, d_out, n_out);
}

void ivecs_write (const char *fname, int d, int n, const int *v)
{
    auto ofstream = std::ofstream(fname, std::ios::binary);
    if (!ofstream.is_open())
    {
        std::cerr << "could not open " << fname << std::endl;
        perror("");
        abort();
    }

    for (size_t i = 0; i < n; i++) {
        ofstream.write(reinterpret_cast<char*>(&d), sizeof(int));
        ofstream.write(const_cast<char*>(reinterpret_cast<const char*>(v)), d * sizeof(int));
        v += d;
    }

    ofstream.close();
}


int main() {

    #if defined(__AVX2__)
        std::cout << "use AVX2  ..." << std::endl;
    #elif defined(__AVX__)
        std::cout << "use AVX  ..." << std::endl;
    #elif defined(__SSE2__)
        std::cout << "use SSE  ..." << std::endl;
    #else
        std::cout << "use arch  ..." << std::endl;
    #endif

    // https://github.com/facebookresearch/faiss/wiki/Threads-and-asynchronous-calls
    #ifdef _OPENMP
        omp_set_dynamic(0);     // Explicitly disable dynamic teams
        omp_set_num_threads(4); // Use 1 threads for all consecutive parallel regions

        std::cout << "_OPENMP " << omp_get_num_threads() << " threads" << std::endl;
    #endif

    // glove
    const auto data_path = std::filesystem::path("e:\\Data\\Feature\\GloVe\\");
    const auto repository_file  = (data_path / "glove-100" / "glove-100_base.fvecs").string();
    const auto query_file       = (data_path / "glove-100" / "glove-100_query.fvecs").string();
    const auto groundtruth_base = (data_path / "glove-100" / "glove-100_groundtruth").string();

    // default parameters
    size_t k = 1024;       // top-k
    size_t step_size = 100000; // how many base vectors to add per step

    // faiss index type
    auto index_type = "Flat";

    StopW stopwatch;
    faiss::Index* index;
    printf("[%lld s] Actual memory usage: %zu Mb, Max memory usage: %zu Mb\n", stopwatch.getElapsedTimeSeconds(), getCurrentRSS() / 1000000, getPeakRSS() / 1000000);

    size_t d;
    // keep base dataset and its size in scope for the ground-truth loop
    size_t nb_total;
    float* xb_full;
    {
        printf("[%lld s] Loading database\n", stopwatch.getElapsedTimeSeconds());
        xb_full = fvecs_read(repository_file.c_str(), &d, &nb_total);
        printf("[%lld s] Actual memory usage: %zu Mb, Max memory usage: %zu Mb after loading data\n", stopwatch.getElapsedTimeSeconds(), getCurrentRSS() / 1000000, getPeakRSS() / 1000000);

        printf("[%lld s] Preparing index \"%s\" d=%zu\n", stopwatch.getElapsedTimeSeconds(), index_type, d);
        index = faiss::index_factory((int)d, index_type, faiss::METRIC_L2);
        printf("[%lld s] Actual memory usage: %zu Mb, Max memory usage: %zu Mb after creating the index\n", stopwatch.getElapsedTimeSeconds(), getCurrentRSS() / 1000000, getPeakRSS() / 1000000);

        // We will add the base vectors in chunks of step_size later below.
        // Keep xb_full alive for the whole run.
    }

    size_t nq;
    float* xq;
    {
        printf("[%lld s] Loading queries\n", stopwatch.getElapsedTimeSeconds());
        size_t d2;
        xq = fvecs_read(query_file.c_str(), &d2, &nq);
        assert(d == d2 || !"query does not have same dimension as train set");
        printf("[%lld s] Actual memory usage: %zu Mb, Max memory usage: %zu Mb after loading the query data\n", stopwatch.getElapsedTimeSeconds(), getCurrentRSS() / 1000000, getPeakRSS() / 1000000);
    }


    {
        // For each base size step, extend the index and compute ground truth.
        size_t nb_current = 0;
        size_t step_idx   = 0;

        while (nb_current < nb_total) {
            size_t nb_next = std::min(nb_current + step_size, nb_total);
            size_t to_add  = nb_next - nb_current;

            printf("[%lld s] Step %zu: adding base vectors [%zu, %zu) (count=%zu)\n",
                   stopwatch.getElapsedTimeSeconds(), step_idx, nb_current, nb_next, to_add);

            index->add(to_add, xb_full + nb_current * d);

            printf("[%lld s] Computing ground truth for %zu queries with k=%zu on nb=%zu base vectors\n",
                   stopwatch.getElapsedTimeSeconds(), nq, k, nb_next);

            faiss::idx_t* I = new faiss::idx_t[nq * k];
            float* D = new float[nq * k];

            index->search(nq, xq, k, D, I);

            std::vector<int> gt_ids(nq * k);
            for (size_t i = 0; i < nq * k; ++i) {
                gt_ids[i] = static_cast<int>(I[i]); // keep 0-based
            }

            // ground truth filename based on the base name, k and current base size
            auto gt_filename = groundtruth_base + "_top" + std::to_string(k) +
                               "_nb" + std::to_string(nb_next) + ".ivecs";

            printf("[%lld s] Writing ground truth to %s\n", stopwatch.getElapsedTimeSeconds(), gt_filename.c_str());
            ivecs_write(gt_filename.c_str(), (int)k, (int)nq, gt_ids.data());

            delete[] I;
            delete[] D;

            nb_current = nb_next;
            ++step_idx;
        }
    }

    delete[] xq;
    delete[] xb_full;
    delete index;
    return 0;
}