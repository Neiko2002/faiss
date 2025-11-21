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
    if (ec != std::error_code{})
    {
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
        omp_set_num_threads(1); // Use 1 threads for all consecutive parallel regions

        std::cout << "_OPENMP " << omp_get_num_threads() << " threads" << std::endl;
    #endif

    // Sift1M
    const auto data_path = std::filesystem::path("e:\\Data\\Feature\\SIFT1M\\");
    const auto repository_file  = (data_path / "SIFT1M" / "sift_base.fvecs").string();
    const auto query_file       = (data_path / "SIFT1M" / "sift_query.fvecs").string();
    const auto groundtruth_file = (data_path / "SIFT1M" / "sift_groundtruth_top1024.ivecs").string();
    size_t k = 1024; // default k
    size_t bach = 100000; // default batch size

    // faiss index type
    auto index_type = "Flat";

    StopW stopwatch;
    faiss::Index* index;
    printf("[%lld s] Actual memory usage: %zu Mb, Max memory usage: %zu Mb\n", stopwatch.getElapsedTimeSeconds(), getCurrentRSS() / 1000000, getPeakRSS() / 1000000);

    size_t d;
    {
        printf("[%lld s] Loading database\n", stopwatch.getElapsedTimeSeconds());
        size_t nb;
        float* xb = fvecs_read(repository_file.c_str(), &d, &nb);
        printf("[%lld s] Actual memory usage: %zu Mb, Max memory usage: %zu Mb after loading data\n", stopwatch.getElapsedTimeSeconds(), getCurrentRSS() / 1000000, getPeakRSS() / 1000000);

        printf("[%lld s] Preparing index \"%s\" d=%zu\n", stopwatch.getElapsedTimeSeconds(), index_type, d);
        index = faiss::index_factory((int)d, index_type, faiss::METRIC_L2);
        printf("[%lld s] Actual memory usage: %zu Mb, Max memory usage: %zu Mb after creating the index\n", stopwatch.getElapsedTimeSeconds(), getCurrentRSS() / 1000000, getPeakRSS() / 1000000);

        printf("[%lld s] Indexing database, size %zu*%zu\n", stopwatch.getElapsedTimeSeconds(), nb, d);
        index->add(nb, xb);
        printf("[%lld s] Actual memory usage: %zu Mb, Max memory usage: %zu Mb after filling the index\n", stopwatch.getElapsedTimeSeconds(), getCurrentRSS() / 1000000, getPeakRSS() / 1000000);

        delete[] xb;
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
        printf("[%lld s] Computing ground truth for %zu queries with k=%zu\n", stopwatch.getElapsedTimeSeconds(), nq, k);
        faiss::idx_t* I = new faiss::idx_t[nq * k];
        float* D = new float[nq * k];

        index->search(nq, xq, k, D, I);

        // Convert 0-based faiss ids to 1-based ids if needed
        std::vector<int> gt_ids(nq * k);
        for (size_t i = 0; i < nq * k; ++i) {
            gt_ids[i] = static_cast<int>(I[i]) + 0; // +0 if you want 0-based
        }

        printf("[%lld s] Writing ground truth to %s\n", stopwatch.getElapsedTimeSeconds(), groundtruth_file.c_str());
        ivecs_write(groundtruth_file.c_str(), (int)k, (int)nq, gt_ids.data());
        delete[] I;
        delete[] D;
    }

    delete[] xq;
    delete index;
    return 0;
}