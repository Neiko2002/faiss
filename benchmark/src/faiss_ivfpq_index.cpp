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
#include <string>

#include <sys/stat.h>
#include <sys/types.h>

#include <faiss/AutoTune.h>
#include <faiss/index_factory.h>
#include <faiss/index_io.h>

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

    // SIFT1M
    const auto data_path = std::filesystem::path("e:/Data/Feature/SIFT1M/");
    const auto learn_file       = (data_path / "SIFT1M" / "sift_learn.fvecs").string();
    const auto repository_file  = (data_path / "SIFT1M" / "sift_base.fvecs").string();
    const auto query_file       = (data_path / "SIFT1M" / "sift_query.fvecs").string();
    const auto groundtruth_file = (data_path / "SIFT1M" / "sift_groundtruth.ivecs").string();   
    const auto index_dir        = (data_path / "faiss").string();

    // https://github.com/facebookresearch/faiss/blob/main/demos/demo_ivfpq_indexing.cpp
    // int ncentroids = int(4 * sqrt(nb));

    // faiss index type
    // https://github.com/facebookresearch/faiss/wiki/bench_all_ivf_logs-deep1M
    // https://medium.com/@bb8s/embedding-based-retrieval-approximate-nearest-neighbor-algorithms-used-in-production-systems-b96dd4b2e9a3
    // auto index_type = "IVF1024,Flat";                // IVF nlist=1024
    auto index_type = "IVF1024,PQ64x4fs,RFlat";         // IVF nlist=1024
    // auto index_type = "IVF1024,PQ32x8";              // IVFPQ nlist=1024, ncodes=32, nbits=8
    // auto index_type = "IVF4096,Flat";                // IVF nlist=4096
    // auto index_type = "IVF4096,PQ32x8";              // IVFPQ nlist=4096, ncodes=32, nbits=8 (8 is default and max)
    // auto index_type = "IVF1024,PQ64x8";              // IVFPQ nlist=1024, ncodes=64, nbits=8
    // const auto index_type = "IVF2048,PQ32x8";        // IVFPQ nlist=2048, ncodes=8, nbits=8

    // auto index_type = "IVF1024,PQ64x4fs,Refine(SQfp16)"; // best for SIFT

    // how many percent of the data should be used to train the index
    const float train_percentage = 10;

    // find k best elements
    const auto target_k = 100;
    const auto k_recall_at = 100;

    // index file name
    const auto index_file = string_format("%s/%s,Train%4.1f.ivf", index_dir.c_str(), index_type, train_percentage);
        
    StopW stopwatch;
    faiss::Index* index;
    printf("[%lld s] Actual memory usage: %zu Mb, Max memory usage: %zu Mb\n", stopwatch.getElapsedTimeSeconds(), getCurrentRSS() / 1000000, getPeakRSS() / 1000000);

    size_t d;
    if(std::filesystem::exists(index_file.c_str()))
    {
        index = faiss::read_index(index_file.c_str());
    } 
    else 
    {

        // printf("[%lld s] Loading train set\n", stopwatch.getElapsedTimeSeconds());
        // size_t nt;
        // float* xt = fvecs_read(learn_file.c_str(), &d, &nt);

        printf("[%lld s] Loading database\n", stopwatch.getElapsedTimeSeconds());
        size_t nb;
        float* xb = fvecs_read(repository_file.c_str(), &d, &nb);
        printf("[%lld s] Actual memory usage: %zu Mb, Max memory usage: %zu Mb after loading data\n", stopwatch.getElapsedTimeSeconds(), getCurrentRSS() / 1000000, getPeakRSS() / 1000000);

        printf("[%lld s] Preparing index \"%s\" d=%zu\n", stopwatch.getElapsedTimeSeconds(), index_type, d);
        index = faiss::index_factory((int)d, index_type, faiss::METRIC_L2);
        printf("[%lld s] Actual memory usage: %zu Mb, Max memory usage: %zu Mb after creating the index\n", stopwatch.getElapsedTimeSeconds(), getCurrentRSS() / 1000000, getPeakRSS() / 1000000);

        auto train_size = size_t(nb * (train_percentage / 100));
        printf("[%lld s] Train database, size %zu*%zu\n", stopwatch.getElapsedTimeSeconds(), train_size, d);
        index->train(train_size, xb);
        // index->train(nt, xt);
        printf("[%lld s] Actual memory usage: %zu Mb, Max memory usage: %zu Mb after training the index\n", stopwatch.getElapsedTimeSeconds(), getCurrentRSS() / 1000000, getPeakRSS() / 1000000);

        printf("[%lld s] Indexing database, size %zu*%zu\n", stopwatch.getElapsedTimeSeconds(), nb, d);
        index->add(nb, xb);
        printf("[%lld s] Actual memory usage: %zu Mb, Max memory usage: %zu Mb after filling the index\n", stopwatch.getElapsedTimeSeconds(), getCurrentRSS() / 1000000, getPeakRSS() / 1000000);

        // store
        faiss::write_index(index, index_file.c_str());

        delete[] xb;
    }

    size_t nq;   // number of queries
    float* xq;   // query data
    {
        printf("[%lld s] Loading queries\n", stopwatch.getElapsedTimeSeconds());
        size_t d2;
        xq = fvecs_read(query_file.c_str(), &d2, &nq);
        assert(d == d2 || !"query does not have same dimension as train set");
        printf("[%lld s] Actual memory usage: %zu Mb, Max memory usage: %zu Mb after loading the query data\n", stopwatch.getElapsedTimeSeconds(), getCurrentRSS() / 1000000, getPeakRSS() / 1000000);
    }

    size_t k;         // nb of results per query in the GT
    faiss::idx_t* gt; // nq * k matrix of ground-truth nearest-neighbors
    {
        // load ground-truth and convert int to long
        printf("[%lld s] Loading ground truth for %zu queries\n", stopwatch.getElapsedTimeSeconds(), nq);
        size_t nq2;
        int* gt_int = ivecs_read(groundtruth_file.c_str(), &k, &nq2);
        assert(nq2 == nq || !"incorrect nb of ground truth entries");

        gt = new faiss::idx_t[k * nq];
        for (int i = 0; i < k * nq; i++)
            gt[i] = gt_int[i];

        delete[] gt_int;
        printf("[%lld s] Actual memory usage: %zu Mb, Max memory usage: %zu Mb after loading the ground truth data\n", stopwatch.getElapsedTimeSeconds(), getCurrentRSS() / 1000000, getPeakRSS() / 1000000);
    }

    { // Use the found configuration to perform a search

        // setup output buffers
        faiss::idx_t* I = new faiss::idx_t[nq * target_k];
        float* D = new float[nq * target_k];

        std::vector<float> nprobe_parameter = { 1, 2, 4, 8, 16, 32, 64, 128 }; 
        for (float nprobe : nprobe_parameter) {

            // https://github.com/facebookresearch/faiss/wiki/FAQ#what-does-it-mean-when-a-search-returns--1-ids
            faiss::ParameterSpace().set_index_parameter(index, "nprobe", nprobe);
            faiss::ParameterSpace().set_index_parameter(index, "k_factor_rf", 2);

            // search
            StopW timer;
            index->search(nq, xq, target_k, D, I);
            auto duration_us = timer.getElapsedTimeMicro();

            // evaluate results
            int recall_at_k = 0;
            for (int i = 0; i < nq; i++) {
                auto gt_nn = gt + i * k;
                auto result_nn = I + i * target_k;
                for (int m = 0; m < target_k; m++) {
                    auto result_entry = result_nn[m];

                    // check if result_entry is in the first k elements of the ground truth data
                    for (int j = 0; j < k_recall_at; j++) {  
                        if (gt_nn[j] == result_entry) {
                            recall_at_k++;
                            break;
                        }
                    }
                }
            }
            printf("%dR@%d = %0.4f with %6.f us/query at nprobe = %8.0f\n", k_recall_at, target_k, recall_at_k / float(nq) / k_recall_at, duration_us / float(nq), nprobe);
        }

        delete[] I;
        delete[] D;
    }

    delete[] xq;
    delete[] gt;
    delete index;
    return 0;
}