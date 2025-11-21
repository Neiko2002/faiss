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

    // // SIFT1M
    // const auto data_path = std::filesystem::path("e:/Data/Feature/SIFT1M/");
    // const auto repository_file  = (data_path / "SIFT1M" / "sift_base.fvecs").string();
    // const auto query_file       = (data_path / "SIFT1M" / "sift_query.fvecs").string();
    // const auto groundtruth_file = (data_path / "SIFT1M" / "sift_groundtruth.ivecs").string();
    // const auto query_file       = (data_path / "SIFT1M" / "sift_explore_query.fvecs").string();
    // const auto groundtruth_file = (data_path / "SIFT1M" / "sift_explore_ground_truth.ivecs").string();

    // Msong
    const auto data_path = std::filesystem::path("e:\\Data\\Feature\\Msong\\");
    const auto repository_file  = (data_path / "msong" / "msong_base.fvecs").string();
    const auto query_file       = (data_path / "msong" / "msong_query.fvecs").string();
    const auto groundtruth_file = (data_path / "msong" / "msong_groundtruth.ivecs").string();

    // Deep1M
    // const auto data_path = std::filesystem::path("e:/Data/Feature/Deep1M/");
    // const auto repository_file  = (data_path / "deep1m" / "deep1m_base.fvecs").string();
    // // const auto query_file       = (data_path / "deep1m" / "deep1m_query.fvecs").string();
    // // const auto groundtruth_file = (data_path / "deep1m" / "deep1m_groundtruth.ivecs").string();
    // const auto query_file       = (data_path / "deep1m" / "deep1m_explore_query.fvecs").string();
    // const auto groundtruth_file = (data_path / "deep1m" / "deep1m_explore_ground_truth.ivecs").string();

    // // GloVe
    // const auto data_path        = std::filesystem::path("e:/Data/Feature/GloVe/");
    // const auto repository_file  = (data_path / "glove-100" / "glove-100_base.fvecs").string();
    // // const auto query_file       = (data_path / "glove-100" / "glove-100_query.fvecs").string();
    // // const auto groundtruth_file = (data_path / "glove-100" / "glove-100_groundtruth.ivecs").string();
    // const auto query_file       = (data_path / "glove-100" / "glove-100_explore_query.fvecs").string();
    // const auto groundtruth_file = (data_path / "glove-100" / "glove-100_explore_ground_truth.ivecs").string();

    // UQ-V
    // const auto data_path        = std::filesystem::path("e:/Data/Feature/UQ-V/");
    // const auto repository_file  = (data_path / "uqv/" / "uqv_base.fvecs").string();
    // const auto query_file       = (data_path / "uqv/" / "uqv_query.fvecs").string();
    // const auto groundtruth_file = (data_path / "uqv/" / "uqv_groundtruth.ivecs").string();

    // // Enron
    // const auto data_path        = std::filesystem::path("e:/Data/Feature/Enron/");
    // const auto repository_file  = (data_path / "enron/" / "enron_base.fvecs").string();
    // const auto query_file       = (data_path / "enron/" / "enron_query.fvecs").string();
    // const auto groundtruth_file = (data_path / "enron/" / "enron_groundtruth.ivecs").string();
    // const auto query_file       = (data_path / "enron" / "enron_explore_query.fvecs").string();
    // const auto groundtruth_file = (data_path / "enron" / "enron_explore_ground_truth.ivecs").string();

    // Audio
    // const auto data_path        = std::filesystem::path("e:/Data/Feature/Audio/");
    // const auto repository_file  = (data_path / "audio/" / "audio_base.fvecs").string();
    // // const auto query_file       = (data_path / "audio/" / "audio_query.fvecs").string();
    // // const auto groundtruth_file = (data_path / "audio/" / "audio_groundtruth.ivecs").string();
    // const auto query_file       = (data_path / "audio" / "audio_explore_query.fvecs").string();
    // const auto groundtruth_file = (data_path / "audio" / "audio_explore_ground_truth.ivecs").string();

    // pixabay clipfv=768D
    // const auto data_path        = std::filesystem::path("e:/Data/Feature/Pixabay/clipfv/");
    // const auto repository_file  = (data_path / "pixabay/" / "pixabay_clipfv_base.fvecs").string();
    // const auto query_file       = (data_path / "pixabay/" / "pixabay_clipfv_query.fvecs").string();
    // const auto groundtruth_file = (data_path / "pixabay/" / "pixabay_clipfv_groundtruth.ivecs").string();

    // pixabay gpret=1024D
    // const auto data_path        = std::filesystem::path("e:/Data/Feature/Pixabay/gpret/");
    // const auto repository_file  = (data_path / "pixabay/" / "pixabay_gpret_base.fvecs").string();
    // const auto query_file       = (data_path / "pixabay/" / "pixabay_gpret_query.fvecs").string();
    // const auto groundtruth_file = (data_path / "pixabay/" / "pixabay_gpret_groundtruth.ivecs").string();

    // faiss index type
    auto index_type = "Flat";

    // in order to get 95% precision we reduce the index by 5 percent
    float reduce_index_by = 61; // 5 percent

    StopW stopwatch;
    faiss::Index* index;
    printf("[%lld s] Actual memory usage: %zu Mb, Max memory usage: %zu Mb\n", stopwatch.getElapsedTimeSeconds(), getCurrentRSS() / 1000000, getPeakRSS() / 1000000);

    size_t d;
    {
        printf("[%lld s] Loading database\n", stopwatch.getElapsedTimeSeconds());
        size_t nb;
        float* xb = fvecs_read(repository_file.c_str(), &d, &nb);
        printf("[%lld s] Actual memory usage: %zu Mb, Max memory usage: %zu Mb after loading data\n", stopwatch.getElapsedTimeSeconds(), getCurrentRSS() / 1000000, getPeakRSS() / 1000000);

        // reduce data set size
        auto reduce_count = size_t(nb * (reduce_index_by / 100));
        nb = nb - reduce_count;

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

    size_t k;                   // nb of results per query in the GT
    faiss::idx_t* gt;    // nq * k matrix of ground-truth nearest-neighbors
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
        printf("[%lld s] Setup search structures\n", stopwatch.getElapsedTimeSeconds()); 
        faiss::idx_t* I = new faiss::idx_t[nq * k];
        float* D = new float[nq * k];
        printf("[%lld s] Actual memory usage: %zu Mb, Max memory usage: %zu Mb after setting up the search output structures\n", stopwatch.getElapsedTimeSeconds(), getCurrentRSS() / 1000000, getPeakRSS() / 1000000);

        // search
        printf("[%lld s] Perform TOP%zu searches on %zu queries\n", stopwatch.getElapsedTimeSeconds(), k, nq); 
        StopW timer;
        index->search(nq, xq, k, D, I);
        auto duration_us = timer.getElapsedTimeMicro();
        printf("[%lld s] Actual memory usage: %zu Mb, Max memory usage: %zu Mb after performing the search\n", stopwatch.getElapsedTimeSeconds(), getCurrentRSS() / 1000000, getPeakRSS() / 1000000);

        // evaluate results
        int precisionAtK = 0;
        for (int i = 0; i < nq; i++) {
            auto gt_nn = gt + i * k;
            auto result_nn = I + i * k;
            for (int m = 0; m < k; m++) {
                auto result_entry = result_nn[m];

                // check if result_entry is in the first k elements of the ground truth data
                for (int j = 0; j < k; j++) {     
                    if (gt_nn[j] == result_entry) {
                        precisionAtK++;
                        break;
                    }
                }
            }
        }
        printf("R@%d = %.4f with %8.4f us/query\n", k, precisionAtK / float(nq) / k, duration_us / float(nq));

        delete[] I;
        delete[] D;
    }

    delete[] xq;
    delete[] gt;
    delete index;
    return 0;
}