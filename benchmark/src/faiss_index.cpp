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

#include "stopwatch.h"

#include <faiss/AutoTune.h>
#include <faiss/index_factory.h>
#include <faiss/index_io.h>

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

/**
 * https://github.com/facebookresearch/faiss/blob/main/demos/demo_sift1M.cpp
 */
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

    //SIFT1M
    const auto data_path = std::filesystem::path("e:/Data/Feature/SIFT1M/");
    const auto learn_file       = (data_path / "SIFT1M" / "sift_learn.fvecs").string();
    const auto repository_file  = (data_path / "SIFT1M" / "sift_base.fvecs").string();
    const auto query_file       = (data_path / "SIFT1M" / "sift_query.fvecs").string();
    const auto groundtruth_file = (data_path / "SIFT1M" / "sift_groundtruth.ivecs").string();   
    const auto index_dir        = (data_path / "faiss").string();

    // find k best elements
    const auto k_recall_at = 1;

    // Select index type
    // https://github.com/facebookresearch/faiss/wiki/The-index-factory
    
    // this is typically the fastest one.
    // const char* index_key = "IVF4096,Flat";
    // const char *index_key = "IVF1024,Flat";                // IVF nlist=1024

    // these ones have better memory usage
    // const char *index_key = "Flat";
    // const char *index_key = "PQ32";            // code_size=32, nbit=8
    // const char *index_key = "PCA80,Flat";
    // const char *index_key = "IVF4096,PQ8+16";  // ncentroids=4096, code_size=8, nbit=16
    // const char *index_key = "IVF4096,PQ32";    // IndexIVFPQ: ncentroids=4096, code_size=32, nbit=8
    // const char *index_key = "IMI2x8,PQ32";
    // const char *index_key = "IMI2x8,PQ8+16";
    // const char *index_key = "OPQ16_64,IMI2x8,PQ8+16";

    // https://github.com/facebookresearch/faiss/wiki/Indexing-1M-vectors#preliminary-experiment-ivf-re-ranking
    // const char *index_key = "IVF1024,SQ8"; 
    const char *index_key = "IVF1024,PQ64x4fs,Refine(SQfp16)"; // best for SIFT
    // const char *index_key = "IVF1024,PQ504x4fsr,Refine(SQfp16)"; // best for GloVe

    // index file name
    const auto index_file = string_format("%s/%s_traindata.ivf", index_dir.c_str(), index_key);
        
    StopW stopwatch;
    faiss::Index* index;
    printf("[%lld s] Actual memory usage: %zu Mb, Max memory usage: %zu Mb\n", stopwatch.getElapsedTimeSeconds(), getCurrentRSS() / 1000000, getPeakRSS() / 1000000);

    // load train data and train index
    size_t d;
    if(std::filesystem::exists(learn_file.c_str()))
    {
        printf("[%lld s] Loading train set\n", stopwatch.getElapsedTimeSeconds());

        size_t nt;
        float* xt = fvecs_read(learn_file.c_str(), &d, &nt);

        printf("[%lld s] Preparing index \"%s\" d=%zu\n",
               stopwatch.getElapsedTimeSeconds(),
               index_key,
               d);
        index = faiss::index_factory((int)d, index_key, faiss::METRIC_L2);

        printf("[%lld s] Training on %zu vectors\n", stopwatch.getElapsedTimeSeconds(), nt);

        index->train(nt, xt);
        delete[] xt;
    }

    // load dataset and build index
    {
        printf("[%lld s] Loading database\n", stopwatch.getElapsedTimeSeconds());

        size_t nb, d2;
        float* xb = fvecs_read(repository_file.c_str(), &d2, &nb);
        assert(d == d2 || !"dataset does not have same dimension as train set");
        printf("[%lld s] Actual memory usage: %zu Mb, Max memory usage: %zu Mb after loading data\n", stopwatch.getElapsedTimeSeconds(), getCurrentRSS() / 1000000, getPeakRSS() / 1000000);

        printf("[%lld s] Actual memory usage: %zu Mb, Max memory usage: %zu Mb after creating the index\n", stopwatch.getElapsedTimeSeconds(), getCurrentRSS() / 1000000, getPeakRSS() / 1000000);

        printf("[%lld s] Indexing database, size %zu*%zu\n",
               stopwatch.getElapsedTimeSeconds(),
               nb,
               d);

        index->add(nb, xb);
        printf("[%lld s] Actual memory usage: %zu Mb, Max memory usage: %zu Mb after filling the index\n", stopwatch.getElapsedTimeSeconds(), getCurrentRSS() / 1000000, getPeakRSS() / 1000000);

        // store
        faiss::write_index(index, index_file.c_str());

        delete[] xb;
    }

    // load queries
    size_t nq;
    float* xq;
    {
        printf("[%lld s] Loading queries\n", stopwatch.getElapsedTimeSeconds());

        size_t d2;
        xq = fvecs_read(query_file.c_str(), &d2, &nq);
        assert(d == d2 || !"query does not have same dimension as train set");
        printf("[%lld s] Actual memory usage: %zu Mb, Max memory usage: %zu Mb after loading the query data\n", stopwatch.getElapsedTimeSeconds(), getCurrentRSS() / 1000000, getPeakRSS() / 1000000);
    }

    // load ground truth
    size_t k;         // nb of results per query in the GT
    faiss::idx_t* gt; // nq * k matrix of ground-truth nearest-neighbors
    {
        printf("[%lld s] Loading ground truth for %zu queries\n",
               stopwatch.getElapsedTimeSeconds(),
               nq);

        // load ground-truth and convert int to long
        size_t nq2;
        int* gt_int = ivecs_read(groundtruth_file.c_str(), &k, &nq2);
        assert(nq2 == nq || !"incorrect nb of ground truth entries");

        gt = new faiss::idx_t[k * nq];
        for (int i = 0; i < k * nq; i++) {
            gt[i] = gt_int[i];
        }
        delete[] gt_int;
        printf("[%lld s] Actual memory usage: %zu Mb, Max memory usage: %zu Mb after loading the ground truth data\n", stopwatch.getElapsedTimeSeconds(), getCurrentRSS() / 1000000, getPeakRSS() / 1000000);
    }

    // run auto-tuning finds good nprobe and hamming threshold for an efficent search
    faiss::OperatingPoints ops; // Result of the auto-tuning
    //  std::string selected_params; // Result of the auto-tuning
    { 
        printf("[%lld s] Preparing auto-tune criterion 1-recall at 1 "
               "criterion, with k=%zu nq=%zu\n",
               stopwatch.getElapsedTimeSeconds(),
               k,
               nq);

        // https://github.com/facebookresearch/faiss/wiki/Index-IO,-cloning-and-hyper-parameter-tuning#auto-tuning-the-runtime-parameters
        faiss::OneRecallAtRCriterion crit(nq, 1); // 1-recall@R 
        // faiss::IntersectionCriterion crit(nq, 100); // // 100-recall@R (aka. intersection)
        crit.set_groundtruth((int)k, nullptr, gt);
        crit.nnn = k; // by default, the criterion will request only 1 NN

        printf("[%lld s] Preparing auto-tune parameters\n", stopwatch.getElapsedTimeSeconds());

        faiss::ParameterSpace params;
        params.initialize(index);
        params.thread_over_batches = false;

        printf("[%lld s] Auto-tuning over %zu parameters (%zu combinations)\n",
               stopwatch.getElapsedTimeSeconds(),
               params.parameter_ranges.size(),
               params.n_combinations());

        params.explore(index, nq, xq, crit, &ops);

        printf("[%lld s] Found the following operating points: \n", stopwatch.getElapsedTimeSeconds());

        ops.display();

        // // keep the first parameter that obtains > 0.5 1-recall@1
        // for (int i = 0; i < ops.optimal_pts.size(); i++) {
        //     if (ops.optimal_pts[i].perf > 0.5) {
        //         selected_params = ops.optimal_pts[i].key;
        //         break;
        //     }
        // }
        // assert(selected_params.size() >= 0 || !"could not find good enough op point");
        printf("[%lld s] Actual memory usage: %zu Mb, Max memory usage: %zu Mb after auto tuning\n", stopwatch.getElapsedTimeSeconds(), getCurrentRSS() / 1000000, getPeakRSS() / 1000000);
    }

    // run tests with a single thread
    // https://github.com/facebookresearch/faiss/wiki/Threads-and-asynchronous-calls
    #ifdef _OPENMP
        omp_set_dynamic(0);     // Explicitly disable dynamic teams
        omp_set_num_threads(1); // Use 1 threads for all consecutive parallel regions

        std::cout << "_OPENMP " << omp_get_num_threads() << " threads" << std::endl;
    #endif

    // Use the found configuration to perform a search
    printf("[%lld s] Perform a search on %zu queries\n", stopwatch.getElapsedTimeSeconds(), nq);
    for (int o = 0; o < ops.optimal_pts.size(); o++) {

        // skip bad configurations
        if (ops.optimal_pts[o].perf < 0.8) 
            continue;

        // set search parameters
        std::string selected_params = ops.optimal_pts[o].key;
        if(selected_params.empty() == false) {
            faiss::ParameterSpace params;
            // printf("[%lld s] Setting parameter configuration \"%s\" on index\n", stopwatch.getElapsedTimeSeconds(), selected_params.c_str());
            params.set_index_parameters(index, selected_params.c_str());
        }

        // output buffers
        faiss::idx_t* I = new faiss::idx_t[nq * k];
        float* D = new float[nq * k];
        // printf("[%lld s] Actual memory usage: %zu Mb, Max memory usage: %zu Mb after setting up the search output structures\n", stopwatch.getElapsedTimeSeconds(), getCurrentRSS() / 1000000, getPeakRSS() / 1000000);

        StopW timer;
        index->search(nq, xq, k, D, I);
        auto duration_us = timer.getElapsedTimeMicro();
        // printf("[%lld s] Actual memory usage: %zu Mb, Max memory usage: %zu Mb after performing the search\n", stopwatch.getElapsedTimeSeconds(), getCurrentRSS() / 1000000, getPeakRSS() / 1000000);

        // evaluate result by hand.
        // printf("[%lld s] Compute recalls\n", stopwatch.getElapsedTimeSeconds());
        int n_1 = 0, n_10 = 0, n_100 = 0;
        for (int i = 0; i < nq; i++) {
            auto gt_nn = gt + i * k;

            // iterate over all ground thruth data
            for (int m = 0; m < k_recall_at; m++) {
                auto gt_value = gt_nn[m];

                // k1-recall@k (is in the k first elements of the predicion, the k first best ground truth element)
                for (int j = 0; j < k; j++) {
                    if (I[i * k + j] == gt_value) {
                        if (j < 1)
                            n_1++;
                        if (j < 10)
                            n_10++;
                        if (j < 100)
                            n_100++;
                    }
                }
            }
        }
        float p_1 = n_1 / float(nq) / std::min(k_recall_at, 1);
        float p_10 = n_10 / float(nq) / std::min(k_recall_at, 10);
        float p_100 = n_100 / float(nq) / std::min(k_recall_at, 100);
        printf("us/query = %8.2f, %d-R@1 = %.4f, %d-R@10 = %.4f, %d-R@100 = %.4f with parameter %s \n", duration_us / float(nq), k_recall_at, p_1, k_recall_at, p_10, k_recall_at, p_100, selected_params.c_str());

        delete[] I;
        delete[] D;
    }

    delete[] xq;
    delete[] gt;
    delete index;
    return 0;
}