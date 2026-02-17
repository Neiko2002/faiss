#include "dataset.h"
#include "logging.h"
#include "stopwatch.h"
#include "util.h"

#include <faiss/IndexFlat.h>
#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/index_factory.h>

#include <algorithm>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#ifndef DATA_PATH
#define DATA_PATH "data"
#endif

using namespace benchmark;

void run_faiss_test(const Dataset& ds, bool force_test)
{
    // We run two main scenarios:
    // 1. Full Dataset
    // 2. Half Dataset

    std::string method_name = "Faiss_Flat";

    // Ensure output directory exists
    std::filesystem::path output_dir = ds.data_root() / ds.name() / "faiss";
    std::filesystem::create_directories(output_dir);

    std::string log_file = (output_dir / "faiss_flat.log").string();
    set_log_file(log_file, true);  // Append mode
    attach_cout_to_log();
    attach_cerr_to_log();

    log("=== %s Benchmark %s ===\n", method_name.c_str(), ds.name());

    struct Result
    {
        float pct;
        float recall;
        double time_us;
    };

    // --- Scenario 1: Full Dataset ---
    log("\n--- Scenario 1: Full Dataset ---\n");
    {
        // Varied subset percentages
        std::vector<float> percentages = {0.19f, 0.75f, 0.95f, 1.0f};

        // Load Full Base Data
        log("Loading full base data...\n");
        size_t ram_before = getProcessCurrentRSS();
        auto base_data = ds.load_base(false);  // full
        size_t ram_after = getProcessCurrentRSS();
        log("Base data loaded. RAM: %.2f MB (+%.2f MB)\n", ram_after / (1024.0 * 1024.0),
            (ram_after - ram_before) / (1024.0 * 1024.0));

        // Load Query and GT
        auto query_data = ds.load_query();
        size_t k = 100;
        auto ground_truth = ds.load_groundtruth(k, false);  // full GT

        // Load Explore Data
        size_t k_explore = DatasetInfo::EXPLORE_TOPK;
        auto explore_queries = ds.load_explore_query();
        auto explore_gt = ds.load_explore_groundtruth(k_explore, false);  // false = full

        std::vector<Result> test_results;
        std::vector<Result> explore_results;
        for (float pct : percentages)
        {
            size_t subset_size = (size_t)(base_data.num * pct);

            StopW sw_build;
            // Create Index
            faiss::IndexFlatL2 index(base_data.dim);

            // Add subset
            // faiss takes float*
            index.add(subset_size, base_data.data);

            // Search
            StopW sw_search;
            std::vector<faiss::idx_t> I(query_data.num * k);
            std::vector<float> D(query_data.num * k);

            index.search(query_data.num, query_data.data, k, D.data(), I.data());
            double time_per_query_us = sw_search.getElapsedTimeMicro() / (double)query_data.num;

            // Recall Calculation
            size_t correct = 0;
            for (size_t i = 0; i < query_data.num; ++i)
            {
                // GT for this query
                const auto& gt_vec = ground_truth[i];  // vector<uint32_t> of size k

                // Results for this query
                for (size_t j = 0; j < k; ++j)
                {
                    faiss::idx_t id = I[i * k + j];
                    if (std::binary_search(gt_vec.begin(), gt_vec.end(), (uint32_t)id))
                    {
                        correct++;
                    }
                }
            }
            float recall = (float)correct / (query_data.num * k);
            test_results.push_back({pct, recall, time_per_query_us});

            // --- Exploration Test ---
            StopW sw_explore;
            std::vector<faiss::idx_t> I_explore(explore_queries.num * k_explore);
            std::vector<float> D_explore(explore_queries.num * k_explore);
            index.search(explore_queries.num, explore_queries.data, k_explore, D_explore.data(), I_explore.data());
            double time_explore_us = sw_explore.getElapsedTimeMicro() / (double)explore_queries.num;

            size_t correct_explore = 0;
            for (size_t i = 0; i < explore_queries.num; ++i)
            {
                const auto& gt_vec = explore_gt[i];
                for (size_t j = 0; j < k_explore; ++j)
                {
                    faiss::idx_t id = I_explore[i * k_explore + j];
                    if (std::binary_search(gt_vec.begin(), gt_vec.end(), (uint32_t)id))
                    {
                        correct_explore++;
                    }
                }
            }
            float recall_explore = (float)correct_explore / (explore_queries.num * k_explore);
            explore_results.push_back({pct, recall_explore, time_explore_us});
        }

        log("\nTest Queries:\n");
        for (const auto& r : test_results)
        {
            log("%3.0f%% \t recall %.5f \t time_us_per_query %6.0fus\n", r.pct * 100.0f, r.recall, r.time_us);
        }

        log("\nExploration Queries:\n");
        for (const auto& r : explore_results)
        {
            log("%3.0f%% \t recall %.5f \t time_us_per_query %6.0fus\n", r.pct * 100.0f, r.recall, r.time_us);
        }
    }

    // --- Scenario 2: Half Dataset ---
    log("\n--- Scenario 2: Half Dataset ---\n");
    {
        // Varied subset percentages of the HALF dataset
        std::vector<float> percentages = {0.75f, 0.95f, 1.0f};

        // Load Half Base Data
        log("Loading half base data...\n");
        size_t ram_before = getProcessCurrentRSS();
        auto base_data = ds.load_base(true);  // half = true
        size_t ram_after = getProcessCurrentRSS();
        log("Base data (half) loaded. RAM: %.2f MB (+%.2f MB)\n", ram_after / (1024.0 * 1024.0),
            (ram_after - ram_before) / (1024.0 * 1024.0));

        // Load Query and GT (Half GT)
        auto query_data = ds.load_query();
        size_t k = 100;
        auto ground_truth = ds.load_groundtruth(k, true);  // half GT

        // Load Explore Data (Half)
        size_t k_explore = DatasetInfo::EXPLORE_TOPK;
        auto explore_queries = ds.load_explore_query();
        auto explore_gt = ds.load_explore_groundtruth(k_explore, true);  // true = half

        std::vector<Result> test_results;
        std::vector<Result> explore_results;
        for (float pct : percentages)
        {
            size_t subset_size = (size_t)(base_data.num * pct);

            StopW sw_build;
            faiss::IndexFlatL2 index(base_data.dim);
            index.add(subset_size, base_data.data);

            StopW sw_search;
            std::vector<faiss::idx_t> I(query_data.num * k);
            std::vector<float> D(query_data.num * k);

            index.search(query_data.num, query_data.data, k, D.data(), I.data());
            double time_per_query_us = sw_search.getElapsedTimeMicro() / (double)query_data.num;

            size_t correct = 0;
            for (size_t i = 0; i < query_data.num; ++i)
            {
                const auto& gt_vec = ground_truth[i];
                for (size_t j = 0; j < k; ++j)
                {
                    faiss::idx_t id = I[i * k + j];
                    if (std::binary_search(gt_vec.begin(), gt_vec.end(), (uint32_t)id))
                    {
                        correct++;
                    }
                }
            }
            float recall = (float)correct / (query_data.num * k);
            test_results.push_back({pct, recall, time_per_query_us});

            // --- Exploration Test ---
            StopW sw_explore;
            std::vector<faiss::idx_t> I_explore(explore_queries.num * k_explore);
            std::vector<float> D_explore(explore_queries.num * k_explore);

            index.search(explore_queries.num, explore_queries.data, k_explore, D_explore.data(), I_explore.data());
            double time_explore_us = sw_explore.getElapsedTimeMicro() / (double)explore_queries.num;

            size_t correct_explore = 0;
            for (size_t i = 0; i < explore_queries.num; ++i)
            {
                const auto& gt_vec = explore_gt[i];
                for (size_t j = 0; j < k_explore; ++j)
                {
                    faiss::idx_t id = I_explore[i * k_explore + j];
                    if (std::binary_search(gt_vec.begin(), gt_vec.end(), (uint32_t)id))
                    {
                        correct_explore++;
                    }
                }
            }
            float recall_explore = (float)correct_explore / (explore_queries.num * k_explore);
            explore_results.push_back({pct, recall_explore, time_explore_us});
        }

        log("\nTest Queries:\n");
        for (const auto& r : test_results)
        {
            log("%3.0f%% \t recall %.5f \t time_us_per_query %6.0fus\n", r.pct * 100.0f, r.recall, r.time_us);
        }

        log("\nExploration Queries:\n");
        for (const auto& r : explore_results)
        {
            log("%3.0f%% \t recall %.5f \t time_us_per_query %6.0fus\n", r.pct * 100.0f, r.recall, r.time_us);
        }
    }

    reset_log_to_console();
}

int main(int argc, char** argv)
{
    DatasetName dataset_name = DatasetName::ALL;
    std::string data_root = DATA_PATH;
    bool force_test = false;

    for (int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];
        DatasetName ds_name = DatasetName::from_string(arg);
        if (ds_name.is_valid())
        {
            dataset_name = ds_name;
        }
        else if (arg == "--force-test")
        {
            force_test = true;
        }
        else
        {
            data_root = arg;
        }
    }

    if (dataset_name == DatasetName::ALL)
    {
        for (const auto& name : DatasetName::all())
        {
            Dataset ds(name, data_root);
            run_faiss_test(ds, force_test);
        }
    }
    else
    {
        Dataset ds(dataset_name, data_root);
        run_faiss_test(ds, force_test);
    }

    return 0;
}
