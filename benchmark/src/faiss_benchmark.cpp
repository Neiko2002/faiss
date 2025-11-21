#include <filesystem>
#include <iostream>
#include <fstream>
#include <vector>
#include <unordered_set>
#include <omp.h>

#include <faiss/index.h>
#include <faiss/AutoTune.h>
#include <faiss/index_io.h>
#include <faiss/index_factory.h>

#include "config.h"
#include "example_lib.h"

void load_data(const char* filename, float*& data, unsigned& num, unsigned& dim) {
  std::ifstream in(filename, std::ios::binary);
  if(!in.is_open()){std::cout<<"open file error"<<std::endl;exit(-1);}
  in.read((char*)&dim,4);
  in.seekg(0,std::ios::end);
  std::ios::pos_type ss = in.tellg();
  size_t fsize = (size_t)ss;
  num = (unsigned)(fsize / (dim+1) / 4);
  data = new float[num * dim * sizeof(float)];

  in.seekg(0,std::ios::beg);
  for(size_t i = 0; i < num; i++){
    in.seekg(4,std::ios::cur);
    in.read((char*)(data+i*dim),dim*4);
  }
  in.close();
}

static std::vector<std::unordered_set<uint32_t>> get_ground_truth(const uint32_t* ground_truth, const size_t ground_truth_size, const uint32_t ground_truth_dims, const size_t k)
{
    auto answers = std::vector<std::unordered_set<uint32_t>>(ground_truth_size);
    answers.reserve(ground_truth_size);
    for (int i = 0; i < ground_truth_size; i++)
    {
        auto& gt = answers[i];
        gt.reserve(k);
        for (size_t j = 0; j < k; j++) 
            gt.insert(ground_truth[ground_truth_dims * i + j]);
    }

    return answers;
}

int main() {
    std::cout << "hello world" << std::endl;

    #if defined(USE_AVX)
        std::cout << "use AVX2" << std::endl;
    #elif defined(USE_SSE)
        std::cout << "use SSE" << std::endl;
    #else
        std::cout << "use arch" << std::endl;
    #endif

    const auto data_path = std::filesystem::path("e:/Data/Feature/SIFT1M/");

    // //SIFT1M
    const auto repository_file =        (data_path / "SIFT1M/sift_base.fvecs").string();
    const auto graph_file =             (data_path / "faiss" / "flat.faiss").string();
    //const auto graph_file =             (data_path / "faiss" / "IVFPQ.faiss").string();

    // database features 
    std::cout << "Load basedata and graph" << std::endl;
    float* data_load = NULL;
    unsigned points_num, dim;
    load_data(repository_file.c_str(), data_load, points_num, dim);
    std::cout << "points_num: " << points_num << ", dim: " << dim << std::endl;


    // https://github.com/facebookresearch/faiss/wiki/The-index-factory
    faiss::Index*  index = faiss::index_factory(dim, "Flat", faiss::METRIC_L2); // IndexFlat
    //faiss::Index*  index = faiss::index_factory(dim, "PQ16x12", faiss::METRIC_L2); // IndexIVFPQ
    index->train(points_num, data_load);
    index->add(points_num, data_load);

    // store
    faiss::write_index(index, graph_file.c_str());


    std::cout << "Load Query Data" << std::endl;
    size_t top_k;
    size_t query_size;
    auto ground_truth = ivecs_read(path_groundtruth.c_str(), top_k, query_size);
    auto query_features = fvecs_read(path_query.c_str(), vecdim, query_size);
    std::cout << "Actual memory usage: " << getCurrentRSS() / 1000000 << " Mb, Max memory usage: " << getPeakRSS() / 1000000 << " Mb after loading query data" << std::endl;


    std::string search_key = parameters.Get<std::string>("pq_search_key");
    faiss::ParameterSpace f_params;
    f_params.set_index_parameters(index, search_key.c_str());

    faiss::idx_t *Ids = new faiss::idx_t[k];
    float *Dists = new float[k];

    index->search(1, query, k, Dists, Ids);
    for(unsigned i=0; i<k; i++){
      indices[i] = Ids[i];
    }


    exampleLib::test_print();

    return 0;
}