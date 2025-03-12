#include <cstdio>
#include <cstdint>
#include <cassert>
#include <string>
#include <vector>
#include <utility>
#include <fcntl.h>
#include <chrono>
#include <thread>
#include <immintrin.h>
#include <omp.h>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <random>
#include <utility>
#include <iostream>
#include "core/type.hpp"
#include "core/graph.hpp"
#include "core/io.hpp"
#include "set"
#define THRESHOLD_OPENMP_LOCAL(para, length, THRESHOLD, ...) if((length) > THRESHOLD) \
{ \
    _Pragma(para) \
    __VA_ARGS__ \
} \
else  \
{ \
    __VA_ARGS__ \
} (void)0
const uint64_t MAXL = 134217728;
const uint64_t commonTag = 666;
// a random generation function will be used here to generate the random selection of edges
// Taken n as the number of random numbers to generate
// Taken N as the range of random numbers
// Taken seed as the seed for the random number generator
struct PairHash {
    size_t operator()(const std::pair<uint64_t, uint64_t>& p) const {
        return std::hash<uint64_t>{}(p.first) ^ (std::hash<uint64_t>{}(p.second) << 1);
    }
};

std::vector<uint64_t> readNumbersFromFile(const std::string& fileName) {
    std::vector<uint64_t> numbers;
    std::ifstream file(fileName);

    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + fileName);
    }

    int number;
    while (file >> number) {
        numbers.push_back(number);
    }

    file.close();
    return numbers;
}


uint64_t encodeTag(uint64_t batch, bool del_flag, bool dynamic_flag) {
    uint64_t encoded_value = 0;
    
    // Encode batch as binary (direct value)
    encoded_value |= (batch << 2);
    
    // Encode del_flag and dynamic_flag at the last two bits
    encoded_value |= (del_flag ? 1ULL : 0ULL) << 1;
    encoded_value |= (dynamic_flag ? 1ULL : 0ULL);
    
    return encoded_value;
}

// Decode function
void decodeTag(uint64_t encoded_value, bool &del_flag, bool &dynamic_flag, uint64_t &batch) {
    if (encoded_value == commonTag)
    {
        dynamic_flag = false;
        return;
    }
    // Extract flags
    del_flag = (encoded_value >> 1) & 1;
    dynamic_flag = encoded_value & 1;
    
    // Extract batch (binary decoding)
    batch = (encoded_value >> 2);
}

bool validate(uint64_t snapshotNum, uint64_t totalBatchNum, uint64_t encoded_value) {
    if (encoded_value == commonTag || snapshotNum == commonTag) {
        return encoded_value == commonTag;
    }
    bool del_flag, dynamic_flag;
    uint64_t batch;
    decodeTag(encoded_value, del_flag, dynamic_flag, batch);
    return dynamic_flag && ((del_flag && batch >= snapshotNum && batch < totalBatchNum) || (!del_flag && batch < snapshotNum));
}

bool validate_add(uint64_t batchNumber, uint64_t encoded_value) {
    if (encoded_value == 666) {
        return true;
    }
    bool del_flag, dynamic_flag;
    uint64_t batch;
    decodeTag(encoded_value, del_flag, dynamic_flag, batch);
    return !del_flag && batch <= batchNumber;
}

bool sortByLargerSecondElement(const std::pair<long, long> &a, const std::pair<long, long> &b) {
  return (a.second > b.second);
}
std::vector<uint64_t> generate_unique_random_numbers(uint64_t n, uint64_t N, uint64_t seed) 
{
    std::vector<uint64_t> numbers(N);
    std::iota(numbers.begin(), numbers.end(), 0); // Fill with 0, 1, ..., N-1

    std::default_random_engine engine(seed);
    std::shuffle(numbers.begin(), numbers.end(), engine);

    numbers.resize(n); // Keep only the first n numbers
    return numbers;
}

auto rootCompute(Graph<uint64_t>& graph, uint64_t root, uint64_t snapshotNum, uint64_t totalBatchNum) {
    auto result = graph.alloc_vertex_tree_array<uint64_t>();
    auto continue_reduce_func = [](uint64_t depth, uint64_t total_result, uint64_t local_result) -> std::pair<bool, uint64_t> {
        return std::make_pair(local_result > 0, total_result + local_result);
    };

    auto continue_reduce_print_func = [](uint64_t depth, uint64_t total_result, uint64_t local_result) -> std::pair<bool, uint64_t> {
        fprintf(stderr, "active(%lu) >= %lu\n", depth, local_result);
        return std::make_pair(local_result > 0, total_result + local_result);
    };

    // Fix: Use std::remove_reference to handle the adjedge_type
    using AdjEdgeType = typename std::remove_reference<decltype(graph)>::type::adjedge_type;

    auto update_func = [snapshotNum, totalBatchNum](uint64_t src, uint64_t dst, uint64_t src_data, uint64_t dst_data, auto adjedge) -> std::pair<bool, uint64_t> {
        return validate(snapshotNum, totalBatchNum, adjedge.data) 
            ? std::make_pair((src_data || dst_data) != dst_data, src_data || dst_data)
            : std::make_pair(false, src_data || dst_data);
    };

    auto active_result_func = [](uint64_t old_result, uint64_t src, uint64_t dst, uint64_t src_data, uint64_t old_dst_data, uint64_t new_dst_data) -> uint64_t {
        return old_result + 1;
    };

    auto equal_func = [](uint64_t src, uint64_t dst, uint64_t src_data, uint64_t dst_data, AdjEdgeType adjedge) -> bool
    {
        return src_data == dst_data;
    };

    auto init_label_func = [=](uint64_t vid) -> std::pair<uint64_t, bool>
    {
        bool is_source = (vid == root); // 仅起始点可达
        return {is_source, true}; // 需要处理的点初始状态
    };
    auto start = std::chrono::system_clock::now();
    graph.build_tree<uint64_t>(init_label_func, continue_reduce_func, update_func, active_result_func, result);
    auto end = std::chrono::system_clock::now();
    // fprintf(stderr, "Version Time: %lf\n", 1e-6*(uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(end-start).count());
    return result;
}

auto rootCompute(Graph<uint64_t>& graph, uint64_t root) {
    auto result = graph.alloc_vertex_tree_array<uint64_t>();
    auto continue_reduce_func = [](uint64_t depth, uint64_t total_result, uint64_t local_result) 
    -> std::pair<bool, uint64_t>
   {
       return std::make_pair(local_result > 0, total_result + local_result);
   };
   
   auto continue_reduce_print_func = [](uint64_t depth, uint64_t total_result, uint64_t local_result) 
       -> std::pair<bool, uint64_t>
   {
       return std::make_pair(local_result > 0, total_result + local_result);
   };
   
   using AdjEdgeType = typename std::remove_reference<decltype(graph)>::type::adjedge_type;
   
   auto update_func = [](uint64_t src, uint64_t dst, uint64_t src_data, uint64_t dst_data, AdjEdgeType adjedge) 
       -> std::pair<bool, uint64_t>
   {
       bool new_reachable = src_data || dst_data; // 如果源节点或目标节点可达，则传播可达性
       return std::make_pair(new_reachable != dst_data, new_reachable); // 若状态改变则更新
   };
   
   // 计算 SSR 传播过程中被标记的点数
   auto active_result_func = [](uint64_t old_result, uint64_t src, uint64_t dst, uint64_t src_data, 
                               uint64_t old_dst_data, uint64_t new_dst_data) -> uint64_t
   {
       return old_result + (old_dst_data != new_dst_data); // 仅当可达性发生变化时计数
   };
   
   // 判断可达性是否收敛
   auto equal_func = [](uint64_t src, uint64_t dst, uint64_t src_data, uint64_t dst_data, AdjEdgeType adjedge) 
       -> bool
   {
       return src_data == dst_data;
   };
   
   // 初始化可达性状态
   auto init_label_func = [=](uint64_t vid) -> std::pair<uint64_t, bool>
   {
       bool is_source = (vid == root); // 仅起始点可达
       return {is_source, true}; // 需要处理的点初始状态
   };

    auto start = std::chrono::system_clock::now();
    graph.build_tree<uint64_t>(init_label_func, continue_reduce_func, update_func, active_result_func, result);
    auto end = std::chrono::system_clock::now();
    // fprintf(stderr, "Base Time: %lf\n", 1e-6*(uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(end-start).count());
    return result;
}

float rootIncrementalCompute (Graph<uint64_t>& graph, uint64_t root, 
                                                decltype(graph.alloc_vertex_tree_array<uint64_t>())& originalVertexArray, 
                                                std::vector<std::pair<uint64_t, uint64_t>>& additionBatch, std::vector<std::pair<uint64_t, uint64_t>>& deletionBatch)
{

    auto continue_reduce_func = [](uint64_t depth, uint64_t total_result, uint64_t local_result) 
    -> std::pair<bool, uint64_t>
   {
       return std::make_pair(local_result > 0, total_result + local_result);
   };
   
   auto continue_reduce_print_func = [](uint64_t depth, uint64_t total_result, uint64_t local_result) 
       -> std::pair<bool, uint64_t>
   {
       return std::make_pair(local_result > 0, total_result + local_result);
   };
   
   using AdjEdgeType = typename std::remove_reference<decltype(graph)>::type::adjedge_type;
   
   auto update_func = [](uint64_t src, uint64_t dst, uint64_t src_data, uint64_t dst_data, AdjEdgeType adjedge) 
       -> std::pair<bool, uint64_t>
   {
       bool new_reachable = src_data || dst_data; // 如果源节点或目标节点可达，则传播可达性
       return std::make_pair(new_reachable != dst_data, new_reachable); // 若状态改变则更新
   };
   
   // 计算 SSR 传播过程中被标记的点数
   auto active_result_func = [](uint64_t old_result, uint64_t src, uint64_t dst, uint64_t src_data, 
                               uint64_t old_dst_data, uint64_t new_dst_data) -> uint64_t
   {
       return old_result + (old_dst_data != new_dst_data); // 仅当可达性发生变化时计数
   };
   
   // 判断可达性是否收敛
   auto equal_func = [](uint64_t src, uint64_t dst, uint64_t src_data, uint64_t dst_data, AdjEdgeType adjedge) 
       -> bool
   {
       return src_data == dst_data;
   };
   
   // 初始化可达性状态
   auto init_label_func = [=](uint64_t vid) -> std::pair<uint64_t, bool>
   {
       bool is_source = (vid == root); // 仅起始点可达
       return {is_source, true}; // 需要处理的点初始状态
   };


    std::atomic_uint64_t add_edge_len(0), del_edge_len(0);            
    std::vector<std::remove_reference_t<decltype(graph)>::edge_type> added_edges(additionBatch.size()), deled_edges(deletionBatch.size());
    added_edges.clear(); deled_edges.clear();
    auto start = std::chrono::system_clock::now();
    THRESHOLD_OPENMP_LOCAL("omp parallel for", additionBatch.size(), 1024,
        for(uint64_t i = 0; i < additionBatch.size(); i++)
            {
                auto e = additionBatch[i];
                auto old_num = graph.add_edge({e.first, e.second, (e.first+e.second)%16 + 1}, true);
                if(!old_num) added_edges[add_edge_len.fetch_add(1)] = {e.first, e.second, (e.first+e.second)%16 + 1};
            }
    );
    graph.update_tree_add<uint64_t, uint64_t>(
        continue_reduce_func,
        update_func,
        active_result_func,
        originalVertexArray, added_edges, additionBatch.size(), true 
    );
    THRESHOLD_OPENMP_LOCAL("omp parallel for", deletionBatch.size(), 1024,
        for(uint64_t i = 0; i < deletionBatch.size(); i++)
            {
                auto e = deletionBatch[i];
                auto old_num = graph.del_edge({e.first, e.second, (e.first+e.second)%16 + 1}, true);
                if(old_num==1) deled_edges[del_edge_len.fetch_add(1)] = {e.first, e.second, (e.first+e.second)%16 + 1};
            }
    );
    graph.update_tree_del<uint64_t, uint64_t>(
        init_label_func,
        continue_reduce_func,
        update_func,
        active_result_func,
        equal_func,
        originalVertexArray, deled_edges, deletionBatch.size(), true
    );
    auto end = std::chrono::system_clock::now();
    return 1e-6*(uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(end-start).count();
}

float rootNoneMutationIncrementalCompute(Graph<uint64_t>& graph, 
                                        uint64_t root, 
                                        uint64_t snapshotNum, uint64_t totalBatchNum,
                                        decltype(graph.alloc_vertex_tree_array<uint64_t>())& originalVertexArray, 
                                        std::vector<std::vector<std::pair<uint64_t, uint64_t>>>& addBatches,
                                        std::vector<std::vector<std::pair<uint64_t, uint64_t>>>& delBatches,
                                        std::vector<uint64_t>& additionBatchIndex,
                                        std::vector<uint64_t>& deletionBatchIndex)
{
    auto continue_reduce_func = [](uint64_t depth, uint64_t total_result, uint64_t local_result) -> std::pair<bool, uint64_t> {
        return std::make_pair(local_result > 0, total_result + local_result);
    };

    auto continue_reduce_print_func = [](uint64_t depth, uint64_t total_result, uint64_t local_result) -> std::pair<bool, uint64_t> {
        fprintf(stderr, "active(%lu) >= %lu\n", depth, local_result);
        return std::make_pair(local_result > 0, total_result + local_result);
    };

    // Fix: Use std::remove_reference to handle the adjedge_type
    using AdjEdgeType = typename std::remove_reference<decltype(graph)>::type::adjedge_type;

    auto update_func = [snapshotNum, totalBatchNum](uint64_t src, uint64_t dst, uint64_t src_data, uint64_t dst_data, auto adjedge) -> std::pair<bool, uint64_t> {
        return validate(snapshotNum, totalBatchNum, adjedge.data) 
            ? std::make_pair((src_data || dst_data) != dst_data, src_data || dst_data)
            : std::make_pair(false, src_data || dst_data);
    };

    auto active_result_func = [](uint64_t old_result, uint64_t src, uint64_t dst, uint64_t src_data, uint64_t old_dst_data, uint64_t new_dst_data) -> uint64_t {
        return old_result + 1;
    };

    auto equal_func = [](uint64_t src, uint64_t dst, uint64_t src_data, uint64_t dst_data, AdjEdgeType adjedge) -> bool
    {
        return src_data == dst_data;
    };

    auto init_label_func = [=](uint64_t vid) -> std::pair<uint64_t, bool>
    {
        bool is_source = (vid == root); // 仅起始点可达
        return {is_source, true}; // 需要处理的点初始状态
    };
    auto batchNum = additionBatchIndex.size() + deletionBatchIndex.size();
    auto batch_size = addBatches[0].size();
    std::vector<std::remove_reference_t<decltype(graph)>::edge_type> addedEdgesNoneMutation(batchNum *  batch_size);
    std::atomic_uint64_t lengthNoneMutation(0);
    for (auto addIndex: additionBatchIndex)
    {
        THRESHOLD_OPENMP_LOCAL("omp parallel for", batch_size, 1024,
        for(uint64_t j=0; j<batch_size; j++)
        {
            auto &e = addBatches[addIndex][j];
            addedEdgesNoneMutation[lengthNoneMutation.fetch_add(1)] = {e.first, e.second, 666};
        }
        );
    }
    for (auto delIndex: deletionBatchIndex)
    {
        THRESHOLD_OPENMP_LOCAL("omp parallel for", batch_size, 1024,
        for(uint64_t j=0; j<batch_size; j++)
        {
            auto &e = delBatches[delIndex][j];
            addedEdgesNoneMutation[lengthNoneMutation.fetch_add(1)] = {e.first, e.second, 666};
        }
        );
    }
    auto start = std::chrono::system_clock::now();
    graph.update_tree_add<uint64_t, uint64_t>(
        continue_reduce_func,
        update_func,
        active_result_func,
        originalVertexArray, addedEdgesNoneMutation, lengthNoneMutation, true 
    );
    auto end = std::chrono::system_clock::now();
    return 1e-6*(uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(end-start).count();
}

float rootNoneMutationIncrementalCompute(Graph<uint64_t>& graph,
                                            uint64_t root,
                                            uint64_t batchNumber,
                                            uint64_t totalBatchNum,
                                            decltype(graph.alloc_vertex_tree_array<uint64_t>())& originalVertexArray,
                                            std::vector<std::pair<uint64_t, uint64_t>>& addBatches)
{
    auto continue_reduce_func = [](uint64_t depth, uint64_t total_result, uint64_t local_result) -> std::pair<bool, uint64_t> {
        return std::make_pair(local_result > 0, total_result + local_result);
    };

    auto continue_reduce_print_func = [](uint64_t depth, uint64_t total_result, uint64_t local_result) -> std::pair<bool, uint64_t> {
        fprintf(stderr, "active(%lu) >= %lu\n", depth, local_result);
        return std::make_pair(local_result > 0, total_result + local_result);
    };

    // Fix: Use std::remove_reference to handle the adjedge_type
    using AdjEdgeType = typename std::remove_reference<decltype(graph)>::type::adjedge_type;

    auto update_func = [batchNumber](uint64_t src, uint64_t dst, uint64_t src_data, uint64_t dst_data, auto adjedge) -> std::pair<bool, uint64_t> {
        return validate_add(batchNumber, adjedge.data) 
            ? std::make_pair((src_data || dst_data) != dst_data, src_data || dst_data)
            : std::make_pair(false, src_data || dst_data);
    };

    auto active_result_func = [](uint64_t old_result, uint64_t src, uint64_t dst, uint64_t src_data, uint64_t old_dst_data, uint64_t new_dst_data) -> uint64_t {
        return old_result + 1;
    };

    auto equal_func = [](uint64_t src, uint64_t dst, uint64_t src_data, uint64_t dst_data, AdjEdgeType adjedge) -> bool
    {
        return src_data == dst_data;
    };

    auto init_label_func = [=](uint64_t vid) -> std::pair<uint64_t, bool>
    {
        bool is_source = (vid == root); // 仅起始点可达
        return {is_source, true}; // 需要处理的点初始状态
    };
    auto batchsize = addBatches.size();
    std::vector<std::remove_reference_t<decltype(graph)>::edge_type> addedEdgesNoneMutation(batchsize);
    std::atomic_uint64_t lengthNoneMutation(0);
    THRESHOLD_OPENMP_LOCAL("omp parallel for", batchsize, 1024,
    for(uint64_t i = 0; i < batchsize; i++)
    {
        auto &e = addBatches[i];
        addedEdgesNoneMutation[lengthNoneMutation.fetch_add(1)] = {e.first, e.second, 666};
    }
    );
    auto start = std::chrono::system_clock::now();
    graph.update_tree_add<uint64_t, uint64_t>(
        continue_reduce_func,
        update_func,
        active_result_func,
        originalVertexArray, addedEdgesNoneMutation, batchsize, true 
    );
    auto end = std::chrono::system_clock::now();
    return 1e-6*(uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(end-start).count();    
}

int main(int argc, const char** argv) {
    if (argc < 5)
    {
        fprintf(stderr, "usage: %s graph root_file batch_num batch size\n", argv[0]);
        exit(1);
    }
    std::pair<uint64_t, uint64_t> *raw_edges = nullptr;
    std::vector<uint64_t> roots = readNumbersFromFile(argv[2]);
    auto root = roots[0];
    uint64_t raw_edges_len;
    std::vector<std::pair<uint64_t, uint64_t>> temp_edges;

    auto read_start = std::chrono::system_clock::now();
    std::ifstream infile(argv[1]);
    if (!infile.is_open()) 
    {
        std::cerr << "Error: Cannot open file " << argv[0] << std::endl;
        return 1;
    }
    std::string line;
     while (std::getline(infile, line)) {
        std::istringstream iss(line);
        uint64_t src, dst;
        if (iss >> src >> dst) {
            temp_edges.emplace_back(src, dst);
        } else {
            std::cerr << "Error: Invalid line \"" << line << "\"" << std::endl;
            // return 1;
        }
    }
    infile.close();
    raw_edges_len = temp_edges.size();
    raw_edges = new std::pair<uint64_t, uint64_t>[raw_edges_len];
    #pragma omp parallel for
    for (uint64_t i = 0; i < raw_edges_len; ++i) {
        raw_edges[i] = temp_edges[i];
    }        
    temp_edges.clear();
    auto read_end = std::chrono::system_clock::now();
    fprintf(stderr, "read: %.6lfs\n", 1e-6*(uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(read_end-read_start).count());

    uint64_t batch_num = std::stoull(argv[3]);
    uint64_t batch_size = std::stod(argv[4]) * raw_edges_len;
    fprintf(stderr, "loading graph %s, root file is %s, batch_num is %lu, batch_size is %lu\n", argv[1], argv[2], batch_num, batch_size);
    uint64_t num_vertices = 0;
    {
        auto start = std::chrono::system_clock::now();
        #pragma omp parallel for
        for(uint64_t i=0;i<raw_edges_len;i++)
        {
            const auto &e = raw_edges[i];
            write_max(&num_vertices, e.first+1);
            write_max(&num_vertices, e.second+1);
        }
        auto end = std::chrono::system_clock::now();
        fprintf(stderr, "read: %.6lfs\n", 1e-6*(uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(end-start).count());
        fprintf(stderr, "|E|=%lu\n", raw_edges_len);
    }

    // create batch for streaming batches
    // batch i addition will contains edges marks as i * batch_size to (i+1) * batch_size - 1
    // batch i deletion will contains edges marks as (i+batch_num) * batch_size to (i+batch_num+1) * batch_size - 1
    // special tag will be utilized to mark the edge as addition or deletion and their version as well

    auto totalBatchSize = 2 * batch_num * batch_size;
    auto random_selection = generate_unique_random_numbers(totalBatchSize, raw_edges_len, 123456);
    std::vector<std::vector<std::pair<uint64_t, uint64_t>>> addition_batches(batch_num), deletion_batches(batch_num);
    std::vector<std::pair<uint64_t, bool>> E_tag(raw_edges_len);
    THRESHOLD_OPENMP_LOCAL("omp parallel for", raw_edges_len, 1024,
    for (size_t i = 0; i < raw_edges_len; i++)
    {
        E_tag[i] = {commonTag, false};
    }
    );
    // E_tag is used to mark the edge as addition or deletion and their version as well
    // version commonTag means the edge is always in the graph
    // True means the edge is in the deletion batch
    // False means the edge is in the addition batch
    for (uint64_t batch = 0; batch < batch_num; batch++)
    {
        for (uint64_t i = 0; i < batch_size; i++)
        {
            addition_batches[batch].push_back(raw_edges[random_selection[batch * batch_size + i]]);
            E_tag[random_selection[batch * batch_size + i]] = {batch, false};
            deletion_batches[batch].push_back(raw_edges[random_selection[(batch + batch_num) * batch_size + i]]);
            E_tag[random_selection[(batch + batch_num) * batch_size + i]] = {batch, true};
        }
    }    
    // Stream Begins First
    {
        uint64_t rootCount = 64;
        Graph<uint64_t> graphBase(num_vertices, raw_edges_len, false, true);
        #pragma omp parallel for
        for(uint64_t i=0;i<raw_edges_len;i++)
        {
            const auto &e = raw_edges[i];
            if(E_tag[i].first == commonTag) {graphBase.add_edge({e.first, e.second, (e.first+e.second)%16 + 1}, true);}
        }
        for(auto size = 0; size < batch_num; size++)
        {
            THRESHOLD_OPENMP_LOCAL("omp parallel for", batch_size, 1024,
            for (uint64_t i = 0; i < batch_size; i++)
            {
                const auto &e = deletion_batches[size][i];
                graphBase.add_edge({e.first, e.second, (e.first+e.second)%16 + 1}, true);
            }
            );          
        }

        float streamTotal = 0;
        for(auto rootNum = 0; rootNum < rootCount; rootNum++)
        {
            root = roots[rootNum];
            auto snapshotResults = graphBase.alloc_vertex_tree_array_vector<uint64_t>(batch_num + 1);
            snapshotResults[0] = rootCompute(graphBase, root);        
            for (auto i = 0; i < batch_num; i++)
            {
                THRESHOLD_OPENMP_LOCAL("omp parallel for", graphBase.getNodesNum(), 1024,
                for (auto node = 0; node < graphBase.getNodesNum(); node++)
                {
                    snapshotResults[i+1][node].parent = snapshotResults[i][node].parent;
                    snapshotResults[i+1][node].data = snapshotResults[i][node].data;
                }
                );
                auto time = rootIncrementalCompute(graphBase, root, snapshotResults[i+1], addition_batches[i], deletion_batches[i]);
                streamTotal += time;  
            }
            for (auto i = 0; i < batch_num; i++)
            {
                THRESHOLD_OPENMP_LOCAL("omp parallel for", batch_size, 1024,
                for (uint64_t j = 0; j < batch_size; j++)
                {
                    const auto &e = deletion_batches[i][j];
                    graphBase.add_edge({e.first, e.second, (e.first+e.second)%16 + 1}, true);
                }
                );
                THRESHOLD_OPENMP_LOCAL("omp parallel for", batch_size, 1024,
                for (uint64_t j = 0; j < batch_size; j++)
                {
                    const auto &e = addition_batches[i][j];
                    graphBase.add_edge({e.first, e.second, (e.first+e.second)%16 + 1}, true);
                }
                );
            }            
            if(rootNum == 7)
            {
                fprintf(stderr, "%d root cost %.6lfs Time\n", rootNum, streamTotal);
            }
            if(rootNum == 15)
            {
                fprintf(stderr, "%d root cost %.6lfs Time\n", rootNum, streamTotal);
            }
            if(rootNum == 31)
            {
                fprintf(stderr, "%d root cost %.6lfs Time\n", rootNum, streamTotal);
            }
        }
        fprintf(stderr, "streaming total time %.6lfs\n", streamTotal);
    }

{
    // Intersection Graph Read
    Graph<uint64_t> graph(num_vertices, raw_edges_len, false, true);

    {
        auto start = std::chrono::system_clock::now();
        #pragma omp parallel for
        for(uint64_t i=0;i<raw_edges_len;i++)
        {
            const auto &e = raw_edges[i];
            if(E_tag[i].first == commonTag) {graph.add_edge({e.first, e.second, commonTag}, true);}
        }
        auto end = std::chrono::system_clock::now();
        fprintf(stderr, "Intersection Graph Marked: %.6lfs\n", 1e-6*(uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(end-start).count());
        
        // Init Computation From SnapShot 0 Common Graph Add All deletion Batches
        for(auto size = 0; size < batch_num; size++)
        {
            THRESHOLD_OPENMP_LOCAL("omp parallel for", batch_size, 1024,
            for (uint64_t i = 0; i < batch_size; i++)
            {
                const auto &e = deletion_batches[size][i];
                uint64_t versionTag = encodeTag(size, true, true);
                graph.add_edge({e.first, e.second, versionTag}, true);
            }
            );          
        }
        for(auto size = 0; size < batch_num; size++)
        {
            THRESHOLD_OPENMP_LOCAL("omp parallel for", batch_size, 1024,
            for (uint64_t i = 0; i < batch_size; i++)
            {
                const auto &e = addition_batches[size][i];
                uint64_t versionTag = encodeTag(size, false, true);
                graph.add_edge({e.first, e.second, versionTag}, true);
            }
            );          
        }
    }

    uint64_t rootCount = 64;
    float directHopTime = 0;
    for(auto rootNum = 0; rootNum < rootCount; rootNum++)
    {    
        root = roots[rootNum];
        auto snapshotVersionResults = graph.alloc_vertex_tree_array_vector<uint64_t>(batch_num + 1);
        auto commonLabels = rootCompute(graph, root, commonTag, batch_num);
        for (auto i = 0; i < batch_num + 1; i++)
        {
            // snapshotVersionResults[i] = rootCompute(graph, root, i, batch_num);
            THRESHOLD_OPENMP_LOCAL("omp parallel for", graph.getNodesNum(), 1024,
            for (uint64_t j = 0; j < graph.getNodesNum(); j++)
            {
                snapshotVersionResults[i][j].parent = commonLabels[j].parent;
                snapshotVersionResults[i][j].data = commonLabels[j].data;
            }
            );
            // add batch include {0, 1, 2, ... i -1}
            // del batch include {i, i+1, i+2, ... batch_num - 1}
            std::vector<uint64_t> additionIdex = {};
            std::vector<uint64_t> deletionIndex = {};
            for (auto j = 0; j < i; j++)
            {
                additionIdex.push_back(j);
            }
            for (auto j = i; j < batch_num; j++)
            {
                deletionIndex.push_back(j);
            }
            auto time = rootNoneMutationIncrementalCompute(graph, root, i, batch_num, snapshotVersionResults[i], addition_batches, deletion_batches, additionIdex, deletionIndex);
            directHopTime += time;
        }
            if(rootNum == 7)
            {
                fprintf(stderr, "%d root cost %.6lfs Time\n", rootNum, directHopTime);
            }
            if(rootNum == 15)
            {
                fprintf(stderr, "%d root cost %.6lfs Time\n", rootNum, directHopTime);
            }
            if(rootNum == 31)
            {
                fprintf(stderr, "%d root cost %.6lfs Time\n", rootNum, directHopTime);
            }        
    }
    fprintf(stderr, "direct hop total time %.6lfs\n", directHopTime);


    // Work Sharing will invole One Direct Jmp and One single Jmp
    // Direct Hop will involves all Edges in Target Snapshot
    // single Jmp will involves only addition Edges now and Before
    float workSharingTotal = 0;
    for(auto rootNum = 0; rootNum < rootCount; rootNum++)
    {    
        root = roots[rootNum];
        auto commonLabels = rootCompute(graph, root, commonTag, batch_num);
        auto workSharingLabels = graph.alloc_vertex_tree_array_vector<uint64_t>(batch_num + 1);
        {
            // Direct Hop include deletion batches {jmp, jmp+1, jmp+2, ... batch_num - 1}
            // Single Jmp include addition batches {jmp}
            // at jmp, we will consolidate snapshot jmp
            for (auto jmp = 0; jmp < batch_num; jmp++)
            {
                // consolidation
                std::vector<uint64_t> deletionIndex = {};
                std::vector<uint64_t> additionIndex = {};
                for (auto j = jmp; j < batch_num; j++)
                {
                    deletionIndex.push_back(j);
                }
                THRESHOLD_OPENMP_LOCAL("omp parallel for", graph.getNodesNum(), 1024,
                for (uint64_t j = 0; j < graph.getNodesNum(); j++)
                {
                    workSharingLabels[jmp][j].parent = commonLabels[j].parent;
                    workSharingLabels[jmp][j].data = commonLabels[j].data;
                }
                );
                auto time = rootNoneMutationIncrementalCompute(graph, root, jmp, batch_num, workSharingLabels[jmp], addition_batches, deletion_batches, additionIndex, deletionIndex);
                workSharingTotal += time;
                auto time1 = rootNoneMutationIncrementalCompute(graph, root, jmp, batch_num, commonLabels, addition_batches[jmp]);
                workSharingTotal += time1;
            }
        }
            THRESHOLD_OPENMP_LOCAL("omp parallel for", graph.getNodesNum(), 1024,
            for (uint64_t j = 0; j < graph.getNodesNum(); j++)
            {
                workSharingLabels[batch_num][j].parent = commonLabels[j].parent;
                workSharingLabels[batch_num][j].data = commonLabels[j].data;
            }
            );
            if(rootNum == 7)
            {
                fprintf(stderr, "%d root cost %.6lfs Time\n", rootNum, workSharingTotal);
            }
            if(rootNum == 15)
            {
                fprintf(stderr, "%d root cost %.6lfs Time\n", rootNum, workSharingTotal);
            }
            if(rootNum == 31)
            {
                fprintf(stderr, "%d root cost %.6lfs Time\n", rootNum, workSharingTotal);
            }                    
    }
    fprintf(stderr, "work sharing total time %.6lfs\n", workSharingTotal);
    auto commonGraphTime = std::min(directHopTime, workSharingTotal);
    fprintf(stderr, "CommonGraph total time %.6lfs\n", commonGraphTime);
    }
    return 0;
}