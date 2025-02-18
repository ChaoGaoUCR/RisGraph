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
// a random generation function will be used here to generate the random selection of edges
// Taken n as the number of random numbers to generate
// Taken N as the range of random numbers
// Taken seed as the seed for the random number generator
struct PairHash {
    size_t operator()(const std::pair<uint64_t, uint64_t>& p) const {
        return std::hash<uint64_t>{}(p.first) ^ (std::hash<uint64_t>{}(p.second) << 1);
    }
};
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

auto rootCompute(Graph<uint64_t>& graph, uint64_t root) {
    auto result = graph.alloc_vertex_tree_array<uint64_t>();

    auto continue_reduce_func = [](uint64_t depth, uint64_t total_result, uint64_t local_result) -> std::pair<bool, uint64_t>
    {
        return std::make_pair(local_result>0, total_result+local_result);
    };
    auto continue_reduce_print_func = [](uint64_t depth, uint64_t total_result, uint64_t local_result) -> std::pair<bool, uint64_t>
    {
        fprintf(stderr, "active(%lu) >= %lu\n", depth, local_result);
        return std::make_pair(local_result>0, total_result+local_result);
    };
    using AdjEdgeType = typename std::remove_reference<decltype(graph)>::type::adjedge_type;

    auto update_func = [](uint64_t src, uint64_t dst, uint64_t src_data, uint64_t dst_data, AdjEdgeType adjedge) -> std::pair<bool, uint64_t>
    {
        return std::make_pair(std::min(src_data, adjedge.data) > dst_data, std::min(src_data, adjedge.data));
    };
    auto active_result_func = [](uint64_t old_result, uint64_t src, uint64_t dst, uint64_t src_data, uint64_t old_dst_data, uint64_t new_dst_data) -> uint64_t
    {
        return old_result+1;
    };
    auto equal_func = [](uint64_t src, uint64_t dst, uint64_t src_data, uint64_t dst_data, AdjEdgeType adjedge) -> bool
    {
        return std::min(src_data, adjedge.data) == dst_data;
    };
    auto init_label_func = [=](uint64_t vid) -> std::pair<uint64_t, bool>
    {
        return {vid==root?MAXL:0, vid==root};
    };

    graph.build_tree<uint64_t>(init_label_func, continue_reduce_func, update_func, active_result_func, result);
    return result;
}

float rootIncrementalCompute (Graph<uint64_t>& graph, uint64_t root, 
                                                decltype(graph.alloc_vertex_tree_array<uint64_t>())& originalVertexArray, 
                                                std::vector<std::pair<uint64_t, uint64_t>>& additionBatch, std::vector<std::pair<uint64_t, uint64_t>>& deletionBatch)
{

    auto continue_reduce_func = [](uint64_t depth, uint64_t total_result, uint64_t local_result) -> std::pair<bool, uint64_t>
    {
        return std::make_pair(local_result>0, total_result+local_result);
    };
    auto continue_reduce_print_func = [](uint64_t depth, uint64_t total_result, uint64_t local_result) -> std::pair<bool, uint64_t>
    {
        fprintf(stderr, "active(%lu) >= %lu\n", depth, local_result);
        return std::make_pair(local_result>0, total_result+local_result);
    };
    using AdjEdgeType = typename std::remove_reference<decltype(graph)>::type::adjedge_type;

    auto update_func = [](uint64_t src, uint64_t dst, uint64_t src_data, uint64_t dst_data, AdjEdgeType adjedge) -> std::pair<bool, uint64_t>
    {
        return std::make_pair(std::min(src_data, adjedge.data) > dst_data, std::min(src_data, adjedge.data));
    };
    auto active_result_func = [](uint64_t old_result, uint64_t src, uint64_t dst, uint64_t src_data, uint64_t old_dst_data, uint64_t new_dst_data) -> uint64_t
    {
        return old_result+1;
    };
    auto equal_func = [](uint64_t src, uint64_t dst, uint64_t src_data, uint64_t dst_data, AdjEdgeType adjedge) -> bool
    {
        return std::min(src_data, adjedge.data) == dst_data;
    };
    auto init_label_func = [=](uint64_t vid) -> std::pair<uint64_t, bool>
    {
        return {vid==root?MAXL:0, vid==root};
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

std::vector<std::vector<std::pair<uint64_t, uint64_t>>> directHopBatchConstruct(std::vector<std::vector<std::pair<uint64_t, uint64_t>>>& addBatches, 
                    std::vector<std::vector<std::pair<uint64_t, uint64_t>>>& delBatches,
                    uint64_t snapShotNum)
{
    std::vector<std::vector<std::pair<uint64_t, uint64_t>>> hopBatches;
    auto batchNum = addBatches.size();
    if (snapShotNum > batchNum)
    {
        fprintf(stderr, "Error: snapShotNum should <= batchNum\n");
        exit(1);
    }
    if (delBatches.size() != batchNum)
    {
        fprintf(stderr, "Error: delBatches.size() != addBatches.size()\n");
        exit(1);
    }

    for (uint64_t i = 0; i < snapShotNum; i++)
    {
        hopBatches.push_back(addBatches[i]);
    }
    for (uint64_t i = snapShotNum; i < batchNum; i++)
    {
        hopBatches.push_back(delBatches[i]);
    }
    return hopBatches;
}

float rootNoneMutationIncremental(Graph<uint64_t>& graph, uint64_t root, 
            decltype(graph.alloc_vertex_tree_array<uint64_t>())& originalVertexArray, 
            std::vector<std::vector<std::pair<uint64_t, uint64_t>>>& batches)
        {
        graph.InitStreamDH(batches);
    auto continue_reduce_func = [](uint64_t depth, uint64_t total_result, uint64_t local_result) -> std::pair<bool, uint64_t>
    {
        return std::make_pair(local_result>0, total_result+local_result);
    };
    auto continue_reduce_print_func = [](uint64_t depth, uint64_t total_result, uint64_t local_result) -> std::pair<bool, uint64_t>
    {
        fprintf(stderr, "active(%lu) >= %lu\n", depth, local_result);
        return std::make_pair(local_result>0, total_result+local_result);
    };
    using AdjEdgeType = typename std::remove_reference<decltype(graph)>::type::adjedge_type;

    auto update_func = [](uint64_t src, uint64_t dst, uint64_t src_data, uint64_t dst_data, AdjEdgeType adjedge) -> std::pair<bool, uint64_t>
    {
        return std::make_pair(std::min(src_data, adjedge.data) > dst_data, std::min(src_data, adjedge.data));
    };
    auto active_result_func = [](uint64_t old_result, uint64_t src, uint64_t dst, uint64_t src_data, uint64_t old_dst_data, uint64_t new_dst_data) -> uint64_t
    {
        return old_result+1;
    };
    auto equal_func = [](uint64_t src, uint64_t dst, uint64_t src_data, uint64_t dst_data, AdjEdgeType adjedge) -> bool
    {
        return std::min(src_data, adjedge.data) == dst_data;
    };
    auto init_label_func = [=](uint64_t vid) -> std::pair<uint64_t, bool>
    {
        return {vid==root?MAXL:0, vid==root};
    };
        auto batch_num = batches.size();
        auto batch_size = batches[0].size();
        std::vector<std::remove_reference_t<decltype(graph)>::edge_type> addedEdgesNoneMutation(batch_num *  batch_size);
        std::atomic_uint64_t lengthNoneMutation(0);
        THRESHOLD_OPENMP_LOCAL("omp parallel for", batch_num*batch_size, 1024,
        for(uint64_t i = 0; i < batch_num; i++)
        {
            for(uint64_t j=0; j<batch_size; j++)
            {
                auto &e = batches[i][j];
                addedEdgesNoneMutation[lengthNoneMutation.fetch_add(1)] = {e.first, e.second, (e.first+e.second)%16 + 1};
            }
        }
        );
        auto start = std::chrono::system_clock::now();
        graph.update_tree_add<uint64_t, uint64_t>(
            continue_reduce_func,
            update_func,
            active_result_func,
            originalVertexArray, addedEdgesNoneMutation, lengthNoneMutation, true 
        );
        auto end = std::chrono::system_clock::now();
        graph.clearBatchIndex();
        graph.clearBatch();
        return 1e-6*(uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(end-start).count();
        }

float rootNoneMutationIncremental(Graph<uint64_t>& graph, uint64_t root, 
    decltype(graph.alloc_vertex_tree_array<uint64_t>())& originalVertexArray, 
    std::vector<std::vector<std::pair<uint64_t, uint64_t>>>& addBatches,
    std::vector<std::vector<std::pair<uint64_t, uint64_t>>>& delBatches)
    {
        // This function will use all additionBatch to do Incremental Computation and all deletionBatch to do traverse
        graph.InitStreamBatch(addBatches, delBatches);
    auto continue_reduce_func = [](uint64_t depth, uint64_t total_result, uint64_t local_result) -> std::pair<bool, uint64_t>
    {
        return std::make_pair(local_result>0, total_result+local_result);
    };
    auto continue_reduce_print_func = [](uint64_t depth, uint64_t total_result, uint64_t local_result) -> std::pair<bool, uint64_t>
    {
        fprintf(stderr, "active(%lu) >= %lu\n", depth, local_result);
        return std::make_pair(local_result>0, total_result+local_result);
    };
    using AdjEdgeType = typename std::remove_reference<decltype(graph)>::type::adjedge_type;

    auto update_func = [](uint64_t src, uint64_t dst, uint64_t src_data, uint64_t dst_data, AdjEdgeType adjedge) -> std::pair<bool, uint64_t>
    {
        return std::make_pair(std::min(src_data, adjedge.data) > dst_data, std::min(src_data, adjedge.data));
    };
    auto active_result_func = [](uint64_t old_result, uint64_t src, uint64_t dst, uint64_t src_data, uint64_t old_dst_data, uint64_t new_dst_data) -> uint64_t
    {
        return old_result+1;
    };
    auto equal_func = [](uint64_t src, uint64_t dst, uint64_t src_data, uint64_t dst_data, AdjEdgeType adjedge) -> bool
    {
        return std::min(src_data, adjedge.data) == dst_data;
    };
    auto init_label_func = [=](uint64_t vid) -> std::pair<uint64_t, bool>
    {
        return {vid==root?MAXL:0, vid==root};
    };
        
        auto batch_num = addBatches.size();
        auto batch_size = addBatches[0].size();
        auto traverseBatchNum = delBatches.size();
        std::vector<std::remove_reference_t<decltype(graph)>::edge_type> addedEdgesNoneMutation(batch_num *  batch_size);
        std::atomic_uint64_t lengthNoneMutation(0);
        THRESHOLD_OPENMP_LOCAL("omp parallel for", batch_num*batch_size, 1024,
        for(uint64_t i = 0; i < batch_num; i++)
        {
            for(uint64_t j=0; j<batch_size; j++)
            {
                auto &e = addBatches[i][j];
                addedEdgesNoneMutation[lengthNoneMutation.fetch_add(1)] = {e.first, e.second, (e.first+e.second)%16 + 1};
            }
        }
        );
        std::vector<uint64_t> additionBatchIndex = {0};
        std::vector<uint64_t> deletionBatchIndex = {};
        if (traverseBatchNum > 0)
        {
            deletionBatchIndex.push_back(0);
        }
        graph.batchCoverageUpdate(additionBatchIndex, deletionBatchIndex);
        auto start = std::chrono::system_clock::now();
        graph.update_tree_add<uint64_t, uint64_t>(
            continue_reduce_func,
            update_func,
            active_result_func,
            originalVertexArray, addedEdgesNoneMutation, lengthNoneMutation, true 
        );
        auto end = std::chrono::system_clock::now();
        graph.clearBatchIndex();
        graph.clearBatch();
        return 1e-6*(uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(end-start).count();
    }


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

int main(int argc, const char** argv) {
    if (argc < 5)
    {
        fprintf(stderr, "usage: %s graph root_file batch_num batch size\n", argv[0]);
        exit(1);
    }
    std::pair<uint64_t, uint64_t> *raw_edges = nullptr;
    std::vector<uint64_t> roots = readNumbersFromFile(argv[2]);
    auto root = 1;
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
        E_tag[i] = {666, false};
    }
    );
    // E_tag is used to mark the edge as addition or deletion and their version as well
    // version 666 means the edge is always in the graph
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
    // Intersection Graph Read
    Graph<uint64_t> graph(num_vertices, raw_edges_len, false, true);
    {
        auto start = std::chrono::system_clock::now();
        #pragma omp parallel for
        for(uint64_t i=0;i<raw_edges_len;i++)
        {
            const auto &e = raw_edges[i];
            if(E_tag[i].first == 666) {graph.add_edge({e.first, e.second, (e.first+e.second)%16 + 1}, true);}
        }
        auto end = std::chrono::system_clock::now();
        fprintf(stderr, "Intersection Graph Marked: %.6lfs\n", 1e-6*(uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(end-start).count());
    }
    fprintf(stderr,"%lu number of Edges in the graph\n", graph.get_degree());
    {
        // Init Computation From SnapShot 0 Common Graph Add All deletion Batches
        for(auto size = 0; size < batch_num; size++)
        {
            THRESHOLD_OPENMP_LOCAL("omp parallel for", batch_size, 1024,
            for (uint64_t i = 0; i < batch_size; i++)
            {
                const auto &e = deletion_batches[size][i];
                graph.add_edge({e.first, e.second, (e.first+e.second)%16 + 1}, true);
            }
            );          
        }
    }
    auto snapshotResults = graph.alloc_vertex_tree_array_vector<uint64_t>(batch_num + 1);
    snapshotResults[0] = rootCompute(graph, root);
    float streamTotal = 0;
    for (auto i = 0; i < batch_num; i++)
    {
        THRESHOLD_OPENMP_LOCAL("omp parallel for", graph.getNodesNum(), 1024,
        for (auto node = 0; node < graph.getNodesNum(); node++)
        {
            snapshotResults[i+1][node].parent = snapshotResults[i][node].parent;
            snapshotResults[i+1][node].data = snapshotResults[i][node].data;
        }
    );
        auto time = rootIncrementalCompute(graph, root, snapshotResults[i+1], addition_batches[i], deletion_batches[i]);
        // fprintf(stderr, "batch %d incremental compute %.6lfs\n", i, time);
        streamTotal += time;
    }
    fprintf(stderr, "streaming total time %.6lfs\n", streamTotal);

    // Start Processing Graph from Intersection Graph
    {
        for (auto batch = 0; batch < batch_num; batch++)
        {
            THRESHOLD_OPENMP_LOCAL("omp parallel for", batch_size, 1024,
            for (uint64_t i = 0; i < batch_size; i++)
            {
                const auto &e = addition_batches[batch][i];
                graph.del_edge({e.first, e.second, (e.first+e.second)%16 + 1}, true);
            }
            );
        }        
    }
    if(raw_edges_len - 2 * batch_num * batch_size != graph.get_degree())
    {
        fprintf(stderr, "%lu %lu\n", raw_edges_len - batch_num * batch_size, graph.get_degree());
        fprintf(stderr, "Error: Number of Edges in Intersection Graph is not equal to the number of edges in the snapshot\n");
        return 1;
    }
    else
    {
        fprintf(stderr, "Number of Edges in Snapshot is correct\n");
    }
    auto commonLabels = rootCompute(graph, root);
    // Process Graph With DirectHop
    {
        auto directHopLabels = graph.alloc_vertex_tree_array_vector<uint64_t>(batch_num + 1);
        // graph.InitStreamBatch(batch_num, batch_size, addition_batches, deletion_batches);
        float hopTotal = 0;
        for (auto snapshotNum = 0; snapshotNum < batch_num + 1; snapshotNum++)
        {
            THRESHOLD_OPENMP_LOCAL("omp parallel for", graph.getNodesNum(), 1024,
            for (uint64_t i = 0; i < graph.getNodesNum(); i++)
            {
                directHopLabels[snapshotNum][i].data = commonLabels[i].data;
                directHopLabels[snapshotNum][i].parent = commonLabels[i].parent;
            }
            );            
            // DirectHope will include batch_Num of batches
            // i = {0, 1, 2... batch_num}
            // Snapshot i will inclde {addition_{0}, .. addition_{i-1}}
            // Snapshot i will inclde {deletion_{i}, .. deletion_{batch_num}}
            auto directHopBatches = directHopBatchConstruct(addition_batches, deletion_batches, snapshotNum);
            auto time = rootNoneMutationIncremental(graph, root, directHopLabels[snapshotNum], directHopBatches);
            // fprintf(stderr, "direct hop batch %d incremental compute %.6lfs\n", snapshotNum, time);
            std::atomic<uint64_t> correctResult(0);
            THRESHOLD_OPENMP_LOCAL("omp parallel for", graph.getNodesNum(), 1024,
            for (uint64_t i = 0; i < graph.getNodesNum(); i++)
            {
                if (directHopLabels[snapshotNum][i].data == snapshotResults[snapshotNum][i].data)
                {
                    correctResult.fetch_add(1);
                }
            }
            );
            hopTotal += time;
            // fprintf(stderr, "direct hop batch %d correctness %.6lf %%\n", snapshotNum, (double) (100 *correctResult.load()) / graph.getNodesNum());
        }
        fprintf(stderr, "direct hop total time %.6lfs\n", hopTotal);
    }
    if(raw_edges_len - 2 * batch_num * batch_size != graph.get_degree())
    {
        fprintf(stderr, "%lu %lu\n", raw_edges_len - batch_num * batch_size, graph.get_degree());
        fprintf(stderr, "Error: Number of Edges in Intersection Graph is not equal to the number of edges in the snapshot\n");
        return 1;
    }
    else
    {
        fprintf(stderr, "Number of Edges in Snapshot is correct\n");
    }
    // Process with WorkSharing
    float wsTime = 0;
    {
        auto bottomHopLabels = rootCompute(graph, root);
        auto workSharingLabels = graph.alloc_vertex_tree_array_vector<uint64_t>(batch_num + 1);
        std::vector<std::vector<std::pair<uint64_t, uint64_t>>> computationBatches;
        std::vector<std::vector<std::pair<uint64_t, uint64_t>>> traverseBatches = {};
        // computation Batches starts at del_0, del_1, del_2, ... del_{batch_num-1} and each jmp will remove del_{jmp} after current computation
        // traverse Batches starts at None and each jmp will add add_{jmp} after current computation
        // left jmp include both computationBacthes and traverseBatches
        // righjt jmp include only traverseBatches and add_{jmp}
        for (auto jmp = 0; jmp < batch_num; jmp++)
        {
            computationBatches.push_back(deletion_batches[jmp]);
        }
        for (auto jmp = 0; jmp < batch_num; jmp++)
        {
            THRESHOLD_OPENMP_LOCAL("omp parallel for", graph.getNodesNum(), 1024,
            for (uint64_t i = 0; i < graph.getNodesNum(); i++)
            {
                workSharingLabels[jmp][i].data = bottomHopLabels[i].data;
                workSharingLabels[jmp][i].parent = bottomHopLabels[i].parent;
            }
            );
            auto time = rootNoneMutationIncremental(graph, root, workSharingLabels[jmp], computationBatches);
            // auto time = rootNoneMutationIncremental(graph, root, workSharingLabels[jmp], computationBatches, traverseBatches);

            wsTime += time;
            // fprintf(stderr, "work sharing batch %d incremental compute %.6lfs\n", jmp, time);
            std::atomic<uint64_t> correctResult(0);
            THRESHOLD_OPENMP_LOCAL("omp parallel for", graph.getNodesNum(), 1024,
            for (uint64_t i = 0; i < graph.getNodesNum(); i++)
            {
                if (workSharingLabels[jmp][i].data == snapshotResults[jmp][i].data)
                {
                    correctResult.fetch_add(1);
                }
            }
            );
            // fprintf(stderr, "work sharing batch %d correctness %.6lf %%\n", jmp, (double) (100 *correctResult.load()) / graph.getNodesNum());
            // std::vector<std::vector<std::pair<uint64_t, uint64_t>>> rightComputationBatch = {addition_batches[jmp]};
            // auto time2 = rootNoneMutationIncremental(graph, root, bottomHopLabels, rightComputationBatch, traverseBatches);
            auto start = std::chrono::system_clock::now();
            std::vector<std::pair<uint64_t, uint64_t>> emptyBatch;
            rootIncrementalCompute(graph, root, bottomHopLabels, addition_batches[jmp], emptyBatch);
            auto end = std::chrono::system_clock::now();
            auto time2 = 1e-6*(uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(end-start).count();
            wsTime += time2;
            computationBatches.erase(computationBatches.begin());
            traverseBatches.push_back(addition_batches[jmp]);
        }
        fprintf(stderr, "work sharing total time %.6lfs\n", wsTime);
    }


    return 0;
}