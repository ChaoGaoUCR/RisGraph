/* Copyright 2020 Guanyu Feng, Tsinghua University
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

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
#include <unordered_set>
#include <unordered_map>
#include <atomic>
#define THRESHOLD_OPENMP_LOCAL(para, length, THRESHOLD, ...) if((length) > THRESHOLD) \
{ \
    _Pragma(para) \
    __VA_ARGS__ \
} \
else  \
{ \
    __VA_ARGS__ \
} (void)0
// a random generation function will be used here to generate the random selection of edges
// Taken n as the number of random numbers to generate
// Taken N as the range of random numbers
// Taken seed as the seed for the random number generator
bool sortByLargerSecondElement(const std::pair<long, long> &a, const std::pair<long, long> &b) {
  return (a.second > b.second);
}

// Custom hash function for std::pair<uint64_t, uint64_t>
struct PairHash {
    size_t operator()(const std::pair<uint64_t, uint64_t>& p) const {
        return std::hash<uint64_t>()(p.first) ^ (std::hash<uint64_t>()(p.second) << 1);
    }
};

void process_batches(const std::vector<std::vector<std::pair<uint64_t, uint64_t>>>& batches,
                     std::vector<std::atomic<bool>>& mergeEdgesFlag,
                     std::unordered_map<std::pair<uint64_t, uint64_t>, uint64_t, PairHash>& edgeIndexMap,
                     uint64_t startEnd,
                     uint64_t endEnd) 
{
    for (size_t batchIdx = startEnd; batchIdx < endEnd; batchIdx++) 
    {
        #pragma omp parallel for
        for (auto& e : batches[batchIdx]) 
        {
                auto it = edgeIndexMap.find(e);
                if (it != edgeIndexMap.end()) 
                {
                    mergeEdgesFlag[it->second].store(false, std::memory_order_relaxed);
                }
        }
    }
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
std::vector<uint64_t> rank_in(Graph<uint64_t>& G)
{
    std::vector<uint64_t> rank;
    rank.reserve(20);
    std::vector<std::pair<uint64_t, uint64_t>> vertex_;
    auto numOfNodes = G.getNodesNum();
    vertex_.reserve(numOfNodes);
    #pragma omp parallel for
    for (uint64_t i = 0; i < numOfNodes; i++ ) vertex_[i] = std::make_pair(i, G.getAllInDegree(i));
    std::sort(vertex_.begin(), vertex_.end(), sortByLargerSecondElement);
    for (size_t i = 0; i < 20; i++)
    {
        rank.emplace_back(vertex_[i].first);
    }
    return rank;
}
void countBoolVector(std::vector<std::atomic<bool>>& vec)
{
    std::atomic<uint64_t> count(0);
    #pragma omp parallel for
    for (uint64_t i = 0; i < vec.size(); i++)
    {
        if (!vec[i].load())
        {
            count.fetch_add(1);
        }
    }
    fprintf(stderr, "count: %lu Total %lu\n", count.load(), vec.size());
}
std::vector<uint64_t> rank_out(Graph<uint64_t>& G)
{
    std::vector<uint64_t> rank;
    rank.reserve(20);
    std::vector<std::pair<uint64_t, uint64_t>> vertex_;
    auto numOfNodes = G.getNodesNum();
    vertex_.reserve(numOfNodes);
    #pragma omp parallel for
    for (uint64_t i = 0; i < numOfNodes; i++ ) vertex_[i] = std::make_pair(i, G.getAllOutDegree(i));
    std::sort(vertex_.begin(), vertex_.end(), sortByLargerSecondElement);
    for (size_t i = 0; i < 20; i++)
    {
        rank.emplace_back(vertex_[i].first);
    }
    return rank;
}

std::vector<std::pair<uint64_t, uint64_t>> mergeAndRemoveDuplicates(
    const std::vector<std::pair<uint64_t, uint64_t>>& v1,
    const std::vector<std::pair<uint64_t, uint64_t>>& v2) 
{
    std::unordered_set<std::pair<uint64_t, uint64_t>, PairHash> unique_pairs(v1.begin(), v1.end());
    unique_pairs.insert(v2.begin(), v2.end());

    return std::vector<std::pair<uint64_t, uint64_t>>(unique_pairs.begin(), unique_pairs.end());
}

auto rootCompute(Graph<uint64_t>& graph, uint64_t root) {
    auto result = graph.alloc_vertex_tree_array<uint64_t>();
    const uint64_t MAXL = 134217728;
    auto continue_reduce_func = [](uint64_t depth, uint64_t total_result, uint64_t local_result) -> std::pair<bool, uint64_t> {
        return std::make_pair(local_result > 0, total_result + local_result);
    };

    auto continue_reduce_print_func = [](uint64_t depth, uint64_t total_result, uint64_t local_result) -> std::pair<bool, uint64_t> {
        fprintf(stderr, "active(%lu) >= %lu\n", depth, local_result);
        return std::make_pair(local_result > 0, total_result + local_result);
    };

    // Fix: Use std::remove_reference to handle the adjedge_type
    using AdjEdgeType = typename std::remove_reference<decltype(graph)>::type::adjedge_type;

    auto update_func = [](uint64_t src, uint64_t dst, uint64_t src_data, uint64_t dst_data, AdjEdgeType adjedge) -> std::pair<bool, uint64_t> {
        return std::make_pair(src_data + adjedge.data < dst_data, src_data + adjedge.data);
    };

    auto active_result_func = [](uint64_t old_result, uint64_t src, uint64_t dst, uint64_t src_data, uint64_t old_dst_data, uint64_t new_dst_data) -> uint64_t {
        return old_result + 1;
    };

    auto equal_func = [](uint64_t src, uint64_t dst, uint64_t src_data, uint64_t dst_data, AdjEdgeType adjedge) -> bool {
        return src_data + adjedge.data == dst_data;
    };

    auto init_label_func = [=](uint64_t vid) -> std::pair<uint64_t, bool> {
        return {vid == root ? 0 : MAXL, vid == root};
    };
    graph.build_tree<uint64_t>(init_label_func, continue_reduce_func, update_func, active_result_func, result);
    return result;
}
std::vector<std::pair<uint64_t, uint64_t>> core_generate(Graph<uint64_t>& graph)
{
    // choose 20 High Degree Nodes for core graph generation
    auto rankIn = rank_in(graph);
    auto rankOut = rank_out(graph);
    auto outRankResult = graph.alloc_vertex_tree_array_vector<uint64_t>(rankOut.size());
    auto inRankResult = graph.alloc_vertex_tree_array_vector<uint64_t>(rankIn.size());

    std::vector<bool> out_flag(graph.getNodesNum(), false);
    std::vector<bool> in_flag(graph.getNodesNum(), false);
    std::set<std::pair<uint64_t, uint64_t>> edge_set;
    std::vector<std::vector<bool>> edge_flag(graph.getNodesNum());
    std::vector<std::vector<bool>> edge_flag_in(graph.getNodesNum());

    #pragma omp parallel for
    for (uint64_t i = 0; i < graph.getNodesNum(); i++) {
        uint64_t outDegree = graph.getAllOutDegree(i);
        edge_flag[i].resize(outDegree, false);  // 直接 resize 并初始化 false
    }
    for (auto i = 0; i < rankOut.size(); i++)
    {
        uint64_t root = rankOut[i];
        outRankResult[i] = rootCompute(graph, root);
    }
    graph.transpose(); // Transpose Only happens without Delta Batches
    #pragma omp parallel for
    for (uint64_t i = 0; i < graph.getNodesNum(); i++) {
        uint64_t outDegree = graph.getAllOutDegree(i);
        edge_flag_in[i].resize(outDegree, false);
    }
    for (auto i = 0; i < rankIn.size(); i++)
    {
        uint64_t root = rankIn[i];
        inRankResult[i] = rootCompute(graph, root);
    }
    graph.transpose(); // Transpose Only happens without Delta Batches

    //start to find the core graph edges
    for (auto i = 0; i < graph.getNodesNum(); i++)
    {
        if (i % (graph.getNodesNum() / 5) == 0) {
            std::cout << "Progress: " << (i * 100 / graph.getNodesNum()) << "% completed." << std::endl;
        }
        #pragma omp parallel for
        for (uint64_t idx = 0; idx < rankOut.size(); idx++)
        {
            auto outList = graph.get_outgoing_adjlist(i);
            #pragma omp parallel for
            for (uint64_t k = 0; k < graph.getAllOutDegree(i); k++) {
                uint64_t dst = outList[k].nbr;
                uint64_t edgeLen = (i + dst) % 16 + 1;
                if (outRankResult[idx][i].data + edgeLen == outRankResult[idx][dst].data) {
                    if (graph.edgeOutCheck(i, dst)) {
                        out_flag[i] = true;
                        in_flag[dst] = true;
                        edge_flag[i][k] = true;
                    }
                }
            }
        }
    }
    std::cout << "Forward progress: 100% completed." << std::endl;
    graph.transpose();
    for (auto i = 0; i < graph.getNodesNum(); i++)
    {
        if (i % (graph.getNodesNum() / 5) == 0)
        {
            std::cout << "Progress: " << (i * 100 / graph.getNodesNum()) << "% completed." << std::endl;
        }
        #pragma omp parallel for
        for (uint64_t idx = 0; idx < rankIn.size(); idx++)
        {
            auto inList = graph.get_outgoing_adjlist(i);
            #pragma omp parallel for
            for (uint64_t k = 0; k < graph.getAllOutDegree(i); k++) {
                uint64_t dst = inList[k].nbr;
                uint64_t edgeLen = (i + dst) % 16 + 1;
                if (inRankResult[idx][i].data + edgeLen == inRankResult[idx][dst].data) {
                    if (graph.edgeOutCheck(i, dst)) {
                        in_flag[i] = true;
                        out_flag[dst] = true;
                        edge_flag_in[i][k] = true;
                    }
                }
            }
        }
    }
    std::cout << "Backward Progress: 100% completed." << std::endl;
    graph.transpose();
    fprintf(stderr, "Finish to find core graph edge with query results\n");
    #pragma omp parallel for
    for (uint64_t i = 0; i < graph.getNodesNum(); i++)
    {
        if (!out_flag[i])
        {
            if (graph.getAllOutDegree(i) > 0)
            {
                edge_flag[i][0] = true;
            }
        }
    }
    graph.transpose();
    for (uint64_t i = 0; i < graph.getNodesNum(); i++)
    {
        if (!in_flag[i])
        {
            if (graph.getAllOutDegree(i) > 0)
            {
                edge_flag_in[i][0] = true;
            }
        }
    }
    graph.transpose();
    for (uint64_t i = 0; i < graph.getNodesNum(); i++)
    {
        auto outList = graph.get_outgoing_adjlist(i);
        for (uint64_t k = 0; k < graph.getAllOutDegree(i); k++)
        {
            if (edge_flag[i][k])
            {
                auto dst = outList[k].nbr;
                edge_set.insert(std::make_pair(i, dst));
            }
        }
    }
    graph.transpose();
    for (uint64_t i = 0; i < graph.getNodesNum(); i++)
    {
        auto inList = graph.get_outgoing_adjlist(i);
        for (uint64_t k = 0; k < graph.getAllOutDegree(i); k++)
        {
            if (edge_flag_in[i][k])
            {
                auto dst = inList[k].nbr;
                edge_set.insert(std::make_pair(dst, i));
            }
        }
    }
    graph.transpose();
    fprintf(stderr, "Finish to find core graph edge with in/out flag\n");
    fprintf(stderr, "core graph edge size: %lu\n", edge_set.size());
    std::vector<std::pair<uint64_t, uint64_t>> edge_set_vector(edge_set.begin(), edge_set.end());
    return edge_set_vector;
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
int main(int argc, char** argv)
{
    if (argc != 5)
    {
        fprintf(stderr, "usage: %s graph root_file batch_num batch size\n", argv[0]);
        exit(1);
    }
    std::pair<uint64_t, uint64_t> *raw_edges = nullptr;
    std::vector<uint64_t> roots = readNumbersFromFile(argv[2]);
    // uint64_t root = std::stoull(argv[2]);
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
    uint64_t numIntersectionEdges = graph.get_degree();
    fprintf(stderr, "Number of Edges in Intersection Graph is %lu\n", numIntersectionEdges);
    auto coreEdges = core_generate(graph);

    // next we calculate edges in union Graph
    {
        for (auto batch = 0; batch < batch_num; batch++)
        {
            #pragma omp parallel for
            for (uint64_t i = 0; i < batch_size; i++)
            {
                auto e = addition_batches[batch][i];
                auto old_num = graph.add_edge({e.first, e.second, (e.first+e.second)%16 + 1}, true);
                auto e1 = deletion_batches[batch][i];
                auto old_num1 = graph.add_edge({e1.first, e1.second, (e1.first+e1.second)%16 + 1}, true);
            }
        }
    }
    fprintf(stderr, "Union Graph Edges: %lu\n", graph.get_degree());
    
    // a flag array will be used to mark the edges that are merged
    auto unionEdges = core_generate(graph);
    auto mergeEdges = mergeAndRemoveDuplicates(coreEdges, unionEdges);

    std::vector<std::atomic<bool>> mergeEdgesFlag(mergeEdges.size());
    std::unordered_map<std::pair<uint64_t, uint64_t>, uint64_t, PairHash> edgeIndexMap;

    for (size_t i = 0; i < mergeEdges.size(); i++) 
    {
        edgeIndexMap[mergeEdges[i]] = i;
    }

    fprintf(stderr, "Merge Edges: %lu\n", mergeEdges.size());
    uint64_t srcToTest = 64;
    // cold start graph to first snapshot
    {
        #pragma omp parallel for
        for (uint64_t i = 0; i < batch_size; i++)
        {
            auto e = addition_batches[0][i];
            graph.del_edge({e.first, e.second, (e.first+e.second)%16 + 1}, true);
        }
        fprintf(stderr, "Cold Start With SnapShot 0: %lu\n", graph.get_degree());
    }
    // Pruning Edges for specific SnapShot
    countBoolVector(mergeEdgesFlag);
    uint64_t SnapshotNum = 0;
    {
        // Pruning Only Happens with Snapshot Specifc Batches
        // For Snapshot i within {0, batch_num}
        // Del_Batch Include del{i, i+1... batch_num-1}
        // Add_Batch Include add{0, 1... i-1}
        // For Snapshot 0: del{1, 2... batch_num-1}
        // For Snapshot batch_num: add{0, 1... batch_num-1}
        // Batches that are not in Snapshot i is add_Batch{i, i+1, .. batch_num-1} and del_Batch{0, 1, ... i-1}
        // For Snapshot 0: add_Batch{0, 1, ... batch_num-1} and del_Batch{}
        // For Snapshot batch_num: add_Batch{} and del_Batch{0, 1, ... batch_num-1}
        auto startEnd = SnapshotNum;
        auto endEnd = batch_num;
        process_batches(addition_batches, mergeEdgesFlag, edgeIndexMap, startEnd, endEnd);
        startEnd = 0;
        endEnd = SnapshotNum;
        process_batches(deletion_batches, mergeEdgesFlag, edgeIndexMap, startEnd, endEnd);
        countBoolVector(mergeEdgesFlag);
    }
    return 0;
}
