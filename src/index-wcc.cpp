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
std::vector<uint64_t> rankHotVertics(Graph<uint64_t>& G, uint64_t numNodes = 5)
{
    std::vector<uint64_t> rank;
    rank.reserve(numNodes);
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

auto rootCompute(Graph<uint64_t>& graph, uint64_t root) {
    auto result = graph.alloc_vertex_tree_array<uint64_t>();
    auto continue_reduce_func = [](uint64_t depth, uint64_t total_result, uint64_t local_result) -> std::pair<bool, uint64_t>
    {
        return std::make_pair(local_result > 0, total_result + local_result);
    };

    auto continue_reduce_print_func = [](uint64_t depth, uint64_t total_result, uint64_t local_result) -> std::pair<bool, uint64_t>
    {
        return std::make_pair(local_result > 0, total_result + local_result);
    };

    using AdjEdgeType = typename std::remove_reference<decltype(graph)>::type::adjedge_type;

    auto update_func = [](uint64_t src, uint64_t dst, uint64_t src_data, uint64_t dst_data, AdjEdgeType adjedge) -> std::pair<bool, uint64_t>
    {
        uint64_t new_label = std::min(src_data, dst_data); // 取较小的 ID 作为 WCC 代表
        return std::make_pair(new_label < dst_data, new_label); // 仅当新 ID 更小时更新
    };

    auto active_result_func = [](uint64_t old_result, uint64_t src, uint64_t dst, uint64_t src_data, uint64_t old_dst_data, uint64_t new_dst_data) -> uint64_t
    {
        return old_result + 1;
    };

    auto equal_func = [](uint64_t src, uint64_t dst, uint64_t src_data, uint64_t dst_data, AdjEdgeType adjedge) -> bool
    {
        return src_data == dst_data;
    };

    auto init_label_func = [](uint64_t vid) -> std::pair<uint64_t, bool>
    {
        return {vid, true}; // 初始化，每个节点的 WCC ID 设为自身 ID
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
    auto start = std::chrono::system_clock::now();
    std::vector<uint64_t> nodeStartIndex(graph.getNodesNum() + 1, 0);
    uint64_t totalEdges = 0;

    for (uint64_t i = 0; i < graph.getNodesNum(); i++) {
        nodeStartIndex[i] = totalEdges;
        totalEdges += graph.getAllOutDegree(i);
    }
    nodeStartIndex[graph.getNodesNum()] = totalEdges;

    THRESHOLD_OPENMP_LOCAL("omp parallel for", totalEdges, 1024,
    for (uint64_t edgeIndex = 0; edgeIndex < totalEdges; edgeIndex++) {
        // **计算 src（起始节点）**
        uint64_t src = std::upper_bound(nodeStartIndex.begin(), nodeStartIndex.end(), edgeIndex) - nodeStartIndex.begin() - 1;
        // **计算 k（当前 src 的出边索引）**
        uint64_t k = edgeIndex - nodeStartIndex[src];

        // **获取目标节点 dst**
        uint64_t dst = graph.getOutDstForMainCSR(src, k);
        
        // **执行计算**
        uint64_t edgeLen = (src + dst) % 16 + 1;
        for (uint64_t idx = 0; idx < rankOut.size(); idx++) {
            if (outRankResult[idx][src].data + edgeLen == outRankResult[idx][dst].data) {
                if (graph.get_edge_num({src, dst, edgeLen}) > 0) {
                    #pragma omp critical
                    {
                        out_flag[src] = true;
                        in_flag[dst] = true;
                        edge_flag[src][k] = true;
                    }
                }
            }
        }
    }
    );
    std::cout << "Forward progress: 100% completed." << std::endl;
    auto end = std::chrono::system_clock::now();
    fprintf(stderr, "Forward Time: %.6lfs\n", 1e-6*(uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(end-start).count());
    graph.transpose();
    start = std::chrono::system_clock::now();
    totalEdges = 0;
    for (uint64_t i = 0; i < graph.getNodesNum(); i++) {
        nodeStartIndex[i] = totalEdges;
        totalEdges += graph.getAllOutDegree(i);
    }
    nodeStartIndex[graph.getNodesNum()] = totalEdges;

    THRESHOLD_OPENMP_LOCAL("omp parallel for", totalEdges, 1024,
    for (uint64_t edgeIndex = 0; edgeIndex < totalEdges; edgeIndex++) {
        // **计算 src（起始节点）**
        uint64_t src = std::upper_bound(nodeStartIndex.begin(), nodeStartIndex.end(), edgeIndex) - nodeStartIndex.begin() - 1;

        // **计算 k（当前 src 的出边索引）**
        uint64_t k = edgeIndex - nodeStartIndex[src];

        // **获取目标节点 dst**
        uint64_t dst = graph.getOutDstForMainCSR(src, k);
        
        // **执行计算**
        uint64_t edgeLen = (src + dst) % 16 + 1;
        for (uint64_t idx = 0; idx < rankIn.size(); idx++) {
            if (inRankResult[idx][src].data + edgeLen == inRankResult[idx][dst].data) {
                if (graph.get_edge_num({src, dst, edgeLen})) {
                    #pragma omp critical
                    {
                        in_flag[src] = true;
                        out_flag[dst] = true;
                        edge_flag_in[src][k] = true;
                    }
                }
            }
        }
    }
    );
    std::cout << "Backward Progress: 100% completed." << std::endl;
    end = std::chrono::system_clock::now();
    fprintf(stderr, "Backward Time: %.6lfs\n", 1e-6*(uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(end-start).count());
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
    fprintf(stderr," Percentage of core graph edge: %.2f\n", 100.0 * edge_set.size() / graph.get_degree());
    std::vector<std::pair<uint64_t, uint64_t>> edge_set_vector(edge_set.begin(), edge_set.end());
    return edge_set_vector;
}

bool versionCheck(uint64_t version, bool addOrDel, uint64_t snapShot)
{
    // snapShot = 999 means the Union Graph
    if (snapShot == 999)
    {
        return true;
    }
    if (snapShot == 666) // Generate From Common Graph
    {
        if (version == 666)
        {
            return true;
        }
        else
        {
            return false;
        }
    }
    else
    {
        // snapshot from 0 to batch_num
        if (version == 666)
        {
            return true;
        }
        if (addOrDel) // Deletion {snapShot, snapShot + 1, ..., batch_num}
        {
            if (version < snapShot)
            {
                return false;
            }
            else
            {
                return true;
            }
        }
        else // Addition {0, 1, ..., snapShot - 1}
        {
            if (version >= snapShot)
            {
                return false;
            }
            else
            {
                return true;
            }
        }
    }
}
bool checkGraphAndEdgeList(Graph<uint64_t>& graph, std::vector<std::pair<uint64_t, bool>> &E_tag, uint64_t version)
{
    auto graphSize = graph.get_degree();
    std::atomic<uint64_t> edgeCount(0);
    THRESHOLD_OPENMP_LOCAL("omp parallel for", E_tag.size(), 1024,
        for (uint64_t i = 0; i < E_tag.size(); i++)
        {
            auto tag = E_tag[i];
            if (versionCheck(tag.first, tag.second, version))
            {
                edgeCount.fetch_add(1);
            }
        }
    );
    fprintf(stderr, "Graph Size: %lu, Edge Count For Version %lu is %lu\n", graphSize, version ,edgeCount.load());
    if (edgeCount.load() == graphSize)
    {
        return true;
    }
    else
    {
        return false;
    }
}
std::vector<std::pair<uint64_t, uint64_t>> coreGenerateVector(Graph<uint64_t>& graph, 
                                                                std::pair<uint64_t, uint64_t> *raw_edges, 
                                                                std::vector<std::pair<uint64_t, bool>> &E_tag, 
                                                                uint64_t raw_edges_len,
                                                                uint64_t snapShotNum = 666)
{
    // First Check Whether the graph Match the version
    if(!checkGraphAndEdgeList(graph, E_tag, snapShotNum))
    {
        fprintf(stderr, "Graph and Edge List do not match\n");
        exit(1);
    }
    // choose 20 High Degree Nodes for core graph generation
    auto rankIn = rank_in(graph);
    auto rankOut = rank_out(graph);
    auto outRankResult = graph.alloc_vertex_tree_array_vector<uint64_t>(rankOut.size());
    auto inRankResult = graph.alloc_vertex_tree_array_vector<uint64_t>(rankIn.size());

    std::vector<bool> out_flag(graph.getNodesNum(), false);
    std::vector<bool> in_flag(graph.getNodesNum(), false);
    std::set<std::pair<uint64_t, uint64_t>> edge_set;
    std::vector<bool> edgeOutFlag(raw_edges_len, false);
    std::vector<bool> edgeInFlag(raw_edges_len, false);

    for (auto i = 0; i < rankOut.size(); i++)
    {
        uint64_t root = rankOut[i];
        outRankResult[i] = rootCompute(graph, root);
    }

    graph.transpose(); // Transpose Only happens without Delta Batches
    for (auto i = 0; i < rankIn.size(); i++)
    {
        uint64_t root = rankIn[i];
        inRankResult[i] = rootCompute(graph, root);
    }    
    graph.transpose(); // Transpose Only happens without Delta Batches

    //start to find the core graph edges
    auto start = std::chrono::system_clock::now();

    std::vector<std::vector<bool>> outRankResultAtomic(rankOut.size());
    std::vector<std::vector<bool>> inRankResultAtomic(rankIn.size());

    for (uint64_t i = 0; i < rankOut.size(); i++)
    {
        outRankResultAtomic[i].resize(graph.getNodesNum());
        for (uint64_t j = 0; j < graph.getNodesNum(); j++)
        {
            outRankResultAtomic[i][j] = false;
        }
        inRankResultAtomic[i].resize(graph.getNodesNum());
        for (uint64_t j = 0; j < graph.getNodesNum(); j++)
        {
            inRankResultAtomic[i][j] = false;
        }
    }

    THRESHOLD_OPENMP_LOCAL("omp parallel for", raw_edges_len, 1024,
        for (auto edge = 0; edge < raw_edges_len; edge++)
        {
            auto e = raw_edges[edge];
            auto tag = E_tag[edge];
            if (versionCheck(tag.first, tag.second, snapShotNum))
            {
                uint64_t src = e.first;
                uint64_t dst = e.second;
                for (uint64_t idx = 0; idx < rankOut.size(); idx++) 
                {
                    bool outFlag = (std::min(outRankResult[idx][src].data, outRankResult[idx][dst].data) == outRankResult[idx][dst].data) && !outRankResultAtomic[idx][dst];
                    bool inFlag = (std::min(inRankResult[idx][dst].data, inRankResult[idx][src].data) == inRankResult[idx][src].data) && !inRankResultAtomic[idx][src];
                    if (outFlag || inFlag)
                    {
                        #pragma omp critical
                        {
                            out_flag[src] = true;
                            in_flag[dst] = true;
                            edgeOutFlag[edge] = true;
                            outRankResultAtomic[idx][dst] = true;
                            inRankResultAtomic[idx][src] = true;
                        }
                    }
                }
            }
        }
    );
    auto end = std::chrono::system_clock::now();
    fprintf(stderr, "Forward Time: %.6lfs\n", 1e-6*(uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(end-start).count());
    std::cout << "Forward progress: 100% completed." << std::endl;
    

    for (uint64_t i = 0; i < graph.getNodesNum(); i++)
    {
        if (!out_flag[i])
        {
            if (graph.getAllOutDegree(i) > 0)
            {
                edge_set.insert(std::make_pair(i, graph.getOutDstForMainCSR(i, 0)));
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
                edge_set.insert(std::make_pair(graph.getOutDstForMainCSR(i, 0), i));
            }
        }    
    }
    graph.transpose();

    for (uint64_t i = 0; i < raw_edges_len; i++)
    {
        if (edgeOutFlag[i] || edgeInFlag[i])
        {
            edge_set.insert(raw_edges[i]);
        }
    }

    fprintf(stderr, "Finish to find core graph edge with in/out flag\n");
    fprintf(stderr, "core graph edge size: %lu\n", edge_set.size());
    fprintf(stderr," Percentage of core graph edge: %.2f\n", 100.0 * edge_set.size() / graph.get_degree());
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

void parallel_insert(tbb::concurrent_unordered_set<std::pair<uint64_t, uint64_t>, PairHash>& set, 
                     const std::vector<std::pair<uint64_t, uint64_t>>& data) {
    size_t num_threads = std::thread::hardware_concurrency();
    size_t chunk_size = (data.size() + num_threads - 1) / num_threads;
    std::vector<std::thread> threads;

    for (size_t i = 0; i < num_threads; ++i) {
        threads.emplace_back([&, i]() {
            size_t start = i * chunk_size;
            size_t end = std::min(start + chunk_size, data.size());
            for (size_t j = start; j < end; ++j) {
                set.insert(data[j]);
            }
        });
    }
    for (auto& t : threads) {
        t.join();
    }
}

std::vector<std::pair<uint64_t, uint64_t>> MergeTwoVectorEdge(std::vector<std::pair<uint64_t, uint64_t>> &a, 
                                                              std::vector<std::pair<uint64_t, uint64_t>> &b) {
    tbb::concurrent_unordered_set<std::pair<uint64_t, uint64_t>, PairHash> merged_set;

    parallel_insert(merged_set, a);
    parallel_insert(merged_set, b);
    fprintf(stderr, "Merged Set Size: %lu\n", merged_set.size());
    return std::vector<std::pair<uint64_t, uint64_t>>(merged_set.begin(), merged_set.end());
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
    // Union Graph Read
    Graph<uint64_t> graph(num_vertices, raw_edges_len, false, true);
    {
        auto start = std::chrono::system_clock::now();
        #pragma omp parallel for
        for(uint64_t i=0;i<raw_edges_len;i++)
        {
            const auto &e = raw_edges[i];
            // if(E_tag[i].first == 666) {graph.add_edge({e.first, e.second, (e.first+e.second)%16 + 1}, true);}
            graph.add_edge({e.first, e.second, (e.first+e.second)%16 + 1}, true);
        }
        auto end = std::chrono::system_clock::now();
        fprintf(stderr, "Union Graph Marked: %.6lfs\n", 1e-6*(uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(end-start).count());
    }

    fprintf(stderr, "Number of Edges in Union Graph is %lu\n", graph.get_degree());
    auto unionCoreGraphEdges = coreGenerateVector(graph, raw_edges, E_tag, raw_edges_len, 999);

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
            THRESHOLD_OPENMP_LOCAL("omp parallel for", batch_size, 1024,
            for (uint64_t i = 0; i < batch_size; i++)
            {
                const auto &e = deletion_batches[batch][i];
                graph.del_edge({e.first, e.second, (e.first+e.second)%16 + 1}, true);
            }
            );
        }
    }
    auto commonGraphEdges = graph.get_degree();
    fprintf(stderr, "Common Graph Edges: %lu\n", graph.get_degree());
    
    auto coreCommonGraphEdges = coreGenerateVector(graph, raw_edges, E_tag, raw_edges_len, 666);
    auto coreForAllEdges = MergeTwoVectorEdge(unionCoreGraphEdges, coreCommonGraphEdges);
        
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
    uint64_t srcToCalculate = 1;
    auto preCorrectResult = graph.alloc_vertex_tree_array_vector<uint64_t>(srcToCalculate);
    auto currentCorrectResult = graph.alloc_vertex_tree_array_vector<uint64_t>(srcToCalculate);

    // Initial Boundary Allocation and Initial Results
    std::vector<std::vector<std::pair<uint64_t, uint64_t>>> previousBoundaryAllocation(srcToCalculate);
    std::vector<std::vector<std::pair<uint64_t, uint64_t>>> currentBoundaryAllocation(srcToCalculate);
    for (auto srcToTest = 0; srcToTest < srcToCalculate; srcToTest++)
    {
        // Initial Results
        preCorrectResult[srcToTest].reserve(graph.getNodesNum());
        currentCorrectResult[srcToTest].reserve(graph.getNodesNum());
        // Initial Boundary Allocation
        previousBoundaryAllocation[srcToTest].reserve(graph.getNodesNum());
        currentBoundaryAllocation[srcToTest].reserve(graph.getNodesNum());
        THRESHOLD_OPENMP_LOCAL("omp parallel for", graph.getNodesNum(), 1024,
        for (uint64_t i = 0; i < graph.getNodesNum(); i++)
        {
            previousBoundaryAllocation[srcToTest][i] = std::make_pair(0, MAXL);
            currentBoundaryAllocation[srcToTest][i] = std::make_pair(0, MAXL);
        }
        );   
    }
    std::atomic<uint64_t> correctPrediction(0);

    for (auto snapshotGraph = 0; snapshotGraph < batch_num + 1; snapshotGraph++)
    {
        fprintf(stderr, "------------SnapShot %d Begins--------------\n", snapshotGraph);
        
        Graph<uint64_t> coreForAllGraph(num_vertices, coreForAllEdges.size(), false, true);
        correctPrediction.store(0);
        // Process Core Graph Result

        {
            THRESHOLD_OPENMP_LOCAL("omp parallel for", coreForAllEdges.size(), 1024,
            for (uint64_t i = 0; i < coreForAllEdges.size(); i++)
            {
                const auto &e = coreForAllEdges[i];
                if (graph.get_edge_num({e.first, e.second, (e.first + e.second) % 16 + 1}))
                {
                    coreForAllGraph.add_edge({e.first, e.second, (e.first+e.second)%16 + 1}, true);
                }
            }
            );

            if (graph.get_degree() - batch_num * batch_size != commonGraphEdges)
            {
                fprintf(stderr, "Common Graph Edges is Wrong\n");
                exit(1);
            }
        }
        
        // Tringle inEquality Hot->AnySrc | Target->AnySrc | Hot->Target
        // Target->AnySrc <= Hot->AnySrc + Hot->Target
        // Target->AnySrc >= Hot->AnySrc - Hot->Target
        // Target is root[srcToTest]
        // WCC Hot == Target == AnySrc When Triangle Exists
        for (auto srcToTest = 0; srcToTest < srcToCalculate; srcToTest++)
        {
            auto target = roots[srcToTest];
            if (snapshotGraph != 0)
            {
                currentCorrectResult[srcToTest] = rootCompute(coreForAllGraph, roots[srcToTest]);
                std::vector<bool> isChanging(graph.getNodesNum(), false);
                THRESHOLD_OPENMP_LOCAL("omp parallel for", graph.getNodesNum(), 1024,
                for (uint64_t i = 0; i < graph.getNodesNum(); i++)
                {
                    if (preCorrectResult[srcToTest][i].data != currentCorrectResult[srcToTest][i].data)
                    {
                        isChanging[i] = true;
                    }
                }
                );

                // Hot vertices will be used for bounadry allocation
                auto rankTriple = rankHotVertics(graph);
                for (auto & hotNode : rankTriple)
                {
                    auto tmpResult = rootCompute(coreForAllGraph, hotNode);
                    THRESHOLD_OPENMP_LOCAL("omp parallel for", graph.getNodesNum(), 1024,
                    for (uint64_t anySrc = 0; anySrc < graph.getNodesNum(); anySrc++)
                    {
                        auto hotToAnySrc = tmpResult[anySrc].data;
                        auto hotToTarget = tmpResult[target].data;
                        auto lowerBound = 0;
                        // if (hotToAnySrc > hotToTarget)
                        // {
                        //     auto lowerBound = hotToAnySrc - hotToTarget;
                        // }
                        // else
                        // {
                        //     auto lowerBound = hotToTarget - hotToAnySrc;
                        // }
                        // auto upperBound = hotToAnySrc + hotToTarget;
                        auto upperBound = std::min(hotToAnySrc, hotToTarget);
                        lowerBound = upperBound;
                        currentBoundaryAllocation[srcToTest][anySrc].first = std::max(currentBoundaryAllocation[srcToTest][anySrc].first, static_cast<uint64_t>(lowerBound));
                        currentBoundaryAllocation[srcToTest][anySrc].second = std::min(currentBoundaryAllocation[srcToTest][anySrc].second, static_cast<uint64_t>(upperBound));
                    }
                    );
                }
                std::vector<bool> isBoundaryChanging(graph.getNodesNum(), false);

                // Only Boundary Not Include Each Other will be considered as changing
                {
                    THRESHOLD_OPENMP_LOCAL("omp parallel for", graph.getNodesNum(), 1024,
                    for (uint64_t i = 0; i < graph.getNodesNum(); i++)
                    {
                        auto preLower = previousBoundaryAllocation[srcToTest][i].first;
                        auto preUpper = previousBoundaryAllocation[srcToTest][i].second;
                        auto curLower = currentBoundaryAllocation[srcToTest][i].first;
                        auto curUpper = currentBoundaryAllocation[srcToTest][i].second;
                        if (preLower > curUpper || preUpper < curLower)
                        {
                            isBoundaryChanging[i] = true;
                        }
                    }
                    );
                }
                // Check the correctness of the prediction
                {
                    THRESHOLD_OPENMP_LOCAL("omp parallel for", graph.getNodesNum(), 1024,
                    for (uint64_t i = 0; i < graph.getNodesNum(); i++)
                    {
                        if (isChanging[i] == isBoundaryChanging[i])
                        {
                            correctPrediction.fetch_add(1);
                        }
                    }
                    );
                }
            }
            else
            {
                // Snapshot = 0 wll only compute the correct Results and Boundary Allocation But Not Predicting the correctness
                currentCorrectResult[srcToTest] = rootCompute(coreForAllGraph, roots[srcToTest]);
                
                // Boundary Updates
                {
                    auto rankTriple = rankHotVertics(graph);
                    for (auto & hotNode : rankTriple)
                    {
                        auto tmpResult = rootCompute(coreForAllGraph, hotNode);
                        THRESHOLD_OPENMP_LOCAL("omp parallel for", graph.getNodesNum(), 1024,
                        for (uint64_t anySrc = 0; anySrc < graph.getNodesNum(); anySrc++)
                        {
                            auto hotToAnySrc = tmpResult[anySrc].data;
                            auto hotToTarget = tmpResult[target].data;
                            auto lowerBound = 0;
                            // if (hotToAnySrc > hotToTarget)
                            // {
                            //     auto lowerBound = hotToAnySrc - hotToTarget;
                            // }
                            // else
                            // {
                            //     auto lowerBound = hotToTarget - hotToAnySrc;
                            // }
                            // auto upperBound = hotToAnySrc + hotToTarget;
                            auto upperBound = std::min(hotToAnySrc, hotToTarget);
                            lowerBound = upperBound;
                            currentBoundaryAllocation[srcToTest][anySrc].first = std::max(currentBoundaryAllocation[srcToTest][anySrc].first, static_cast<uint64_t>(lowerBound));
                            currentBoundaryAllocation[srcToTest][anySrc].second = std::min(currentBoundaryAllocation[srcToTest][anySrc].second, static_cast<uint64_t>(upperBound));
                        }
                        );
                    }
                }
            }

            // ALways Update the Previous Result and Boundary Allocation
            THRESHOLD_OPENMP_LOCAL("omp parallel for", graph.getNodesNum(), 1024,
            for (uint64_t i = 0; i < graph.getNodesNum(); i++)
            {
                preCorrectResult[srcToTest][i] = currentCorrectResult[srcToTest][i];
                previousBoundaryAllocation[srcToTest][i] = currentBoundaryAllocation[srcToTest][i];
            }
            );

        }
        if (snapshotGraph != 0)
        {
            fprintf(stderr, "Correct Prediction: %.2f%%\n", (100.0 * correctPrediction.load())/ (graph.getNodesNum() * srcToCalculate));
        }

    // Process For Next Batch    
    {
        if (snapshotGraph == batch_num)
            {
                break;
            }
            THRESHOLD_OPENMP_LOCAL("omp parallel for", batch_size, 1024,
            for (uint64_t i = 0; i < batch_size; i++)
            {
                const auto &e = addition_batches[snapshotGraph][i];
                graph.add_edge({e.first, e.second, (e.first+e.second)%16 + 1}, true);
            }
            );
            THRESHOLD_OPENMP_LOCAL("omp parallel for", batch_size, 1024,
            for (uint64_t i = 0; i < batch_size; i++)
            {
                const auto &e = deletion_batches[snapshotGraph][i];
                graph.del_edge({e.first, e.second, (e.first+e.second)%16 + 1}, true);
            }
            );
    }
    
    }
    return 0;
}
