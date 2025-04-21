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
    std::iota(numbers.begin(), numbers.end(), 0);
    std::default_random_engine engine(seed);
    std::shuffle(numbers.begin(), numbers.end(), engine);
    numbers.resize(n);
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

auto rootCompute(Graph<uint64_t>& graph, uint64_t root) {
    auto result = graph.alloc_vertex_tree_array<uint64_t>();
    const uint64_t INF = UINT64_MAX;
    auto continue_reduce_func = [](uint64_t depth, uint64_t total_result, uint64_t local_result) -> std::pair<bool, uint64_t> {
        return std::make_pair(local_result > 0, total_result + local_result);
    };

    auto continue_reduce_print_func = [](uint64_t depth, uint64_t total_result, uint64_t local_result) -> std::pair<bool, uint64_t> {
        fprintf(stderr, "active(%lu) >= %lu\n", depth, local_result);
        return std::make_pair(local_result > 0, total_result + local_result);
    };

    using AdjEdgeType = typename std::remove_reference<decltype(graph)>::type::adjedge_type;

    auto update_func = [](uint64_t src, uint64_t dst, uint64_t src_data, uint64_t dst_data, AdjEdgeType adjedge) -> std::pair<bool, uint64_t> {
        // For CC, we propagate the component ID (root) to all connected nodes
        return std::make_pair(src_data < dst_data, src_data);
    };

    auto active_result_func = [](uint64_t old_result, uint64_t src, uint64_t dst, uint64_t src_data, uint64_t old_dst_data, uint64_t new_dst_data) -> uint64_t {
        return old_result + 1;
    };

    auto equal_func = [](uint64_t src, uint64_t dst, uint64_t src_data, uint64_t dst_data, AdjEdgeType adjedge) -> bool {
        return src_data == dst_data;
    };

    auto init_label_func = [=](uint64_t vid) -> std::pair<uint64_t, bool> {
        return {vid, vid == root};  // Each node starts as its own component
    };

    graph.build_tree<uint64_t>(init_label_func, continue_reduce_func, update_func, active_result_func, result);
    return result;
}

// ... [Rest of the code remains the same as the original file, just change the algorithm name in the output messages]
// ... [The main function and other helper functions remain the same]

int main(int argc, char** argv)
{
    // ... [Same as the original main function, just change the algorithm name in the output messages]
    // ... [The rest of the implementation remains the same]
    return 0;
} 