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
std::vector<uint64_t> generate_unique_random_numbers(uint64_t n, uint64_t N, uint64_t seed) 
{
    std::vector<uint64_t> numbers(N);
    std::iota(numbers.begin(), numbers.end(), 0); // Fill with 0, 1, ..., N-1

    std::default_random_engine engine(seed);
    std::shuffle(numbers.begin(), numbers.end(), engine);

    numbers.resize(n); // Keep only the first n numbers
    return numbers;
}
int main(int argc, char** argv)
{
    if (argc != 5)
    {
        fprintf(stderr, "usage: %s graph root batch_num batch size\n", argv[0]);
        exit(1);
    }
    std::pair<uint64_t, uint64_t> *raw_edges = nullptr;
    uint64_t root = std::stoull(argv[2]);
    uint64_t raw_edges_len;
    std::vector<std::pair<uint64_t, uint64_t>> temp_edges;
    // std::tie(raw_edges, raw_edges_len) = mmap_binary(argv[1]);

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
    fprintf(stderr, "loading graph %s, root is %lu, batch_num is %lu, batch_size is %lu\n", argv[1], root, batch_num, batch_size);
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
    
    // Adding All Deletion Batch for Computation
    {
        auto start = std::chrono::system_clock::now();
        #pragma omp parallel for
        for(uint64_t i=0; i<batch_num; i++)
        {
            for(uint64_t j=0; j<batch_size; j++)
            {
                const auto &e = deletion_batches[i][j];
                graph.add_edge({e.first, e.second, (e.first+e.second)%16 + 1}, true);
            }
        }
        auto end = std::chrono::system_clock::now();
        fprintf(stderr, "add: %.6lfs\n", 1e-6*(uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(end-start).count());
    }
    uint64_t numSnapshot0Edges = graph.get_degree();
    fprintf(stderr, "Number of Edges in Snapshot is %lu\n", numSnapshot0Edges);
    if (numIntersectionEdges != numSnapshot0Edges - batch_num * batch_size)
    {
        fprintf(stderr, "Error: Number of Edges in Intersection Graph is not equal to the number of edges in the snapshot\n");
    }
    else
    {
        fprintf(stderr, "Number of Edges in Snapshot is correct\n");
    }
    auto labels = graph.alloc_vertex_tree_array<uint64_t>();
    const uint64_t MAXL = 134217728;
    auto continue_reduce_func = [](uint64_t depth, uint64_t total_result, uint64_t local_result) -> std::pair<bool, uint64_t>
    {
        return std::make_pair(local_result>0, total_result+local_result);
    };
    auto continue_reduce_print_func = [](uint64_t depth, uint64_t total_result, uint64_t local_result) -> std::pair<bool, uint64_t>
    {
        fprintf(stderr, "active(%lu) >= %lu\n", depth, local_result);
        return std::make_pair(local_result>0, total_result+local_result);
    };
    auto update_func = [](uint64_t src, uint64_t dst, uint64_t src_data, uint64_t dst_data, decltype(graph)::adjedge_type adjedge) -> std::pair<bool, uint64_t>
    {
        return std::make_pair(src_data+adjedge.data < dst_data, src_data + adjedge.data);
    };
    auto active_result_func = [](uint64_t old_result, uint64_t src, uint64_t dst, uint64_t src_data, uint64_t old_dst_data, uint64_t new_dst_data) -> uint64_t
    {
        return old_result+1;
    };
    auto equal_func = [](uint64_t src, uint64_t dst, uint64_t src_data, uint64_t dst_data, decltype(graph)::adjedge_type adjedge) -> bool
    {
        return src_data + adjedge.data == dst_data;
    };
    auto init_label_func = [=](uint64_t vid) -> std::pair<uint64_t, bool>
    {
        return {vid==root?0:MAXL, vid==root};
    };

    {
        auto start = std::chrono::system_clock::now();

        graph.build_tree<uint64_t, uint64_t>(
            init_label_func,
            continue_reduce_func,
            update_func,
            active_result_func,
            labels
        );

        auto end = std::chrono::system_clock::now();
        fprintf(stderr, "Initial exec: %.6lfs\n", 1e-6*(uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(end-start).count());
    }
    // Execution for all batches Here
    {
        std::vector<double> add_mutation_time, add_compute_time, del_mutation_time, del_compute_time;
        for (uint64_t batch = 0; batch < batch_num; batch++)
        {
            std::atomic_uint64_t length(0);
            std::atomic_uint64_t add_edge_len(0), del_edge_len(0);            
            std::vector<decltype(graph)::edge_type> added_edges(addition_batches[batch].size()), deled_edges(deletion_batches[batch].size());
            added_edges.clear(); deled_edges.clear();            
            auto start = std::chrono::system_clock::now();
            THRESHOLD_OPENMP_LOCAL("omp parallel for", batch_size, 1024,
            for(uint64_t i = 0; i < batch_size; i++)
            {
                auto e = addition_batches[batch][i];
                auto old_num = graph.add_edge({e.first, e.second, (e.first+e.second)%16 + 1}, true);
                if(!old_num) added_edges[length.fetch_add(1)] = {e.first, e.second, (e.first+e.second)%16 + 1};
            }
            );
            auto end = std::chrono::system_clock::now();
            add_mutation_time.push_back(1e-6*(uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(end-start).count());
            start = std::chrono::system_clock::now();
            graph.update_tree_add<uint64_t, uint64_t>(
                continue_reduce_func,
                update_func,
                active_result_func,
                labels, added_edges, batch_size, true 
            );
            end = std::chrono::system_clock::now();
            add_compute_time.push_back(1e-6*(uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(end-start).count());
            start = std::chrono::system_clock::now();
            length = 0;
            THRESHOLD_OPENMP_LOCAL("omp parallel for", batch_size, 1024,
            for(uint64_t i=0;i<batch_size;i++)
            {   
                const auto &e = deletion_batches[batch][i];
                auto old_num = graph.del_edge({e.first, e.second, (e.first+e.second)%16 + 1}, true);
                if(old_num==1) deled_edges[length.fetch_add(1)] = {e.first, e.second, (e.first+e.second)%16 + 1};

            }
            );
            end = std::chrono::system_clock::now();
            del_mutation_time.push_back(1e-6*(uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(end-start).count());
            start = std::chrono::system_clock::now();
            graph.update_tree_del<uint64_t, uint64_t>(
                init_label_func,
                continue_reduce_func,
                update_func,
                active_result_func,
                equal_func,
                labels, deled_edges, batch_size, true
            );
            end = std::chrono::system_clock::now();
            del_compute_time.push_back(1e-6*(uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(end-start).count());
            fprintf(stderr, "batch %lu add mutation %.6lfs, add compute %.6lfs, del mutation %.6lfs, del compute %.6lfs\n", batch, add_mutation_time.back(), add_compute_time.back(), del_mutation_time.back(), del_compute_time.back());
    }
        double add_mutation_time_sum = 0;
        double add_compute_time_sum = 0;
        double del_mutation_time_sum = 0;
        double del_compute_time_sum = 0;
        for (auto add_time : add_mutation_time)
        {
            add_mutation_time_sum += add_time;
        }
        for (auto add_time : add_compute_time)
        {
            add_compute_time_sum += add_time;
        }
        for (auto del_time : del_mutation_time)
        {
            del_mutation_time_sum += del_time;
        }
        for (auto del_time : del_compute_time)
        {
            del_compute_time_sum += del_time;
        }
        fprintf(stderr, "add mutation %.6lfs, add compute %.6lfs, del mutation %.6lfs, del compute %.6lfs\n", add_mutation_time_sum, add_compute_time_sum, del_mutation_time_sum, del_compute_time_sum);
        fprintf(stderr, "RISGRAPH Total Time: %.6lfs\n", add_mutation_time_sum + add_compute_time_sum + del_mutation_time_sum + del_compute_time_sum);
    }
    return 0;
}
