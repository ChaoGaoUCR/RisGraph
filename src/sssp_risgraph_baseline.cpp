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
int main(int argc, char** argv)
{
    if (argc != 4)
    {
        fprintf(stderr, "usage: %s graph root batch_prefix\n", argv[0]);
        exit(1);
    }
    else
    {
        fprintf(stderr, "loading graph %s, root is %s, file_prefix is %s\n", argv[1], argv[2], argv[3]);
    }
    std::pair<uint64_t, uint64_t> *raw_edges;
    uint64_t root = std::stoull(argv[2]);
    uint64_t raw_edges_len;
    std::tie(raw_edges, raw_edges_len) = mmap_binary(argv[1]);
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
    Graph<uint64_t> graph(num_vertices, raw_edges_len, false, true);
    {
        auto start = std::chrono::system_clock::now();
        #pragma omp parallel for
        for(uint64_t i=0;i<raw_edges_len;i++)
        {
            const auto &e = raw_edges[i];
            graph.add_edge({e.first, e.second, (e.first+e.second)%16 + 1}, true);
        }
        auto end = std::chrono::system_clock::now();
        fprintf(stderr, "add: %.6lfs\n", 1e-6*(uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(end-start).count());
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
    // Perform execution for streaming
    // default 15 batches
    {
        std::vector<std::vector<std::pair<uint64_t, uint64_t>>> addition_batches;
        addition_batches.resize(15);
        std::vector<std::vector<std::pair<uint64_t, uint64_t>>> deletion_batches;
        deletion_batches.resize(15);
        std::string batch_prefix = argv[3];
        std::vector<std::chrono::system_clock::time_point> add_mutation_time;
        std::vector<std::chrono::system_clock::time_point> del_mutation_time;
        std::vector<std::chrono::system_clock::time_point> add_compute_time;
        std::vector<std::chrono::system_clock::time_point> del_compute_time;


        for (uint64_t batch = 0; batch < 15; batch++)
        {
            std::vector<decltype(graph)::edge_type> added_edges(addition_batches[batch].size()), deled_edges(deletion_batches[batch].size());
            std::atomic_uint64_t add_edge_len(0), del_edge_len(0);
            added_edges.clear();
            deled_edges.clear();
            std::string batch_file = batch_prefix + ".add." + std::to_string(batch) + ".txt";
            fprintf(stderr, "Processing batch file is %s\n", batch_file.c_str());
            std::ifstream add_file(batch_file);
            if (!add_file.is_open())
            {
                fprintf(stderr, "Error: Cannot open file %s\n", batch_file.c_str());
                exit(1);
            }
            std::string line;
            while(std::getline(add_file, line))
            {
                std::istringstream iss(line);
                uint64_t src, dst;
                if(!(iss >> src >> dst))
                {
                    fprintf(stderr, "Error: Cannot parse line %s\n", line.c_str());
                    exit(1);
                }
                else
                {
                    addition_batches[batch].emplace_back(src, dst);
                }
            }
            add_file.close();
            batch_file = batch_prefix + ".del." + std::to_string(batch) + ".txt";
            fprintf(stderr, "Processing batch file is %s\n", batch_file.c_str());
            std::ifstream del_file(batch_file);
            if (!del_file.is_open())
            {
                fprintf(stderr, "Error: Cannot open file %s\n", batch_file.c_str());
                exit(1);
            }
            while(std::getline(del_file, line))
            {
                std::istringstream iss(line);
                uint64_t src, dst;
                if(!(iss >> src >> dst))
                {
                    fprintf(stderr, "Error: Cannot parse line %s\n", line.c_str());
                    exit(1);
                }
                else
                {
                    deletion_batches[batch].emplace_back(src, dst);
                }
            }
            del_file.close();
            // add and delete computation once
            add_edge_len = 0; del_edge_len = 0;
            added_edges.clear(); deled_edges.clear();

            auto start = std::chrono::system_clock::now();
            std::atomic_uint64_t length(0);
            THRESHOLD_OPENMP_LOCAL("omp parallel for", addition_batches[batch].size(), 1024, 
                for(uint64_t i=0; i<addition_batches[batch].size(); i++)
                {
                    const auto &e = addition_batches[batch][i];
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
                labels, added_edges, length.load(), true
            );
            end = std::chrono::system_clock::now();
            add_compute_time.push_back(1e-6*(uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(end-start).count());
            length = 0;
            THRESHOLD_OPENMP_LOCAL("omp parallel for", deletion_batches[batch].size(), 1024, 
                for(uint64_t i=0;i<deletion_batches[batch].size();i++)
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
                labels, deled_edges, length.load(), true
            );
            end = std::chrono::system_clock::now();
            del_compute_time.push_back(1e-6*(uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(end-start).count());
            fprintf(stderr, "Batch %lu: add mutation %.6lfs, add compute %.6lfs, del mutation %.6lfs, del compute %.6lfs\n", batch, add_mutation_time.back(), add_compute_time.back(), del_mutation_time.back(), del_compute_time.back());
        }
    }

    return 0;
}
