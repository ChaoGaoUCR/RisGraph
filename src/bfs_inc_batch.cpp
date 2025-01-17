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
#include <chrono>
#include <thread>
#include <fcntl.h>
#include <unistd.h>
#include <immintrin.h>
#include <omp.h>
#include "core/type.hpp"
#include "core/graph.hpp"
#include "core/io.hpp"
#include <tbb/parallel_sort.h>

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
    assert(argc > 3);
    std::pair<uint64_t, uint64_t> *raw_edges;
    uint64_t root = std::stoull(argv[2]);
    uint64_t batch = std::stoull(argv[3]);
    uint64_t raw_edges_len;
    std::tie(raw_edges, raw_edges_len) = mmap_binary(argv[1]);
    uint64_t num_vertices = 0;
    {
        auto start = std::chrono::high_resolution_clock::now();
        #pragma omp parallel for
        for(uint64_t i=0;i<raw_edges_len;i++)
        {
            const auto &e = raw_edges[i];
            write_max(&num_vertices, e.first+1);
            write_max(&num_vertices, e.second+1);
        }
        auto end = std::chrono::high_resolution_clock::now();
        fprintf(stderr, "read: %.6lfs\n", 1e-6*(uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(end-start).count());
        fprintf(stderr, "|E|=%lu\n", raw_edges_len);
    }
    Graph<void> graph(num_vertices, raw_edges_len, false, true);
    //std::random_shuffle(raw_edges.begin(), raw_edges.end());
    uint64_t imported_edges = raw_edges_len*0.9;
    //uint64_t total_batches = std::min(1000000*batch, 200000000lu);
    //if(raw_edges_len*0.1 > total_batches) imported_edges = raw_edges_len - total_batches; 
    {
        auto start = std::chrono::high_resolution_clock::now();
        #pragma omp parallel for
        for(uint64_t i=0;i<imported_edges;i++)
        {
            const auto &e = raw_edges[i];
            graph.add_edge({e.first, e.second}, true);
        }
        auto end = std::chrono::high_resolution_clock::now();
        fprintf(stderr, "add: %.6lfs\n", 1e-6*(uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(end-start).count());
    }

    auto labels = graph.alloc_vertex_tree_array<uint64_t>();
    const uint64_t MAXL = 65536;
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
        return std::make_pair(src_data+1 < dst_data, src_data + 1);
    };
    auto active_result_func = [](uint64_t old_result, uint64_t src, uint64_t dst, uint64_t src_data, uint64_t old_dst_data, uint64_t new_dst_data) -> uint64_t
    {
        return old_result+1;
    };
    auto equal_func = [](uint64_t src, uint64_t dst, uint64_t src_data, uint64_t dst_data, decltype(graph)::adjedge_type adjedge) -> bool
    {
        return src_data + 1 == dst_data;
    };
    auto init_label_func = [=](uint64_t vid) -> std::pair<uint64_t, bool>
    {
        return {vid==root?0:MAXL, vid==root};
    };

    {
        auto start = std::chrono::high_resolution_clock::now();

        graph.build_tree<uint64_t, uint64_t>(
            init_label_func,
            continue_reduce_print_func,
            update_func,
            active_result_func,
            labels
        );

        auto end = std::chrono::high_resolution_clock::now();
        fprintf(stderr, "exec: %.6lfs\n", 1e-6*(uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(end-start).count());
    }

    std::atomic_uint64_t add_edge_len(0), del_edge_len(0);
    std::vector<decltype(graph)::edge_type> add_edge, del_edge;
    auto history_labels = graph.alloc_history_array<uint64_t>();
    auto start = std::chrono::high_resolution_clock::now();
    uint64_t max_batches = 1000000, num_batches = 0;
    for(uint64_t local_begin=imported_edges;local_begin+batch<=raw_edges_len;local_begin+=batch)
    {
        add_edge_len = 0; del_edge_len = 0;
        add_edge.clear(); del_edge.clear();
        auto local_end = std::min(local_begin+batch, raw_edges_len);

        {
            static std::vector<decltype(graph)::edge_type> added_edges;
            added_edges.reserve(local_end - local_begin);
            std::atomic_uint64_t length(0);
            THRESHOLD_OPENMP_LOCAL("omp parallel for", local_end - local_begin, 1024, 
                for(uint64_t i=local_begin;i<local_end;i++)
                {
                    const auto &e = raw_edges[i];
                    auto old_num = graph.add_edge({e.first, e.second}, true);
                    if(!old_num) added_edges[length.fetch_add(1)] = {e.first, e.second};
                }
            );
            added_edges.resize(length.load());
            graph.update_tree_add<uint64_t, uint64_t>(
                continue_reduce_func,
                update_func,
                active_result_func,
                labels, added_edges, true
            );
        }

        {
            static std::vector<decltype(graph)::edge_type> deled_edges;
            deled_edges.reserve(local_end - local_begin);
            std::atomic_uint64_t length(0);
            THRESHOLD_OPENMP_LOCAL("omp parallel for", local_end - local_begin, 1024, 
                for(uint64_t i=local_begin;i<local_end;i++)
                {   
                    const auto &e = raw_edges[i-imported_edges];
                    auto old_num = graph.del_edge({e.first, e.second}, true);
                    if(old_num==1) deled_edges[length.fetch_add(1)] = {e.first, e.second};
                }
            );
            deled_edges.resize(length.load());
            graph.update_tree_del<uint64_t, uint64_t>(
                init_label_func,
                continue_reduce_func,
                update_func,
                active_result_func,
                equal_func,
                labels, deled_edges, true
            );
        }
        if(++num_batches >= max_batches) break;
    }
    auto end = std::chrono::high_resolution_clock::now();
    uint64_t wall_nanoseconds = (end-start).count();
    fprintf(stderr, "wall_times = %lf us\n", (1e-3)*wall_nanoseconds);
    fprintf(stderr, "wall_mean = %lf us\n", (1e-3)*wall_nanoseconds/num_batches);

    return 0;
}
