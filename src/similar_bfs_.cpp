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
    if(argc <= 5)
    {
        fprintf(stderr, "usage: %s graph root batch_size random_seed compute_batch\n", argv[0]);
        exit(1);
    }
    std::pair<uint64_t, uint64_t> *raw_edges;
    uint64_t root = std::stoull(argv[2]);
    uint64_t batch = std::stoull(argv[3]);
    uint64_t random_seed = std::stoull(argv[4]);
    int compute_batch = std::stoull(argv[5]);
    std::srand(random_seed);
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
    Graph<uint64_t> graph(num_vertices, raw_edges_len, false, true);
    double imported_rate = 1;
    uint64_t imported_edges = raw_edges_len*imported_rate;
    {
        auto start = std::chrono::high_resolution_clock::now();
        #pragma omp parallel for
        for(uint64_t i=0;i<imported_edges;i++)
        {
            const auto &e = raw_edges[i];
            graph.add_edge({e.first, e.second, 1}, true);
        }
        auto end = std::chrono::high_resolution_clock::now();
        fprintf(stderr, "add: %.6lfms\n", 1e-3*(uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(end-start).count());
    }
    std::vector<uint64_t> track_before;
    std::vector<uint64_t> track_after;
    track_before.resize(num_vertices);
    // track_after.resize(num_vertices);
    for (uint64_t i = 0; i < num_vertices; i++)
    {
        track_before[i] = 0;
        // track_after[i] = 0;
    }
    
    auto labels = graph.alloc_vertex_tree_array<uint64_t>();
    const uint64_t MAXL = 134217728;
    auto continue_reduce_func = [](uint64_t depth, uint64_t total_result, uint64_t local_result) -> std::pair<bool, uint64_t>
    {
        return std::make_pair(local_result>0, total_result+local_result);
    };
    auto continue_reduce_print_func = [](uint64_t depth, uint64_t total_result, uint64_t local_result) -> std::pair<bool, uint64_t>
    {
        // fprintf(stderr, "active(%lu) >= %lu\n", depth, local_result);
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

    // {
    //     auto start = std::chrono::high_resolution_clock::now();

    //     graph.build_tree<uint64_t, uint64_t>(
    //         init_label_func,
    //         continue_reduce_print_func,
    //         update_func,
    //         active_result_func,
    //         labels
    //     );

    //     auto end = std::chrono::high_resolution_clock::now();
    //     fprintf(stderr, "init exec: %.6lfs\n", 1e-6*(uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(end-start).count());
    // }

    std::atomic_uint64_t add_edge_len(0), del_edge_len(0);
    std::vector<decltype(graph)::edge_type> add_edge, del_edge;
    // std::vector<std::pair<uint64_t, uint64_t>> edges_changed;
    // edges_changed.reserve(batch);
    // uint64_t seed = std::rand() % (raw_edges_len - batch);
    uint64_t edge_begin, edge_after_add, edge_after_del;
    edge_begin = graph.count_edges();
    for (int round = 1; round < compute_batch+1; round++)
    {
        uint64_t batch_current = batch * round;
        std::vector<_Float64> ratio;
        ratio.resize(9);
        for (int small_round = 1; small_round < 10; small_round++)
        {
        // printf("*********************************************\n");
        fprintf(stderr,"*********************************************\n");
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
            fprintf(stderr, "Init exec: %.6lfms\n", 1e-3*(uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(end-start).count());
        }                
            fprintf(stderr, "This round the batch has %ld edges\n",batch_current);
            // printf("Begining Graph has %ld edges\n",graph.count_edges());
            srand(random_seed * small_round);
            // uint64_t batch_current = batch * round;
            uint64_t seed = std::rand() % (raw_edges_len - batch_current);
    // del edges and doing deletion computation
    {
        // printf("before deletion %ld edges left\n",graph.count_edges());
        std::vector<decltype(graph)::edge_type> deled_edges(batch_current);
        std::atomic_uint64_t length(0);
            auto del_mutation_start = std::chrono::high_resolution_clock::now();            
            THRESHOLD_OPENMP_LOCAL("omp parallel for", batch_current, 1024, 
                for(uint64_t i=0;i<batch_current;i++)
                {   
                    const auto &e = raw_edges[i+seed];
                    auto old_num = graph.del_edge({e.first, e.second, 1}, true);
                }
            );
        auto del_mutation_end = std::chrono::high_resolution_clock::now();
        // printf("after deletion %ld edges left\n",graph.count_edges());
        fprintf(stderr, "del mutation time: %.6lfms\n", 1e-3*(uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(del_mutation_end- del_mutation_start).count());               
                for(uint64_t i=0;i<batch_current;i++)
                {
                    const auto &e = raw_edges[i+seed];
                    deled_edges[length.fetch_add(1)] = {e.first, e.second, 1};   
                }
        // std::cout<< length.load()<<std::endl;                
        auto del_compute_start = std::chrono::high_resolution_clock::now();            
            graph.update_tree_del<uint64_t, uint64_t>(
                init_label_func,
                continue_reduce_func,
                update_func,
                active_result_func,
                equal_func,
                labels, deled_edges, length.load(), true
            );
        auto del_compute_end = std::chrono::high_resolution_clock::now();            
        fprintf(stderr, "del compute time: %.6lfms\n", 1e-3*(uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(del_compute_end- del_compute_start).count());                                 
    }
    edge_after_del = graph.count_edges();
        // printf("Before addition, Graph has %ld edges\n",graph.count_edges());

        //add edges and doing addition computation        
    {
            std::vector<decltype(graph)::edge_type> added_edges(batch_current);    
            // printf("before addition %ld edges left\n",graph.count_edges());
                auto add_mutation_start = std::chrono::high_resolution_clock::now();            
                THRESHOLD_OPENMP_LOCAL("omp parallel for", batch_current, 1024, 
                    for(uint64_t i=0;i<batch_current;i++)
                    {
                        const auto &e = raw_edges[i+seed];
                        auto old_num = graph.add_edge({e.first, e.second, 1}, true);
                    }
                );
            auto add_mutation_end = std::chrono::high_resolution_clock::now();
            // printf("After addition %ld edges left\n",graph.count_edges());
            fprintf(stderr, "add mutation time: %.6lfms\n", 1e-3*(uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(add_mutation_end- add_mutation_start).count());
            // addition batch_current updates
            std::atomic_uint64_t length(0);
            for (uint64_t i = 0; i < batch_current; i++)
            {
                const auto &e = raw_edges[i+seed];
                added_edges[length.fetch_add(1)] = {e.first, e.second, 1};
            }
            fprintf(stderr, "Before compute %ld nodes are activate %ld nodes are total\n", graph.count_activate(), graph.nodes_track.size());
            uint64_t tracker_count = 0;
            uint64_t edge_similar = 0;
            uint64_t edge_traverse = 0;
            auto add_compute_start = std::chrono::high_resolution_clock::now();
                graph.update_tree_add<uint64_t, uint64_t>(
                    continue_reduce_func,
                    update_func,
                    active_result_func,
                    labels, added_edges, length.load(), true
                );
            auto add_compute_end = std::chrono::high_resolution_clock::now();
            fprintf(stderr, "add compute: %.6lfms\n", 1e-3*(uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(add_compute_end-add_compute_start).count());
            fprintf(stderr, "After compute %ld nodes are activate %ld nodes are total\n", graph.count_activate(), graph.nodes_track.size());
            if (small_round == 1)
            {
                //first round only change track_before, don't compare
                for (uint64_t tracker = 0; tracker < num_vertices; tracker++)
                {
                    track_before[tracker] = graph.nodes_track[tracker];
                }
            }
            else{
                //first compare tracker_before and tracker now
                for (uint64_t tracker = 0; tracker < num_vertices; tracker++)
                {
                    if((track_before[tracker] == 1) && (graph.nodes_track[tracker] == 1)){
                        tracker_count++;
                        edge_similar += graph.count_out_going(tracker);
                    }
                    if (graph.nodes_track[tracker] == 1)
                    {
                        edge_traverse += graph.count_out_going(tracker);
                    }
                    
                    track_before[tracker] =  graph.nodes_track[tracker];
                }
                ratio[small_round] = static_cast<_Float64>(edge_similar)/edge_traverse;                
            }
            graph.clear_track();
            fprintf(stderr, "last round and this round has %ld ndoes similar, after reset %ld nodes are still active in graph.tracker\n", tracker_count, graph.count_activate());
            fprintf(stderr, "edge similar is %ld, edge traverse is %ld\n", edge_similar, edge_traverse);
        }
        edge_after_add = graph.count_edges();
        if (edge_after_add != edge_begin)
        {
            fprintf(stderr ,"graph is not recovered! The edge left is %ld\n", (edge_begin - edge_after_add));
        }
        }
        auto max_s = std::max_element(ratio.begin(), ratio.end());
        fprintf(stderr, "The maximim similarity is %f %\n", (*max_s)*100);
        // for ( auto i:ratio)
        // {
        //     std::cout << i << std::endl;
        // }
        
        fprintf(stderr, "$$$$$$$$$$$$$$$ %d round is over $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n", round);
    }
    
    // printf("Graph has %ld edges\n",graph.count_edges());
    // {
    //         std::atomic_uint64_t length(0);
    //         // printf("before deletion %ld edges in graph\n",graph.count_edges());
    //         // auto del_mutation_start = std::chrono::high_resolution_clock::now();            
    //         THRESHOLD_OPENMP_LOCAL("omp parallel for", batch, 1024, 
    //             for(uint64_t i=0;i<batch;i++)
    //             {   
    //                 const auto &e = raw_edges[i+seed];
    //                 auto old_num = graph.del_edge({e.first, e.second, 1}, true);
    //             }
    //         );
    // }
    // printf("At begin graph has %ld edges, and edge length is %ld\n",graph.count_edges(), raw_edges_len);
    
//     edge_begin = graph.count_edges();
//     {
//         auto start = std::chrono::high_resolution_clock::now();
//         graph.build_tree<uint64_t, uint64_t>(
//             init_label_func,
//             continue_reduce_print_func,
//             update_func,
//             active_result_func,
//             labels
//         );
//         auto end = std::chrono::high_resolution_clock::now();
//         fprintf(stderr, "init exec: %.6lfms\n", 1e-3*(uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(end-start).count());
//     }
//     //add edges and doing addition computation        
//    {
//         std::vector<decltype(graph)::edge_type> added_edges(batch);    
//         // printf("before addition %ld edges left\n",graph.count_edges());
//             auto add_mutation_start = std::chrono::high_resolution_clock::now();            
//             THRESHOLD_OPENMP_LOCAL("omp parallel for", batch, 1024, 
//                 for(uint64_t i=0;i<batch;i++)
//                 {
//                     const auto &e = raw_edges[i+seed];
//                     auto old_num = graph.add_edge({e.first, e.second, 1}, true);
//                 }
//             );
//         auto add_mutation_end = std::chrono::high_resolution_clock::now();
//         // printf("after addition %ld edges left\n",graph.count_edges());
//         fprintf(stderr, "add mutation time: %.6lfms\n", 1e-3*(uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(add_mutation_end- add_mutation_start).count());
//         // addition batch updates
//         std::atomic_uint64_t length(0);
//         for (uint64_t i = 0; i < batch; i++)
//         {
//             const auto &e = raw_edges[i+seed];
//             added_edges[length.fetch_add(1)] = {e.first, e.second, 1};
//         }
//         auto add_compute_start = std::chrono::high_resolution_clock::now();
//             graph.update_tree_add<uint64_t, uint64_t>(
//                 continue_reduce_func,
//                 update_func,
//                 active_result_func,
//                 labels, added_edges, length.load(), true
//             );
//         auto add_compute_end = std::chrono::high_resolution_clock::now();
//         fprintf(stderr, "add compute: %.6lfms\n", 1e-3*(uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(add_compute_end-add_compute_start).count());        
//    }
//     edge_after_add = graph.count_edges();

    // del edges and doing deletion computation
    // {
    //     // printf("before deletion %ld edges left\n",graph.count_edges());
    //     std::vector<decltype(graph)::edge_type> deled_edges(batch);
    //     std::atomic_uint64_t length(0);
    //         auto del_mutation_start = std::chrono::high_resolution_clock::now();            
    //         THRESHOLD_OPENMP_LOCAL("omp parallel for", batch, 1024, 
    //             for(uint64_t i=0;i<batch;i++)
    //             {   
    //                 const auto &e = raw_edges[i+seed];
    //                 auto old_num = graph.del_edge({e.first, e.second, 1}, true);
    //             }
    //         );
    //     auto del_mutation_end = std::chrono::high_resolution_clock::now();
    //     // printf("after deletion %ld edges left\n",graph.count_edges());
    //     fprintf(stderr, "del mutation time: %.6lfms\n", 1e-3*(uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(del_mutation_end- del_mutation_start).count());               
    //             for(uint64_t i=0;i<batch;i++)
    //             {
    //                 const auto &e = raw_edges[i+seed];
    //                 deled_edges[length.fetch_add(1)] = {e.first, e.second, 1};   
    //             }
    //     // std::cout<< length.load()<<std::endl;                
    //     auto del_compute_start = std::chrono::high_resolution_clock::now();            
    //         graph.update_tree_del<uint64_t, uint64_t>(
    //             init_label_func,
    //             continue_reduce_func,
    //             update_func,
    //             active_result_func,
    //             equal_func,
    //             labels, deled_edges, length.load(), true
    //         );
    //     auto del_compute_end = std::chrono::high_resolution_clock::now();            
    //     fprintf(stderr, "del compute time: %.6lfms\n", 1e-3*(uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(del_compute_end- del_compute_start).count());                                 
    // }
    // edge_after_del = graph.count_edges();
    // if ((edge_after_del!=edge_begin) || (edge_after_add - edge_begin != batch))
    // {
    //     printf("Graph Mutation not sucess, the difference is %ld, %ld\n", (edge_after_del - edge_begin), (edge_after_add - edge_begin));
    // }
    
    return 0;
}
