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
            graph.add_edge({e.first, e.second, (e.first+e.second)%128+1}, true);
        }
        auto end = std::chrono::high_resolution_clock::now();
        fprintf(stderr, "add: %.6lfms\n", 1e-3*(uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(end-start).count());
    }
    
    auto labels = graph.alloc_vertex_tree_array<uint64_t>();
    auto forward_query = graph.alloc_vertex_tree_array<uint64_t>();
    auto backward_query = graph.alloc_vertex_tree_array<uint64_t>();
    auto core_query = graph.alloc_vertex_tree_array<uint64_t>();    
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
    uint64_t root_value = 0;
    uint64_t other_value = MAXL;


    std::atomic_uint64_t add_edge_len(0), del_edge_len(0);
    std::vector<decltype(graph)::edge_type> add_edge, del_edge;
    uint64_t edge_begin, edge_after_add, edge_after_del;
    edge_begin = graph.count_edges();
    std::set<std::pair<uint64_t, uint64_t>> query_specific_core_graph;
    // {
    //     //test outgoing edges
    //     uint64_t node = 1653;
    //     // for (uint64_t i = 0; i < graph.get_outgoing_degree(node); i++)
    //     // {
    //     //     fprintf(stderr,"edge is (%ld, %ld)\n", node, graph.get_dst_number(node, i));
    //     // }
    //     fprintf(stderr,"node %ld has %ld outgoing edges\n", node, graph.get_outgoing_degree(node));
        
    // }
    //create query specific core-graph
    // {

    //         graph.build_tree<uint64_t, uint64_t>(
    //             init_label_func,
    //             continue_reduce_print_func,
    //             update_func,
    //             active_result_func,
    //             forward_query
    //         );
    //         std::vector<bool> outflag(num_vertices);
    //         fprintf(stderr,"finished query\n");
    //         #pragma omp parallel for
    //         for (uint64_t i = 0; i < num_vertices; i++)
    //         {
    //             outflag[i] = false;
    //         }
    //         for (uint64_t i = 0; i < num_vertices; i++)
    //         {
    //             uint64_t degreeV = graph.get_outgoing_degree(i);
    //             // fprintf(stderr,"%ld\n",i);
    //             if ((forward_query[i].data != other_value) && (i != root))
    //             {
    //                 for (uint64_t j = 0; j < degreeV; j++)
    //                 {
    //                     uint64_t dst = graph.get_dst_number(i, j);
    //                     uint64_t edge_len = (i+dst)%128+1;
    //                     if (forward_query[i].data + edge_len == forward_query[dst].data)
    //                     {
    //                         if (graph.check_edge({i,dst}) == true)
    //                         {
    //                             query_specific_core_graph.insert(std::make_pair(i,dst));
    //                             outflag[i] = true;
    //                             // inflag[dst] = true;                                
    //                         }
    //                         else{fprintf(stderr,"wrong edge (%ld, %ld)\n", i, dst);}
    //                     }
    //                 }
    //             }
    //             else if(i == root){
    //                 for (uint64_t j = 0; j < degreeV; j++)
    //                 {
    //                     uint64_t dst = graph.get_dst_number(i, j);
    //                     uint64_t edge_len = (i+dst)%128+1;
    //                     if (forward_query[i].data + edge_len == forward_query[dst].data)
    //                     {
    //                         if (graph.check_edge({i,dst}) == true)
    //                         {
    //                             query_specific_core_graph.insert(std::make_pair(i,dst));
    //                             outflag[i] = true;
    //                             // inflag[dst] = true;                                
    //                         }
    //                         else{fprintf(stderr,"wrong edge (%ld, %ld)\n", i, dst);}

    //                     }
    //                 }                    
    //             }
    //         }
    //         fprintf(stderr,"finished sampling\n");
    //         for (uint64_t i = 0; i < num_vertices; i++)
    //         {
    //             if (outflag[i] == false)
    //             {
    //                 if (graph.get_outgoing_degree(i)!=0)
    //                 {
    //                     uint64_t dst = graph.get_dst_number(i, 0);
    //                     query_specific_core_graph.insert(std::make_pair(i, dst));
    //                 }
    //             }
    //         }
    //     fprintf(stderr, "we have %ld edges in query specific core graph, and we have %ld edges in whole graph\n", query_specific_core_graph.size(), raw_edges_len);
    //     Graph<uint64_t> core_graph(num_vertices, query_specific_core_graph.size(), false, true);
    //     for (auto e: query_specific_core_graph)
    //     {
    //         core_graph.add_edge({e.first, e.second, (e.first+e.second)%128+1}, true);
    //     }
    //     query_specific_core_graph.clear();
    //         core_graph.build_tree<uint64_t, uint64_t>(
    //             init_label_func,
    //             continue_reduce_print_func,
    //             update_func,
    //             active_result_func,
    //             core_query
    //         );
    //         uint64_t count = 0;
    //         for (uint64_t i = 0; i < num_vertices; i++)
    //         {
    //             if (forward_query[i].data != core_query[i].data)
    //             {
    //                 count++;
    //             }
    //         }
    //         if (count != 0)
    //         {
    //             fprintf(stderr,"%ld results not correct\n",count);    
    //         }
    //         else{fprintf(stderr,"all results are correct\n");}
                            
    // }


    for (int round = 1; round < compute_batch+1; round++)
    {
        uint64_t batch_current = batch * round;
        for (int small_round = 1; small_round < 10; small_round++)
        {
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
                    auto old_num = graph.del_edge({e.first, e.second, (e.first+e.second)%128+1}, true);
                }
            );
        auto del_mutation_end = std::chrono::high_resolution_clock::now();
        // printf("after deletion %ld edges left\n",graph.count_edges());
        fprintf(stderr, "del mutation time: %.6lfms\n", 1e-3*(uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(del_mutation_end- del_mutation_start).count());               
                for(uint64_t i=0;i<batch_current;i++)
                {
                    const auto &e = raw_edges[i+seed];
                    deled_edges[length.fetch_add(1)] = {e.first, e.second, (e.first+e.second)%128+1};   
                }
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
    // sampling smaller graph

            graph.build_tree<uint64_t, uint64_t>(
                init_label_func,
                continue_reduce_print_func,
                update_func,
                active_result_func,
                forward_query
            );
            std::vector<bool> outflag(num_vertices);
            fprintf(stderr,"finished query\n");
            #pragma omp parallel for
            for (uint64_t i = 0; i < num_vertices; i++)
            {
                outflag[i] = false;
            }
            for (uint64_t i = 0; i < num_vertices; i++)
            {
                uint64_t degreeV = graph.get_outgoing_degree(i);
                // fprintf(stderr,"%ld\n",i);
                if ((forward_query[i].data != other_value) && (i != root))
                {
                    for (uint64_t j = 0; j < degreeV; j++)
                    {
                        uint64_t dst = graph.get_dst_number(i, j);
                        uint64_t edge_len = (i+dst)%128+1;
                        if (forward_query[i].data + edge_len == forward_query[dst].data)
                        {
                            if (graph.check_edge({i,dst}) == true)
                            {
                                query_specific_core_graph.insert(std::make_pair(i,dst));
                                outflag[i] = true;
                                // inflag[dst] = true;                                
                            }
                            else{fprintf(stderr,"wrong edge (%ld, %ld)\n", i, dst);}
                        }
                    }
                }
                else if(i == root){
                    for (uint64_t j = 0; j < degreeV; j++)
                    {
                        uint64_t dst = graph.get_dst_number(i, j);
                        uint64_t edge_len = (i+dst)%128+1;
                        if (forward_query[i].data + edge_len == forward_query[dst].data)
                        {
                            if (graph.check_edge({i,dst}) == true)
                            {
                                query_specific_core_graph.insert(std::make_pair(i,dst));
                                outflag[i] = true;
                                // inflag[dst] = true;                                
                            }
                            else{fprintf(stderr,"wrong edge (%ld, %ld)\n", i, dst);}

                        }
                    }                    
                }
            }
            fprintf(stderr,"finished sampling\n");
            for (uint64_t i = 0; i < num_vertices; i++)
            {
                if (outflag[i] == false)
                {
                    if (graph.get_outgoing_degree(i)!=0)
                    {
                        uint64_t dst = graph.get_dst_number(i, 0);
                        query_specific_core_graph.insert(std::make_pair(i, dst));
                    }
                }
            }
        fprintf(stderr, "we have %ld edges in query specific core graph, and we have %ld edges in whole graph\n", query_specific_core_graph.size(), raw_edges_len);
        Graph<uint64_t> core_graph(num_vertices, query_specific_core_graph.size(), false, true);
        for (auto e: query_specific_core_graph)
        {
            core_graph.add_edge({e.first, e.second, (e.first+e.second)%128+1}, true);
        }
        query_specific_core_graph.clear();
            core_graph.build_tree<uint64_t, uint64_t>(
                init_label_func,
                continue_reduce_print_func,
                update_func,
                active_result_func,
                core_query
            );
            uint64_t count = 0;
            for (uint64_t i = 0; i < num_vertices; i++)
            {
                if (forward_query[i].data != core_query[i].data)
                {
                    count++;
                }
            }
            if (count != 0)
            {
                fprintf(stderr,"%ld results not correct\n",count);    
            }
            else{fprintf(stderr,"all results are correct for deleted graph and its core graph\n");}
            query_specific_core_graph.clear();


        //add edges and doing addition computation        
    {
            std::vector<decltype(graph)::edge_type> added_edges(batch_current);    
            // printf("before addition %ld edges left\n",graph.count_edges());
                auto add_mutation_start = std::chrono::high_resolution_clock::now();            
                THRESHOLD_OPENMP_LOCAL("omp parallel for", batch_current, 1024, 
                    for(uint64_t i=0;i<batch_current;i++)
                    {
                        const auto &e = raw_edges[i+seed];
                        auto old_num = graph.add_edge({e.first, e.second, (e.first+e.second)%128+1}, true);
                    }
                );
            auto add_mutation_end = std::chrono::high_resolution_clock::now();
            fprintf(stderr, "add mutation time: %.6lfms\n", 1e-3*(uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(add_mutation_end- add_mutation_start).count());
            // addition batch_current updates
            std::atomic_uint64_t length(0);
            for (uint64_t i = 0; i < batch_current; i++)
            {
                const auto &e = raw_edges[i+seed];
                added_edges[length.fetch_add(1)] = {e.first, e.second, (e.first+e.second)%128+1};
            }
            auto add_compute_start = std::chrono::high_resolution_clock::now();
                graph.update_tree_add<uint64_t, uint64_t>(
                    continue_reduce_func,
                    update_func,
                    active_result_func,
                    labels, added_edges, length.load(), true
                );
            auto add_compute_end = std::chrono::high_resolution_clock::now();
            fprintf(stderr, "add compute: %.6lfms\n", 1e-3*(uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(add_compute_end-add_compute_start).count());
            
            auto core_add_mutation_start = std::chrono::high_resolution_clock::now();            
            THRESHOLD_OPENMP_LOCAL("omp parallel for", batch_current, 1024, 
                    for(uint64_t i=0;i<batch_current;i++)
                    {
                        const auto &e = raw_edges[i+seed];
                        auto old_num = core_graph.add_edge({e.first, e.second, (e.first+e.second)%128+1}, true);
                    }
                );
            auto core_add_mutation_end = std::chrono::high_resolution_clock::now();
            fprintf(stderr, "Core add mutation time: %.6lfms\n", 1e-3*(uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(core_add_mutation_end- core_add_mutation_start).count());            
            
            auto core_add_compute_start = std::chrono::high_resolution_clock::now();
                core_graph.update_tree_add<uint64_t, uint64_t>(
                    continue_reduce_func,
                    update_func,
                    active_result_func,
                    core_query, added_edges, length.load(), true
                );
            auto core_add_compute_end = std::chrono::high_resolution_clock::now();
            fprintf(stderr, "Core add compute: %.6lfms\n", 1e-3*(uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(core_add_compute_end - core_add_compute_start).count());
        }
        edge_after_add = graph.count_edges();
        if (edge_after_add != edge_begin)
        {
            fprintf(stderr ,"graph is not recovered! The edge left is %ld\n", (edge_begin - edge_after_add));
        }
            count = 0;
            for (uint64_t i = 0; i < num_vertices; i++)
            {
                if (forward_query[i].data != core_query[i].data)
                {
                    count++;
                }
            }
            if (count != 0)
            {
                fprintf(stderr,"%ld results not correct\n",count);    
            }
            else{fprintf(stderr,"all results are correct\n");}        
        }
        
        fprintf(stderr, "$$$$$$$$$$$$$$$ %d round is over $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n", round);
    }
    
    
    return 0;
}
