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
    std::vector<long long> add_compute(compute_batch);
    std::vector<long long> add_core_compute(compute_batch);
    for (int z = 0; z < compute_batch; z++)
    {
        add_compute[z] = 0;
        add_core_compute[z] = 0;
    }        

    for (int round = 0; round < compute_batch; round++)
    {
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
    // Graph<uint64_t> common_graph(num_vertices, raw_edges_len, false, true);
    srand(random_seed*round);
    std::vector<uint64_t> tqdm(10);
    // tqdm.resize(10);
    for (size_t i = 0; i < 10; i++)
    {
        tqdm[i] = (num_vertices)/(10-i);
        // fprintf(stderr,"%ld\n",tqdm[i]);
    }

    uint64_t seed = std::rand()%(raw_edges_len - batch*compute_batch);
    // start from seed to seed+batch*compute_batch-1 for edge sampling
    uint64_t imported_edges = raw_edges_len - batch*compute_batch;


    auto label_init = graph.alloc_vertex_tree_array<uint64_t>();
    auto label_add = graph.alloc_vertex_tree_array<uint64_t>();
    auto label_del = graph.alloc_vertex_tree_array<uint64_t>();
    auto core_query = graph.alloc_vertex_tree_array<uint64_t>();
    auto core_query_check = graph.alloc_vertex_tree_array<uint64_t>();
    auto check = graph.alloc_vertex_tree_array<uint64_t>();
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

            
        {       
            auto start = std::chrono::high_resolution_clock::now();
            #pragma omp parallel for
            for(uint64_t i=0;i<raw_edges_len;i++)
            {
                const auto &e = raw_edges[i];
                graph.add_edge({e.first, e.second, (e.first+e.second)%32+1}, true);
            }
            // #pragma omp parallel for
            // for (uint64_t i = seed+compute_batch*batch; i < raw_edges_len; i++)
            // {
            //     const auto &e = raw_edges[i];
            //     graph.add_edge({e.first, e.second, (e.first+e.second)%32+1}, true);            
            // }
            auto end = std::chrono::high_resolution_clock::now();
            fprintf(stderr, "add: %.6lfms\n", 1e-3*(uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(end-start).count());
        }    
        fprintf(stderr,"%ld small graph edges, %ld not being added\n", graph.count_edges(), raw_edges_len-graph.count_edges());
        std::atomic_uint64_t add_edge_len(0), del_edge_len(0), re_edge_len(0), sample_edge_len(0);
        std::vector<decltype(graph)::edge_type> add_edge(compute_batch*batch);
        std::vector<decltype(graph)::edge_type> del_edge(compute_batch*batch);
        std::vector<decltype(graph)::edge_type> sampledd_edge(compute_batch*batch);
        std::vector<std::pair<uint64_t, uint64_t>> modified_edges(compute_batch*batch);
        uint64_t edge_begin, edge_after_add, edge_after_del;
        uint64_t batch_current = compute_batch*batch;
        uint64_t count = 0;
        edge_begin = graph.count_edges();
        std::set<std::pair<uint64_t, uint64_t>> query_specific_core_graph;
        std::vector<std::pair<uint64_t, uint64_t>> Q_Core;
        Graph<uint64_t> core_graph(num_vertices, query_specific_core_graph.size(), false, true);
        // create deletion and addition batch
        {
                std::atomic_uint64_t length_del(0);
                for(uint64_t i=seed;i<seed+batch_current;i++)
                {
                    const auto &e = raw_edges[i];
                    del_edge[length_del.fetch_add(1)] = {e.first, e.second, (e.first+e.second)%32+1};
                    modified_edges[length_del] = std::make_pair(e.first,e.second);   
                }
                std::atomic_uint64_t length_add(0);                    
                for(uint64_t i=seed;i<seed+batch_current;i++)
                {
                    const auto &e = raw_edges[i];
                    add_edge[length_add.fetch_add(1)] = {e.first, e.second, (e.first+e.second)%32+1};   
                }
                fprintf(stderr,"We have %ld add batch and %ld del batch\n",add_edge.size(),del_edge.size());
            auto start = std::chrono::high_resolution_clock::now();
            graph.build_tree<uint64_t, uint64_t>(
                init_label_func,
                continue_reduce_print_func,
                update_func,
                active_result_func,
                label_init
                );
            auto end = std::chrono::high_resolution_clock::now();
            fprintf(stderr, "Init exec: %.6lfms\n", 1e-3*(uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(end-start).count());            
            graph.build_tree<uint64_t, uint64_t>(
                init_label_func,
                continue_reduce_print_func,
                update_func,
                active_result_func,
                label_del
                );            


            //del edges now
            auto del_mutation_start = std::chrono::high_resolution_clock::now();            
            THRESHOLD_OPENMP_LOCAL("omp parallel for", batch_current, 1024, 
                for(uint64_t i=seed;i<seed+batch_current;i++)
                {
                    const auto &e = raw_edges[i];
                    auto old_num = graph.del_edge({e.first, e.second, (e.first+e.second)%32+1}, true);
                }
            );
            auto del_mutation_end = std::chrono::high_resolution_clock::now();
            fprintf(stderr, "del mutation time: %.6lfms\n", 1e-3*(uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(del_mutation_end- del_mutation_start).count());        
            auto del_compute_start = std::chrono::high_resolution_clock::now();
                graph.update_tree_del<uint64_t, uint64_t>(
                    init_label_func,
                    continue_reduce_func,
                    update_func,
                    active_result_func,
                    equal_func,
                    label_del, del_edge, length_del.load(), true
                );
            auto del_compute_end = std::chrono::high_resolution_clock::now();
            fprintf(stderr, "del compute: %.6lfms\n", 1e-3*(uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(del_compute_end- del_compute_start).count());            
            // Now is Common Graph

            //sample core graph
            {graph.build_tree<uint64_t, uint64_t>(
                init_label_func,
                continue_reduce_print_func,
                update_func,
                active_result_func,
                label_add
                );
            // label_add and label_del is common graph results here            
            uint64_t wrong = 0;
            int tqdm_ = 0;
            for (uint64_t src = 0; src < num_vertices; src++)
            {
                uint64_t degreeV = graph.get_outgoing_degree(src);
                uint64_t degreeW = graph.get_incoming_degree(src);
                // fprintf(stderr,"%ld\n",src);
                if (src == tqdm[tqdm_])
                {
                    tqdm_++;
                    fprintf(stderr,"Now is sample for %d 10 percantage of nodes\n",tqdm_);
                }
                for (uint64_t j = 0; j < degreeV; j++)
                {
                    uint64_t dst = graph.get_dst_number(src, j);
                    uint64_t edge_len = (src+dst)%32+1;
                    if (label_del[src].data == other_value)
                    {
                        Q_Core.push_back(std::make_pair(src, dst));
                    } 
                    if (label_del[src].data + edge_len == label_del[dst].data)
                    {
                        Q_Core.push_back(std::make_pair(src, dst));
                    }
                }            
                if (label_del[src].data == other_value)
                {
                    for (uint64_t j = 0; j < degreeW; j++)
                    {
                        uint64_t dst = graph.get_src_number(src, j);
                        uint64_t edge_len = (src+dst)%32+1;
                        // query_specific_core_graph.insert(std::make_pair(dst,src));
                        Q_Core.push_back(std::make_pair(dst, src));

                    }
                }
            }
            for (uint64_t i = 0; i < num_vertices; i++)
            {
                if (label_init[i].data != label_add[i].data)
                {
                    uint64_t degreeV = graph.get_outgoing_degree(i);
                    uint64_t degreeW = graph.get_incoming_degree(i);
                    for (size_t j = 0; j < degreeV; j++)
                    {
                        uint64_t dst = graph.get_dst_number(i, j);
                        uint64_t edge_len = (i+dst)%32+1;
                        Q_Core.push_back(std::make_pair(i, dst));
                    }
                    for (size_t j = 0; j < degreeW; j++)
                    {
                        uint64_t dst = graph.get_src_number(i, j);
                        uint64_t edge_len = (i+dst)%32+1;
                        Q_Core.push_back(std::make_pair(dst, i));
                    }                    
                }
                
            }            
            fprintf(stderr,"before check %ld edges in Q_core\n",Q_Core.size());
            for (auto e: Q_Core)
            {
                if(std::find(modified_edges.begin(), modified_edges.end(), e) != modified_edges.end()){
                    wrong++;
                    // fprintf(stderr, "This edge <%ld %ld> is in deletion batch\n",e.first, e.second);
                    Q_Core.erase(std::find(Q_Core.begin(), Q_Core.end(), e));
                }
            }
            fprintf(stderr,"After check %ld edges in Q_core\n",Q_Core.size());
            fprintf(stderr,"we have %ld edges sampled from deleted edges\n",wrong);
            fprintf(stderr,"out Going edges finished sampling\n");
            fprintf(stderr, "we have %ld edges in query specific core graph, and we have %ld edges in whole graph\n", Q_Core.size(), raw_edges_len);        
            for (auto e: Q_Core)
            {
                core_graph.add_edge({e.first, e.second, (e.first+e.second)%32+1}, true);
            }             
            
            // fprintf(stderr,"Before Resample %ld edgs in core graph\n",core_graph.count_edges());
            // for (uint64_t i = 0; i < num_vertices; i++)
            // {
            //     if (label_init[i].data != label_add[i].data)
            //     {
            //         uint64_t degreeV = graph.get_outgoing_degree(i);
            //         uint64_t degreeW = graph.get_incoming_degree(i);
            //         for (size_t j = 0; j < degreeV; j++)
            //         {
            //             uint64_t dst = graph.get_dst_number(i, j);
            //             uint64_t edge_len = (i+dst)%32+1;
            //             auto old_num = core_graph.add_edge({i, dst, (i+dst)%32+1}, true);
            //         }
                    
            //     }
                
            // }
            // fprintf(stderr,"After Resample %ld edgs in core graph\n",core_graph.count_edges());            

            }
            core_graph.build_tree<uint64_t, uint64_t>(
                init_label_func,
                continue_reduce_print_func,
                update_func,
                active_result_func,
                core_query
            );            
            uint64_t core_count = 0;
            for (uint64_t i = 0; i < num_vertices; i++)
            {
                if (label_del[i].data != core_query[i].data)
                {
                    core_count++;
                }
            }
            fprintf(stderr,"%ld nodes not correct in core graph\n",core_count);
            core_count = 0;
            for (uint64_t i = 0; i < num_vertices; i++)
            {
                if (label_del[i].data != label_add[i].data)
                {
                    core_count++;
                }
            }
            fprintf(stderr,"%ld nodes not correct in Original graph Strem\n",core_count);                          
            {// added edges and count time
            auto add_mutation_start = std::chrono::high_resolution_clock::now();            
            THRESHOLD_OPENMP_LOCAL("omp parallel for", batch_current, 1024, 
                for(uint64_t i=seed;i<seed+batch_current;i++)
                {
                    const auto &e = raw_edges[i];
                    auto old_num = graph.add_edge({e.first, e.second, (e.first+e.second)%32+1}, true);
                }
            );
            auto add_mutation_end = std::chrono::high_resolution_clock::now();
            fprintf(stderr, "add mutation time: %.6lfms\n", 1e-3*(uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(add_mutation_end- add_mutation_start).count());        
            auto add_compute_start = std::chrono::high_resolution_clock::now();
                graph.update_tree_add<uint64_t, uint64_t>(
                    continue_reduce_func,
                    update_func,
                    active_result_func,
                    label_add, add_edge, length_add.load(), true
                );
            auto add_compute_end = std::chrono::high_resolution_clock::now();
            fprintf(stderr, "add compute: %.6lfms\n", 1e-3*(uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(add_compute_end-add_compute_start).count());}

            // add_compute[round] = std::chrono::duration_cast<std::chrono::microseconds>(add_compute_end-add_compute_start).count();
            // core graph added edges
            fprintf(stderr,"core graph has %ld edges now\n",core_graph.count_edges());
            auto core_add_mutation_start = std::chrono::high_resolution_clock::now();            
            THRESHOLD_OPENMP_LOCAL("omp parallel for", batch_current, 1024, 
                for(uint64_t i=seed;i<seed+batch_current;i++)
                {
                    const auto &e = raw_edges[i];
                    // const auto &e = modified_edges[i];
                    auto new_num = core_graph.add_edge({e.first, e.second, (e.first+e.second)%32+1}, true);
                }
            );
            auto core_add_mutation_end = std::chrono::high_resolution_clock::now();
            // fprintf(stderr, "Core add mutation time: %.6lfms\n", 1e-3*(uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(core_add_mutation_end- core_add_mutation_end).count());
            fprintf(stderr,"core graph has %ld edges now\n",core_graph.count_edges());        
            auto core_add_compute_start = std::chrono::high_resolution_clock::now();
                core_graph.update_tree_add<uint64_t, uint64_t>(
                    continue_reduce_func,
                    update_func,
                    active_result_func,
                    core_query, add_edge, length_add.load(), true
                );
            auto core_add_compute_end = std::chrono::high_resolution_clock::now();
            fprintf(stderr, "Core add compute: %.6lfms\n", 1e-3*(uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(core_add_compute_end - core_add_compute_start).count());
            add_core_compute[round] = std::chrono::duration_cast<std::chrono::microseconds>(core_add_compute_end - core_add_compute_start).count();
            query_specific_core_graph.clear();
            uint64_t count = 0;
            for (uint64_t i = 0; i < num_vertices; i++)
            {
                if (core_query[i].data != label_add[i].data)
                {
                    count++;
                    // fprintf(stderr,"%ld nodes is not correct, core: %ld, Graph: %ld\n", i, core_query[i].data, label_add[i].data);
                }
                
            }
            fprintf(stderr,"%ld number of vertices not correct\n",count);



        }        
    }
    // long long add___ = 0;
    // long long core___ = 0;
    // for (int i = 0; i < compute_batch; i++)
    // {
    //     if (add_compute[i] != 0)
    //     {
    //         add___ += add_compute[i];
    //     }
    //     else{fprintf(stderr,"WRONG\n");}
    //     if (add_core_compute[i] != 0)
    //     {
    //         core___ += add_core_compute[i];
    //     }
    //     else{fprintf(stderr,"WRONG\n");}
    // }
    // add___ = add___/compute_batch;
    // core___ = core___/compute_batch;
    // // fprintf(stderr, "%.6lfms\n", 1e-3*(uint64_t)(add___/core___));
    // fprintf(stderr, "whole: %.6lfms\n", 1e-3*(uint64_t)add___);    
    // fprintf(stderr, "Core: %.6lfms\n", 1e-3*(uint64_t)core___);    

    
    return 0;
}
