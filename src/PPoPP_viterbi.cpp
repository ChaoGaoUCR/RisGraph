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
    // Graph<uint64_t> common_graph(num_vertices, raw_edges_len, false, true);
    srand(random_seed);
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
    {
        auto start = std::chrono::high_resolution_clock::now();
        #pragma omp parallel for
        for(uint64_t i=0;i<seed;i++)
        {
            const auto &e = raw_edges[i];
            graph.add_edge({e.first, e.second, (e.first+e.second)%32+1}, true);
        }
        #pragma omp parallel for
        for (uint64_t i = seed+compute_batch*batch; i < raw_edges_len; i++)
        {
            const auto &e = raw_edges[i];
            graph.add_edge({e.first, e.second, (e.first+e.second)%32+1}, true);            
        }
        auto end = std::chrono::high_resolution_clock::now();
        fprintf(stderr, "add: %.6lfms\n", 1e-3*(uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(end-start).count());
    }

    auto label_init = graph.alloc_vertex_tree_array<uint64_t>();
    auto label_add = graph.alloc_vertex_tree_array<uint64_t>();
    auto label_del = graph.alloc_vertex_tree_array<uint64_t>();
    auto core_query = graph.alloc_vertex_tree_array<uint64_t>();
    auto core_query_check = graph.alloc_vertex_tree_array<uint64_t>();
    auto check = graph.alloc_vertex_tree_array<uint64_t>();
    const uint64_t MAXL = 65536;
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
        return std::make_pair(src_data/adjedge.data > dst_data, src_data / adjedge.data);
    };
    auto active_result_func = [](uint64_t old_result, uint64_t src, uint64_t dst, uint64_t src_data, uint64_t old_dst_data, uint64_t new_dst_data) -> uint64_t
    {
        return old_result+1;
    };
    auto equal_func = [](uint64_t src, uint64_t dst, uint64_t src_data, uint64_t dst_data, decltype(graph)::adjedge_type adjedge) -> bool
    {
        return src_data / adjedge.data == dst_data;
    };
    auto init_label_func = [=](uint64_t vid) -> std::pair<uint64_t, bool>
    {
        return {vid==root?MAXL:0, vid==root};
    };
    uint64_t root_value = MAXL;
    uint64_t other_value = 0;
    fprintf(stderr,"%ld small graph edges, %ld not being added\n", graph.count_edges(), raw_edges_len-graph.count_edges());

    std::atomic_uint64_t add_edge_len(0), del_edge_len(0), re_edge_len(0), sample_edge_len(0);
    std::vector<decltype(graph)::edge_type> add_edge(compute_batch*batch);
    std::vector<decltype(graph)::edge_type> del_edge(compute_batch*batch);
    // std::vector<decltype(graph)::edge_type> resample_edge(compute_batch*batch);
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
        //sample core graph
        uint64_t wrong = 0;
        int tqdm_ = 0;
        for (uint64_t src = 0; src < num_vertices; src++)
        {
            uint64_t degreeV = graph.get_outgoing_degree(src);
            uint64_t degreeW = graph.get_incoming_degree(src);
            if (src == tqdm[tqdm_])
            {
                tqdm_++;
                fprintf(stderr,"Now is sample for %d 10 percantage of nodes\n",tqdm_);
            }
            // fprintf(stderr,"%ld\n",src);
            // if (src==1653)
            // {
            //     fprintf(stderr,"%ld, %ld",degreeV, degreeW);
            // }
            
            for (uint64_t j = 0; j < degreeV; j++)
            {
                uint64_t dst = graph.get_dst_number(src, j);
                uint64_t edge_len = (src+dst)%32+1;
                // if (src==1653)
                // {
                //     fprintf(stderr,"(%ld, %ld)\n",src, dst);
                // }
                
                if (label_init[src].data == other_value)
                {
                    // query_specific_core_graph.insert(std::make_pair(src,dst));
                    Q_Core.push_back(std::make_pair(src, dst));
                } 
                if (label_init[src].data + edge_len == label_init[dst].data)
                {

                    // query_specific_core_graph.insert(std::make_pair(src,dst));
                    Q_Core.push_back(std::make_pair(src, dst));


                }
            }            

            if (label_init[src].data == other_value)
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
        fprintf(stderr,"we have %ld edges sampled from deleted edges\n",wrong);
        fprintf(stderr,"out Going edges finished sampling\n");
        // fprintf(stderr, "we have %ld edges in query specific core graph, and we have %ld edges in whole graph\n", query_specific_core_graph.size(), raw_edges_len);
        fprintf(stderr, "we have %ld edges in query specific core graph, and we have %ld edges in whole graph\n", Q_Core.size(), raw_edges_len);        
        // for (auto e: query_specific_core_graph)
        // {
        //     core_graph.add_edge({e.first, e.second, (e.first+e.second)%32+1}, true);
        // }
        for (auto e: Q_Core)
        {
            core_graph.add_edge({e.first, e.second, (e.first+e.second)%32+1}, true);
        }             
    // {
    //     auto core_s_start = std::chrono::high_resolution_clock::now();
    //     // #pragma omp parallel for
    //     for(uint64_t i=0;i<core_len;i++)
    //     {
    //         const auto &e = sampledd_edge[i];
    //         core_graph.add_edge({e.src, e.dst, (e.src + e.dst)%32+1}, true);
    //     }
    //     auto core_s_end = std::chrono::high_resolution_clock::now();
    //     fprintf(stderr, "add: %.6lfms\n", 1e-3*(uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(core_s_end- core_s_start).count());
    // }        
        
        // query_specific_core_graph.clear();
        // common_graph.build_tree<uint64_t, uint64_t>(
        //     init_label_func,
        //     continue_reduce_print_func,
        //     update_func,
        //     active_result_func,
        //     check
        // );
        
        core_graph.build_tree<uint64_t, uint64_t>(
            init_label_func,
            continue_reduce_print_func,
            update_func,
            active_result_func,
            core_query
        );
        for (uint64_t i = 0; i < num_vertices; i++)
        {
            if (label_init[i].data != core_query[i].data)
            {
                count++;
                // fprintf(stderr,"Results correct is %ld, and wrong result is %ld\n", label_init[i].data, core_query[i].data);
                // fprintf(stderr,"The wrong node has %ld in-edges and %ld out-edges\n", core_graph.get_incoming_degree(i), core_graph.get_outgoing_degree(i));
            }
        }
        if (count != 0)
        {
            fprintf(stderr,"%ld results not correct\n",count);    
        }
        else{fprintf(stderr,"all results are correct for SMALL graph and its core graph\n");}
        // added edges and count time
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
                label_init, add_edge, length_add.load(), true
            );
        auto add_compute_end = std::chrono::high_resolution_clock::now();
        fprintf(stderr, "add compute: %.6lfms\n", 1e-3*(uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(add_compute_end-add_compute_start).count());        
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
        // count = 0;
        // for (uint64_t k= 0; k < num_vertices; k++)
        // {
        //     if (std::abs(label_init[k].data - core_query[k].data > 32))
        //     {
        //         count++;
        //         // fprintf(stderr,"Result are %ld and %ld\n",label_init[k].data, core_query[k].data);                
        //     }
        // }
        // fprintf(stderr,"This %ld numbers of nodes results NOT correct\n",count);
        // for (size_t i = 0; i < 10; i++)
        // {
        //     fprintf(stderr,"Result are %ld and %ld\n",label_init[i].data, core_query[i].data);
        // }
        
        
        //sample for addition-after core graph
        // uint64_t not_sample_count = 0;
        // for (uint64_t src = 0; src < num_vertices; src++)
        // {
        //     uint64_t degreeV = graph.get_outgoing_degree(src);
        //     for (uint64_t j = 0; j < degreeV; j++)
        //     {
        //         uint64_t dst = graph.get_dst_number(src, j);
        //         uint64_t edge_len = (src+dst)%32+1;
        //         if (label_add[src].data + edge_len == label_add[dst].data)
        //         {
        //             if (core_graph.check_edge({src,dst,edge_len}) == 0)
        //             {
        //                 // fprintf(stderr,"This edge is not sampled %ld, %ld\n",src, dst);
        //                 // query_specific_core_graph.insert(std::make_pair(src,dst));
        //                 resample_edge[re_edge_len.fetch_add(1)] = {src, dst, ((src+dst)%32+1)};
        //                 auto old_num = core_graph.add_edge({src, dst, (src+dst)%32+1}, true);                        
        //                 not_sample_count++;
        //             }
        //         }
        //     }
        // }
        // // if (std::find(modified_edges.begin(),modified_edges.end(),std::make_pair(add_edge[1].src,add_edge[1].dst))!=modified_edges.end())
        // // {
        // //     fprintf(stderr,"check pass!\n");
        // // }
        // count = 0;
        // uint64_t src_unreach = 0;
        // uint64_t dst_unreach = 0;
        // uint64_t at_least_one_unreach = 0;
        // uint64_t both_reachable = 0;
        // uint64_t not_include_in_previous_dependent_tree = 0;
        // uint64_t should_include_in_previous_dependent_tree = 0;
        
        // for (uint64_t re = 0; re < resample_edge.size(); re++)
        // {
        //     uint64_t edge_len = (resample_edge[re].src + resample_edge[re].dst)%32+1;
        //    if (std::find(modified_edges.begin(),modified_edges.end(),std::make_pair(resample_edge[re].src,resample_edge[re].dst))==modified_edges.end())
        //    {
        //     if (common_graph.check_edge({resample_edge[re].src,resample_edge[re].dst,edge_len}) != 0) // edges in common graph but not in modified batch
        //     {
        //         count++;
        //         if (check[resample_edge[re].src].data == MAXL)
        //         {
        //             src_unreach++;
        //             fprintf(stderr,"This nodes src %ld is not reachable in common graph\n",resample_edge[re].src);
        //         }
        //         if (check[resample_edge[re].dst].data == MAXL)
        //         {
        //             dst_unreach++;
        //             fprintf(stderr,"This nodes dst %ld is not reachable in common graph\n",resample_edge[re].dst);
        //         }
        //         if ((check[resample_edge[re].src].data == MAXL) || (check[resample_edge[re].dst].data == MAXL))
        //         {
        //             at_least_one_unreach++;
        //         }
        //         if ((check[resample_edge[re].src].data != MAXL) && (check[resample_edge[re].dst].data != MAXL))
        //         {
        //             both_reachable++;
        //         }
        //         if (label_add[resample_edge[re].src].data + edge_len == label_add[resample_edge[re].dst].data)
        //         {
        //             should_include_in_previous_dependent_tree++;
        //         }
                
                
        //     }
        //    }
        // }
        // fprintf(stderr,"%ld src unreach, and %ld dst unreach\n",src_unreach,dst_unreach);
        // fprintf(stderr,"%ld edges at least one ends not included\n",at_least_one_unreach);
        // fprintf(stderr,"%ld edges both reachable not included\n",both_reachable);
        // fprintf(stderr,"%ld edges should included\n",should_include_in_previous_dependent_tree);
        // fprintf(stderr,"%ld edges not in modified batch but in common graph\n",count);
        // fprintf(stderr,"%ld edges not sampled\n",not_sample_count);

        // auto re_core_add_compute_start = std::chrono::high_resolution_clock::now();
        //     core_graph.update_tree_add<uint64_t, uint64_t>(
        //         continue_reduce_func,
        //         update_func,
        //         active_result_func,
        //         core_query, resample_edge, re_edge_len.load(), true
        //     );
        // auto re_core_add_compute_end = std::chrono::high_resolution_clock::now();
        // fprintf(stderr, "Re-sample core add compute: %.6lfms\n", 1e-3*(uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(re_core_add_compute_end - re_core_add_compute_start).count());
        // core_graph.build_tree<uint64_t, uint64_t>(
        //     init_label_func,
        //     continue_reduce_print_func,
        //     update_func,
        //     active_result_func,
        //     core_query_check
        // );        
        // count = 0;
        // for (uint64_t i = 0; i < num_vertices; i++)
        // {
        // if (core_query_check[i].data != label_add[i].data)
        // {
        //     count++;
        //     // fprintf(stderr,"Results correct is %ld, and wrong result is %ld\n", label_add[i].data, core_query[i].data);
        //     // fprintf(stderr,"The wrong node has %ld in-edges and %ld out-edges\n", core_graph.get_incoming_degree(i), core_graph.get_outgoing_degree(i));
        // }            
        // }
        // if (count==0)
        // {
        //     fprintf(stderr,"Core added all correct\n");
        // }else{
        //     fprintf(stderr,"%ld edges Not correct\n",count);
        // }        
        // count = 0;
        // for (uint64_t i = 0; i < num_vertices; i++)
        // {
        // if (core_query[i].data != core_query_check[i].data)
        // {
        //     count++;
        //     fprintf(stderr,"Results correct is %ld, and wrong result is %ld\n", core_query_check[i].data, core_query[i].data);

        // }            
        // }
        // if (count==0)
        // {
        //     fprintf(stderr,"Core check all correct\n");
        // }else{
            // fprintf(stderr,"%ld results Not correct\n",count);
        // }
        query_specific_core_graph.clear();

    }
    return 0;
}
