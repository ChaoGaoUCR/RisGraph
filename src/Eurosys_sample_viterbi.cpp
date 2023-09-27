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
#include <iostream>
#include <fstream> 

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
    uint64_t compute_batch = std::stoull(argv[5]);

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
        fprintf(stderr, "read io time: %.6lfs\n", 1e-6*(uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(end-start).count());
        // fprintf(stderr, "|E|=%lu\n", raw_edges_len);
    }    
    Graph<uint64_t> common_graph(num_vertices, raw_edges_len, false, true);
    Graph<uint64_t> graph(num_vertices, raw_edges_len, false, true);
    Graph<uint64_t> core_graph(num_vertices, raw_edges_len, false, true);
    Graph<uint64_t> snapshot_graph(num_vertices, raw_edges_len, false, true);
    // full graph read all the edges, common graph read 100*batch size reduced graph
    // snapshots read addition 50 batches

    uint64_t big_batch = batch * compute_batch;
    uint64_t seed = std::rand()%(raw_edges_len - 2*big_batch);
    std::vector<decltype(graph)::edge_type> add_edge(50*batch);
    std::vector<decltype(graph)::edge_type> del_edge(50*batch);
    std::atomic_uint64_t length_del(0);
    std::atomic_uint64_t length_add(0);        
    {
        // 100 batch randomization init
        for (uint64_t i = seed; i < seed+big_batch; i++)
        {
            auto &e = raw_edges[i];
            del_edge[length_del.fetch_add(1)] = {e.first, e.second, (e.first+e.second)%32+1};
            auto &e1 = raw_edges[i+big_batch];
            add_edge[length_add.fetch_add(1)] = {e1.first, e1.second, (e1.first+e1.second)%32+1};
        }
    }
    // fprintf(stderr,"sampling edges generated finished sample seed is %ld\n", seed);
    {
        // full graph read
        #pragma omp parallel for
        for (uint64_t i = 0; i < raw_edges_len; i++)
        {
            // const auto &e = raw_edges[i];
            graph.add_edge({raw_edges[i].first, raw_edges[i].second, (raw_edges[i].first+raw_edges[i].second)%32+1}, true);
        }
        // common graph read
        #pragma omp parallel for
        for (uint64_t i = 0; i < seed; i++)
        {
            // const auto &e = raw_edges[i];
            common_graph.add_edge({raw_edges[i].first, raw_edges[i].second, (raw_edges[i].first+raw_edges[i].second)%32+1}, true);
        }
        #pragma omp parallel for
        for (uint64_t i = seed+2*big_batch; i < raw_edges_len; i++)
        {
            // const auto &e = raw_edges[i];
            common_graph.add_edge({raw_edges[i].first, raw_edges[i].second, (raw_edges[i].first+raw_edges[i].second)%32+1}, true);
        }
        //snapshots add
        #pragma omp parallel for
        for (uint64_t i = 0; i < seed; i++)
        {
            // const auto &e = raw_edges[i];
            snapshot_graph.add_edge({raw_edges[i].first, raw_edges[i].second, (raw_edges[i].first+raw_edges[i].second)%32+1}, true);
        }
        #pragma omp parallel for
        for (uint64_t i = seed+big_batch; i < raw_edges_len; i++)
        {
            // const auto &e = raw_edges[i];
            snapshot_graph.add_edge({raw_edges[i].first, raw_edges[i].second, (raw_edges[i].first+raw_edges[i].second)%32+1}, true);
        }
        
        fprintf(stderr,"full graph has %ld edges, common graph has %ld edges, First snapshot has %ld edges\n", graph.count_edges(), common_graph.count_edges(), snapshot_graph.count_edges());
    }
    // Graph Function
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

    // graph value array set-up
    auto label_full = graph.alloc_vertex_tree_array<uint64_t>();
    auto label_full_check = graph.alloc_vertex_tree_array<uint64_t>();
    auto label_common = graph.alloc_vertex_tree_array<uint64_t>();
    auto label_common_ = graph.alloc_vertex_tree_array<uint64_t>();
    auto snapshot_label = graph.alloc_vertex_tree_array<uint64_t>();
    auto core_label = graph.alloc_vertex_tree_array<uint64_t>();

    // check full graph results
    graph.build_tree<uint64_t, uint64_t>(
    init_label_func,
    continue_reduce_print_func,
    update_func,
    active_result_func,
    label_full
    );
    graph.build_tree<uint64_t, uint64_t>(
    init_label_func,
    continue_reduce_print_func,
    update_func,
    active_result_func,
    label_full_check
    );    
    // check common graph results
    common_graph.build_tree<uint64_t, uint64_t>(
        init_label_func,
        continue_reduce_print_func,
        update_func,
        active_result_func,
        label_common
        );    
    // check snapshots results
    snapshot_graph.build_tree<uint64_t, uint64_t>(
        init_label_func,
        continue_reduce_print_func,
        update_func,
        active_result_func,
        snapshot_label
        );
    // now we sample core graph from common graph
    // we use a set to record all the edges
    fprintf(stderr,"start sampling\n");
    // std::set<std::pair<uint64_t, uint64_t>> query_specific_core_graph;
    auto sample_begin = std::chrono::high_resolution_clock::now();
    THRESHOLD_OPENMP_LOCAL("omp parallel for", seed, 1024,
        for (uint64_t edge_ptr = 0; edge_ptr < seed; edge_ptr++)
        {
            uint64_t src = raw_edges[edge_ptr].first;
            uint64_t dst = raw_edges[edge_ptr].second;
            uint64_t len = (src + dst)%32 +1;
            if ((label_common[src].data != label_full[src].data) || (label_common[src].data / len == label_common[dst].data))
            {
                core_graph.add_edge({src, dst, len}, true);
            }
        }
    );
    THRESHOLD_OPENMP_LOCAL("omp parallel for", (raw_edges_len - seed+2*big_batch), 1024,
        for (uint64_t edge_ptr = seed+2*big_batch; edge_ptr < raw_edges_len; edge_ptr++)
        {
            uint64_t src = raw_edges[edge_ptr].first;
            uint64_t dst = raw_edges[edge_ptr].second;
            uint64_t len = (src + dst)%32 +1;
            if ((label_common[src].data != label_full[src].data) || (label_common[src].data / len == label_common[dst].data))
            {
                core_graph.add_edge({src, dst, len}, true);
            }
        }
    );
    auto sample_end = std::chrono::high_resolution_clock::now();
    fprintf(stderr, "Sample Core Graph time: %.6lfms\n", 1e-3*(uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(sample_end-sample_begin).count());
    fprintf(stderr,"%ld edges in the core graph\n",core_graph.count_edges());
    // core graph correctness check
    core_graph.build_tree<uint64_t, uint64_t>(
        init_label_func,
        continue_reduce_print_func,
        update_func,
        active_result_func,
        core_label
        );
    uint64_t count_core_check = 0;
    for (uint64_t i = 0; i < num_vertices; i++)
    {
        if (core_label[i].data != label_common[i].data)
        {
            count_core_check ++;
        }
        
    }
    if (count_core_check != 0){
    fprintf(stderr,"%ld numbers of nodes not correct in sampled core graph!\n", count_core_check);}
    else{fprintf(stderr,"All correct in sampled core Graph!\n");}
    // add big add batch to common graph change it to snapshot
    #pragma omp parallel for
    for (uint64_t i = 0; i < big_batch; i++)
    {
        common_graph.add_edge({add_edge[i].src, add_edge[i].dst, add_edge[i].data}, true);
        core_graph.add_edge({add_edge[i].src, add_edge[i].dst, add_edge[i].data}, true);

    }
        // common graph write to file
        std::string argv1 = argv[1];
        std::string core_output = "common_sssp_" + argv1;
        fprintf(stderr,"%s sample begin\n", core_output.c_str());
        std::ofstream file(core_output);
        uint64_t sample_count = 0;
        for (uint64_t i = 0; i < num_vertices; i++)
        {
            uint64_t out_ptr = core_graph.get_outgoing_degree(i);
            for (uint64_t j = 0; j < out_ptr; j++)
            {
                uint64_t dst = core_graph.get_outgoing_adjlist(i)[j].nbr;
                int weight = (i+dst) % 32 +1;
                sample_count++;
                file << i << "\t" << dst << "\t" << weight << std::endl;                
            }
        }
        file.close();
        fprintf(stderr,"%s sample finished with %ld egdes we have %ld edges in core graph \n", core_output.c_str(), sample_count, core_graph.count_edges());
    // fprintf(stderr,"%ld edges in common graph now\n",common_graph.count_edges());
    common_graph.build_tree<uint64_t, uint64_t>(
        init_label_func,
        continue_reduce_print_func,
        update_func,
        active_result_func,
        label_common_
        );
    auto common_add_compute_start = std::chrono::high_resolution_clock::now();
    common_graph.update_tree_add<uint64_t, uint64_t>(
                    continue_reduce_func,
                    update_func,
                    active_result_func,
                    label_common, add_edge, length_add.load(), true
                );
    auto common_add_compute_end = std::chrono::high_resolution_clock::now();
    // fprintf(stderr, "common add compute: %.6lfms\n", 1e-3*(uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(common_add_compute_end-common_add_compute_start).count());
    double common_add_time = 1e-3*(uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(common_add_compute_end-common_add_compute_start).count();
    uint64_t check_common_snapshot = 0;
    for (uint64_t i = 0; i < num_vertices; i++)
    {
        if (label_common_[i].data != snapshot_label[i].data)
        {
            check_common_snapshot++;
            // fprintf(stderr,"%ld %ld\n",label_common_[i].data, snapshot_label[i].data);
        }
    }
    if (check_common_snapshot != 0){
    fprintf(stderr,"%ld numbers of nodes not correct in common graph!\n", check_common_snapshot);}
    else{fprintf(stderr,"All correct in common Graph!\n");}
    auto core_add_compute_start = std::chrono::high_resolution_clock::now();
    core_graph.update_tree_add<uint64_t, uint64_t>(
        continue_reduce_func,
        update_func,
        active_result_func,
        core_label, add_edge, length_add.load(), true
    );    
    auto core_add_compute_end = std::chrono::high_resolution_clock::now();
    // fprintf(stderr, "core add compute time: %.6lfms\n", 1e-3*(uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(core_add_compute_end-core_add_compute_start).count());
    double core_add_time = 1e-3*(uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(core_add_compute_end-core_add_compute_start).count();
    
    uint64_t check_core_snapshot = 0;
    for (size_t i = 0; i < num_vertices; i++)
    {
        if (core_label[i].data != snapshot_label[i].data)
        {
            check_core_snapshot ++;
        }
    }

    if (check_core_snapshot != 0)
    {
        fprintf(stderr,"%ld numbers of nodes not correct in core graph!\n", check_core_snapshot);
    }
    else
    {
        fprintf(stderr,"All correct in core Graph Again!!\n");
    }
    #pragma omp parallel for
    for (uint64_t i = 0; i < big_batch; i++)
    {
        core_graph.del_edge({add_edge[i].src, add_edge[i].dst, add_edge[i].data}, true);
    }    
    core_graph.update_tree_del<uint64_t, uint64_t>(
        init_label_func,
        continue_reduce_func,
        update_func,
        active_result_func,
        equal_func,
        core_label, add_edge, length_add.load(), true
    );        
    #pragma omp parallel for
    for (uint64_t i = 0; i < big_batch; i++)
    {
        core_graph.del_edge({add_edge[i].src, add_edge[i].dst, add_edge[i].data}, true);

    }
    core_graph.update_tree_del<uint64_t, uint64_t>(
            init_label_func,
            continue_reduce_func,
            update_func,
            active_result_func,
            equal_func,
            core_label, add_edge, length_add.load(), true
        );

    //  now write batch file graph at first
    // snapshot i has (n-1) batch n is number of snapshots
    //  i - n-1 deletion batch 0 - i addition batch
    for (int s = 0; s < compute_batch+1; s++)
    {
        // seed - seed+ big_batch-1 --> deletion
        // seed+big_batch seed+2*big_batch --> addition
        std::string argv1 = argv[1];
        std::string batch_name = argv1 + "_sssp_" + std::to_string(s) + ".txt";
        std::ofstream file(batch_name);
        uint64_t del_start = seed + s * batch;
        uint64_t del_end = seed + compute_batch*batch;
        uint64_t add_start = seed + big_batch;
        uint64_t add_end = seed + big_batch + s*batch;
        for (uint64_t i = del_start; i < del_end; i++)
        {
            uint64_t weight = (raw_edges[i].first + raw_edges[i].second) % 32+1;
            file << raw_edges[i].first << "\t" << raw_edges[i].second << "\t" << weight << std::endl;
        }
        for (uint64_t i = add_start; i < add_end; i++)
        {
            uint64_t weight = (raw_edges[i].first + raw_edges[i].second) % 32+1;
            file << raw_edges[i].first << "\t" << raw_edges[i].second << "\t" << weight << std::endl;
        }
        file.close();        
    }
    return 0;
}