#!/bin/bash

# 定义应用程序列表
apps=("build/bfs-core" "build/ssr-core" "build/sswp-core" "build/wcc-core" "build/ssnp-core" "build/sssp-core" "build/viterbi-core")

# 定义 graph 和 source 文件组合
graph_sources=(
    "/home/cgao037/graph/sx/stack_snap source/sx.txt"
    "/home/cgao037/graph/wiki/snap_wiki source/wiki_src.txt"
    "/home/cgao037/graph/or/snap_or source/or.txt"
    "/home/cgao037/graph/wen/snap_wen source/wen.txt"
    "/home/xyin014/graph/dl/snap_dl source/dl_src.txt"    
    "/home/xyin014/graph/ttw/ttw_snap source/TW_queries.txt"
)

# 定义 batch_num 和 batch_size 组合
batch_params=(
    "8 0.01"  "16 0.005" "32 0.0025" "64 0.00125"
    "8 0.02"  "16 0.01"  "32 0.005"  "64 0.0025"
    "8 0.04"  "16 0.02"  "32 0.01"   "64 0.005"
)

# 遍历所有应用程序、graph/source 组合 和 batch 组合
for app in "${apps[@]}"; do
    for gs in "${graph_sources[@]}"; do
        read graphfile root_file <<< "$gs"
        for params in "${batch_params[@]}"; do
            read batch_num batch_size <<< "$params"
            echo "Running: $app $graphfile $root_file $batch_num $batch_size"
            $app "$graphfile" "$root_file" "$batch_num" "$batch_size"
        done
    done
done
