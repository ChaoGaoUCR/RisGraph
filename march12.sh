apps=(
    "build/BFSbaseline" "build/SSNPbaseline" "build/SSSPbaseline"
    "build/CCbaseline" "build/SSRbaseline" "build/SSWPbaseline"
    "build/Viterbibaseline"
)

graph_sources=(
    # "/home/xyin014/graph/sx/stack_snap source/sx.txt"
    # "/home/xyin014/graph/wiki/snap_wiki source/wiki_src.txt"
    "/home/xyin014/graph/or/snap_or source/or.txt"
    "/home/xyin014/graph/wen/snap_wen source/wen.txt"
    "/home/xyin014/graph/dl/snap_dl source/dl_src.txt"
    "/home/xyin014/graph/ttw/ttw_snap source/TW_queries.txt"
)

# 定义 batch_num 和 batch_size 组合
batch_params=(
    "8 0.02"  "16 0.01"  "32 0.005"  "64 0.0025"
)

# 遍历所有应用程序、graph/source 组合 和 batch 组合
for app in "${apps[@]}"; do
    for gs in "${graph_sources[@]}"; do
        read graphfile root_file <<< "$gs"
        for params in "${batch_params[@]}"; do
            read batch_num batch_size <<< "$params"
            for extra_param in {0..15}; do
                echo "Running: $app $graphfile $root_file $batch_num $batch_size $extra_param"
                $app "$graphfile" "$root_file" "$batch_num" "$batch_size" "$extra_param"
            done
        done
    done
done
