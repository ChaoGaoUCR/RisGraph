apps=("build/index-sssp")

# 定义 graph 和 source 文件组合
graph_sources=(
    # "/home/cgao037/graph/sx/stack_snap source/sx.txt"
    # "/home/cgao037/graph/wiki/snap_wiki source/wiki_src.txt"
    # "/home/cgao037/graph/or/snap_or source/or.txt"
    # "/home/cgao037/graph/wen/snap_wen source/wen.txt"
    "/home/cgao037/graph/dl/snap_dl source/dl_src.txt"    
    # "/home/cgao037/graph/ttw/ttw_snap source/TW_queries.txt"
)

# 定义 batch_num 和 batch_size 组合
batch_params=(
    "1 0.04"
    "1 0.08"
    "1 0.16"
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

apps=("build/index-sswp")

# 定义 graph 和 source 文件组合
graph_sources=(
    # "/home/cgao037/graph/sx/stack_snap source/sx.txt"
    # "/home/cgao037/graph/wiki/snap_wiki source/wiki_src.txt"
    "/home/cgao037/graph/or/snap_or source/or.txt"
    # "/home/cgao037/graph/wen/snap_wen source/wen.txt"
    # "/home/cgao037/graph/dl/snap_dl source/dl_src.txt"    
    # "/home/cgao037/graph/ttw/ttw_snap source/TW_queries.txt"
)

# 定义 batch_num 和 batch_size 组合
batch_params=(
    "1 0.04"
    "1 0.08"
    "1 0.16"
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