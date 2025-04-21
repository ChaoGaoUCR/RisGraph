#!/bin/bash

# HotNode Scripts
# Overhead Scripts
# Union Core Comparison Scripts

echo "HotNode Overhead Scripts"

# 定义应用程序列表
apps=("build/HighDegreeNode")

# 定义 graph 和 source 文件组合
graph_sources=(
    # "/home/cgao037/graph/sx/stack_snap source/sx.txt"
    # "/home/cgao037/graph/wiki/snap_wiki source/wiki_src.txt"
    # "/home/cgao037/graph/or/snap_or source/or.txt"
    # "/home/cgao037/graph/wen/snap_wen source/wen.txt"
    # "/home/cgao037/graph/dl/snap_dl source/dl_src.txt"    
    "/home/cgao037/graph/ttw/ttw_snap source/TW_queries.txt"
)

# 定义 batch_num 和 batch_size 组合
batch_params=(
    "1 0.24"
)

# 定义 nodenum 和 flag
nodenums=(1 2 4 8 16)
flags=(0 1)

# 遍历所有应用程序、graph/source 组合、batch 组合、nodenum 和 flag
for app in "${apps[@]}"; do
    for gs in "${graph_sources[@]}"; do
        read graphfile root_file <<< "$gs"
        for params in "${batch_params[@]}"; do
            read batch_num batch_size <<< "$params"
            for nodenum in "${nodenums[@]}"; do
                for flag in "${flags[@]}"; do
                    echo "Running: $app $graphfile $root_file $batch_num $batch_size $nodenum $flag"
                    $app "$graphfile" "$root_file" "$batch_num" "$batch_size" "$nodenum" "$flag"
                done
            done
        done
    done
done

#!/bin/bash
echo "Overhead Scripts"
# 定义应用程序列表
apps=("build/Ovearhead")

# 定义 graph 和 source 文件组合
graph_sources=(
    "/home/cgao037/graph/sx/stack_snap source/sx.txt"
    "/home/cgao037/graph/wiki/snap_wiki source/wiki_src.txt"
    "/home/cgao037/graph/or/snap_or source/or.txt"
    "/home/cgao037/graph/wen/snap_wen source/wen.txt"
    "/home/cgao037/graph/dl/snap_dl source/dl_src.txt"    
    "/home/cgao037/graph/ttw/ttw_snap source/TW_queries.txt"
)

# 定义 batch_num 和 batch_size 组合
batch_params=(
    "1 0.24"
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

echo "Union Intersection Scripts"
#!/bin/bash

# 定义应用程序列表
apps=("build/UnionInterSection")

# 定义 graph 和 source 文件组合
graph_sources=(
    # "/home/cgao037/graph/sx/stack_snap source/sx.txt"
    # "/home/cgao037/graph/wiki/snap_wiki source/wiki_src.txt"
    # "/home/cgao037/graph/or/snap_or source/or.txt"
    # "/home/cgao037/graph/wen/snap_wen source/wen.txt"
    # "/home/cgao037/graph/dl/snap_dl source/dl_src.txt"    
    "/home/cgao037/graph/ttw/ttw_snap source/TW_queries.txt"
)

# 定义 batch_num 和 batch_size 组合
batch_params=(
    "1 0.04"
    "1 0.08"
    "1 0.12"
    "1 0.16"
    "1 0.20"
    "1 0.24"
    "1 0.28"
    "1 0.32"
    "1 0.36"
    "1 0.40"
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

