#!/bin/bash

# 程序列表
PROGRAMS=(
    "build/test-index-bfs"
    "build/test-index-ssnp"
    "build/test-index-ssr"
    "build/test-index-sssp"
    "build/test-index-sswp"
    "build/test-index-viterbi"
    "build/test-index-wcc"
)

# batch_ratio 列表
RATIOS=(0.08 0.16 0.24 0.32 0.40 0.48)

# 图和源文件列表
TASKS=(
    "/home/cgao037/graph/sx/stack_snap source/sx.txt sx_index.txt"
    "/home/cgao037/graph/wiki/snap_wiki source/wiki_src.txt wiki_index.txt"
    "/home/cgao037/graph/or/snap_or source/or.txt or_index.txt"
    "/home/cgao037/graph/wen/snap_wen source/wen.txt wen_index.txt"
    "/home/cgao037/graph/dl/snap_dl source/dl_src.txt dl_index.txt"
    "/home/cgao037/graph/ttw/ttw_snap source/TW_queries.txt ttw_index.txt"
)

# 遍历每个任务
for task in "${TASKS[@]}"; do
    set -- $task
    GRAPH=$1
    SOURCE=$2
    OUTPUT_LOG=$3

    # 清空输出日志
    > "$OUTPUT_LOG"

    for prog in "${PROGRAMS[@]}"; do
        for ratio in "${RATIOS[@]}"; do
            echo "Running $prog with batch_ratio=$ratio" | tee -a "$OUTPUT_LOG"
            "$prog" "$GRAPH" "$SOURCE" 1 "$ratio" 2>&1 | tee -a "$OUTPUT_LOG"
            echo "" >> "$OUTPUT_LOG"
        done
        echo "-----------------------------------" >> "$OUTPUT_LOG"
        echo "" >> "$OUTPUT_LOG"
        sleep 1
    done

    echo "All runs finished for $OUTPUT_LOG."
done

echo "All tasks finished."
