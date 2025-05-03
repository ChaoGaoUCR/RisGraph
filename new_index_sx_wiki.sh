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
RATIOS=(0.08 0.16 0.24 0.32 0.48)

# 分别处理每个图

# 处理 stack 图
GRAPH="/home/cgao037/graph/sx/stack_snap"
SOURCE="source/sx.txt"
OUTPUT_LOG="sx_index.txt"

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

# 处理 wiki 图
GRAPH="/home/cgao037/graph/wiki/snap_wiki"
SOURCE="source/wiki_src.txt"
OUTPUT_LOG="wiki_index.txt"

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

echo "All tasks finished."
