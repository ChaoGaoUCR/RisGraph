#!/bin/bash

# 设置参数
GRAPH="/home/cgao037/graph/or/snap_or"
SOURCE="source/or.txt"
OUTPUT_LOG="or_index.txt"

# 清空输出日志
> "$OUTPUT_LOG"

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

# 遍历程序和比例
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

echo "All runs finished. Logs saved to $OUTPUT_LOG."
