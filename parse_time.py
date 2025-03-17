import re
import sys
import collections

def parse_and_summarize(file_path):
    # 使用嵌套字典存储 application -> (batch_num, batch_size) -> 时间值
    results = collections.defaultdict(lambda: collections.defaultdict(lambda: {'streaming': 0.0, 'common_graph': 0.0}))

    current_app = None  # 记录当前的应用程序
    current_batch = None  # 记录当前的 batch_num 和 batch_size

    with open(file_path, 'r') as file:
        lines = file.readlines()

    for line in lines:
        # 解析 Running 这一行，找到应用程序名、batch_num 和 batch_size
        match_run = re.search(r"Running: (\S+) (\S+) (\S+) (\d+) ([\d\.]+) \d+$", line)
        if match_run:
            current_app = match_run.group(1)  # 提取应用程序名称
            batch_num = match_run.group(4)  # 提取 batch_num
            batch_size = match_run.group(5)  # 提取 batch_size
            current_batch = (batch_num, batch_size)  # 记录当前的 batch 组合
            continue

        # 提取 streaming total time
        match_streaming = re.search(r"streaming total time ([\d\.]+)s", line)
        if match_streaming and current_app and current_batch:
            results[current_app][current_batch]['streaming'] += float(match_streaming.group(1))

        # 提取 CommonGraph total time
        match_common_graph = re.search(r"CommonGraph total time ([\d\.]+)s", line)
        if match_common_graph and current_app and current_batch:
            results[current_app][current_batch]['common_graph'] += float(match_common_graph.group(1))

    # 打印结果
    print("Summed Times for Each Application and Batch Parameter:")
    for app, batches in sorted(results.items()):
        print(f"\nApplication: {app}")
        for (batch_num, batch_size), times in sorted(batches.items()):
            print(f"  Batch {batch_num}, Size {batch_size}: Streaming Total = {times['streaming']:.6f}s, CommonGraph Total = {times['common_graph']:.6f}s")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python summarize_app_batches.py <output_file>")
    else:
        parse_and_summarize(sys.argv[1])
