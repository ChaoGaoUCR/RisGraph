import re
import pandas as pd
import argparse

def parse_log(log_path):
    with open(log_path, 'r') as f:
        log_data = f.read()

    # 正则匹配
    pattern = re.compile(
        r'Running: \.\/build\/(.*?) \/home\/cgao037\/graph\/(.*?)\/.*? (\d+) ([\d\.]+).*?'
        r'Core Graph Size is ([\d\.]+) % For Snapshot.*?Snapshot has ([\d\.]+)% correct result',
        re.S)

    results = []

    for match in pattern.finditer(log_data):
        app = match.group(1)
        graph = match.group(2)
        batch_num = int(match.group(3))
        batch_size = float(match.group(4))
        total_batch_size = batch_num * batch_size
        core_graph_size = float(match.group(5))
        accuracy = float(match.group(6))

        results.append({
            "App": app,
            "Graph": graph,
            "Total Batch Size": total_batch_size,
            "Core Graph Size (%)": core_graph_size,
            "Accuracy (%)": accuracy
        })

    df = pd.DataFrame(results)
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract and aggregate metrics from log file")
    parser.add_argument('log_file', type=str, help='Path to the log file')
    args = parser.parse_args()

    # 解析 log
    df = parse_log(args.log_file)

    # groupby App + Graph + Total Batch Size 后取均值
    grouped_df = df.groupby(['App', 'Graph', 'Total Batch Size']).mean(numeric_only=True).reset_index()

    # 导出
    output_csv = args.log_file.replace('.txt', '_metrics_avg.csv')
    grouped_df.to_csv(output_csv, index=False)

    print(f"Aggregated results saved to {output_csv}")
    print(grouped_df)
