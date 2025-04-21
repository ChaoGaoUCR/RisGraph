#!/bin/bash

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Graph file path
GRAPH_FILE="/home/cgao037/graph/ttw/ttw_snap"
# Query file path
QUERY_FILE="/home/cgao037/RisGraph/source/TW_queries.txt"
# Number of batches
BATCH_NUM=1
# Batch sizes to test
BATCH_SIZES=(0.08 0.16 0.32 0.48)

# Array of programs to run
PROGRAMS=(
    "motivation_figure4"
    "motivation_figure4_sswp"
    "motivation_figure4_bfs"
    "motivation_figure4_cc"
    "motivation_figure4_ssnp"
)

# Create output directory if it doesn't exist
mkdir -p results

# Function to run a program with given parameters
run_program() {
    local program=$1
    local batch_size=$2
    echo "Running $program with batch size $batch_size..."
    
    # Run the program and save output to a file
    "$SCRIPT_DIR/build/$program" "$GRAPH_FILE" "$QUERY_FILE" "$BATCH_NUM" "$batch_size" > "results/${program}_batch${batch_size}.txt" 2>&1
    
    # Check if the program ran successfully
    if [ $? -eq 0 ]; then
        echo "✓ $program completed successfully with batch size $batch_size"
    else
        echo "✗ $program failed with batch size $batch_size"
    fi
}

# Run each program with each batch size
for program in "${PROGRAMS[@]}"; do
    echo "----------------------------------------"
    echo "Running $program..."
    echo "----------------------------------------"
    
    for batch_size in "${BATCH_SIZES[@]}"; do
        run_program "$program" "$batch_size"
        echo "----------------------------------------"
    done
done

echo "All runs completed. Results are saved in the 'results' directory." 