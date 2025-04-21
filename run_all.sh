#!/bin/bash

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Create timestamp for the run
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
echo "Starting runs at: $TIMESTAMP"

# Create a directory for this run's results
RUN_DIR="$SCRIPT_DIR/results_$TIMESTAMP"
mkdir -p "$RUN_DIR"
cd "$RUN_DIR"

echo "----------------------------------------"
echo "Running wastedquery.sh..."
echo "----------------------------------------"
# Run wastedquery.sh and save its output
"$SCRIPT_DIR/wastedquery.sh" > wastedquery_output.txt 2>&1
if [ $? -eq 0 ]; then
    echo "✓ wastedquery.sh completed successfully"
else
    echo "✗ wastedquery.sh failed"
fi

echo "----------------------------------------"
echo "Running run_motivation_figure4.sh..."
echo "----------------------------------------"
# Run run_motivation_figure4.sh and save its output
"$SCRIPT_DIR/run_motivation_figure4.sh" > motivation_figure4_output.txt 2>&1
if [ $? -eq 0 ]; then
    echo "✓ run_motivation_figure4.sh completed successfully"
else
    echo "✗ run_motivation_figure4.sh failed"
fi

echo "----------------------------------------"
echo "All runs completed. Results are saved in directory: $RUN_DIR"
echo "----------------------------------------" 