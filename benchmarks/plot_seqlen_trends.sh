#!/bin/bash

# Script to run benchmarks for all sequence length combinations and generate plots
# Based on the sequence lengths shown in the terminal output

set -e  # Exit on any error

# Array of sequence length combinations
# Format: "decode_len prefill_len prefill_chunk_size num_prefill_reqs num_decode_reqs"
seq_configs=(
    "1024 1024 1024 1 128" # 1k prefill, 1k decode
    "2048 1024 1024 1 128" # 1k prefill, 2k decode
    "4096 1024 1024 1 128" # 1k prefill, 4k decode
    "8192 1024 1024 1 128" # 1k prefill, 8k decode
    "2048 4096 4096 1 128" # 4k prefill, 2k decode
    "4096 4096 4096 1 128" # 4k prefill, 4k decode
    "8192 4096 4096 1 128" # 4k prefill, 8k decode
    "16384 4096 4096 1 128" # 4k prefill, 16k decode
    "1024 8192 8192 1 128" # 8k prefill, 1k decode
    "4096 8192 8192 1 128" # 8k prefill, 4k decode
    "8192 8192 8192 1 128" # 8k prefill, 8k decode
    "16384 8192 8192 1 128" # 8k prefill, 16k decode
    "4096 16384 16384 1 128" # 16k prefill, 4k decode
    "8192 16384 16384 1 128" # 16k prefill, 8k decode

)
repeats=100
save_dir="bench_plots/persistent"

echo "Starting benchmark runs for all sequence length combinations..."
echo "Total configurations: ${#seq_configs[@]}"
echo

# Counter for progress tracking
count=0

for config in "${seq_configs[@]}"; do
    count=$((count + 1))

    # Parse the configuration
    read -r decode_len prefill_len prefill_chunk_size num_prefill_reqs num_decode_reqs <<< "$config"

    echo "=========================================="
    echo "[$count/${#seq_configs[@]}] Running benchmark:"
    echo "  Decode length: ${decode_len}"
    echo "  Prefill length: ${prefill_len}"
    echo "  Prefill chunk size: ${prefill_chunk_size}"
    echo "  Num prefill reqs: $num_prefill_reqs"
    echo "  Num decode reqs: $num_decode_reqs"
    echo "=========================================="

    # Run the benchmark
    echo "Running benchmark with $repeats repeats..."
    python3 benchmarks/bench_batch_attention.py \
        --repeats $repeats \
        --prefill_chunk_size $prefill_chunk_size \
        --num_prefill_reqs $num_prefill_reqs \
        --num_decode_reqs $num_decode_reqs \
        --decode_len $decode_len \
        --prefill_len $prefill_len

    # Check if benchmark completed successfully
    if [ $? -eq 0 ]; then
        echo "✓ Benchmark completed successfully"
    else
        echo "✗ Benchmark failed"
        exit 1
    fi

    echo
done

echo "=========================================="
echo "All benchmarks completed!"
echo "CSV file with all results: bench_batch_attention.csv"
echo

# Generate trend plots from accumulated data
echo "Generating trend plots from all accumulated data..."
python3 benchmarks/plotting/plot_persistent_trends.py

if [ $? -eq 0 ]; then
    echo "✓ Trend plots generated successfully"
    echo "Generated trend plots:"
    ls -la ${save_dir}/persistent_trends_*.png
else
    echo "✗ Failed to generate trend plots"
fi

echo "=========================================="
