#!/bin/bash

target=0.509548
threshold=0.02
count=0
total_runs=100

for ((i = 1; i <= total_runs; i++)); do
    result=$(pypy3 test_qfl.py 0 9 250000)
    echo "Run $i: $result"

    min_allowed=$(echo "$target - $threshold" | bc -l)
    if (( $(echo "$result < $min_allowed" | bc -l) )); then
        ((count++))
    fi
done

echo ""
echo "Number of runs with result < $(echo "$target - $threshold" | bc -l): $count out of $total_runs"
