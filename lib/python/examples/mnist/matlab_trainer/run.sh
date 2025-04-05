#!/bin/bash

# Run all Python processes in parallel
for i in {1..2}
do
    echo "Starting process with config${i}.json"
    python main.py "config${i}.json" &
done

# Wait for all background processes to complete
wait

echo "All processes have completed"
