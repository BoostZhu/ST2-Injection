#!/bin/bash

# Define the root directories
DATA_ROOT="./data"
STYLE_DIR="$DATA_ROOT/style"
CONTENT_DIR="$DATA_ROOT/content"
OUTPUT_DIR="./results/pa_anchor_ma1"

# Ensure the output directory exists
mkdir -p "$OUTPUT_DIR"

# Loop through each style image in the style directory
for style_path in "$STYLE_DIR"/*; do
  # Extract the filename from the path (e.g., "wave.png")
  style_name=$(basename "$style_path")

  # Loop through each content folder in the content directory
  for content_path in "$CONTENT_DIR"/*; do
    # Check if it's actually a directory
    if [ -d "$content_path" ]; then
      # Extract the directory name from the path (e.g., "car")
      content_name=$(basename "$content_path")

      # Print the current combination being processed
      echo "-----------------------------------------------------"
      echo "Applying style '$style_name' to content '$content_name'"
      echo "-----------------------------------------------------"

      # Execute the Python script with the current combination
      python run_styleid_v2v.py \
        --content_name "$content_name" \
        --style_name "$style_name" \
        --data_root "$DATA_ROOT" \
        --output_dir "$OUTPUT_DIR"
        # You can add other parameters like --ddim_steps here if needed
      
      echo "Finished processing '$content_name' with '$style_name'."
      echo ""
    fi
  done
done

echo "All style transfer combinations are complete."