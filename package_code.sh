#!/bin/bash

# Default output file name
OUTPUT_FILE="combined_code.txt"

# --- Configuration: Define what to include ---
# Add your source directories and specific files here.
# Paths should be relative to where you run the script from.
# Directories will be scanned recursively.
# Use spaces to separate items.
FILES_TO_INCLUDE=(
    "base_modules.py"
    "utils.py"
    "module_discovery.py"
    "app.py"
    "project_manager.py"
    "task_executor.py"
    "ui_task_executor.py"
    "config_manager.py"
    "video_assembly.py"
    "llm_modules/"
    "tts_modules/"
    "t2i_modules/"
    "i2v_modules/"
    "t2v_modules/"
)

# --- End of Configuration ---

# Check if an output file name was provided as an argument
if [ "$1" ]; then
  OUTPUT_FILE="$1"
  echo "Using custom output file name: $OUTPUT_FILE"
fi

# Clear the output file to start fresh
> "$OUTPUT_FILE"
echo "Cleared old content from $OUTPUT_FILE."

# A function to process and append a file to the output
process_file() {
    local file_path=$1
    echo "Processing: $file_path"
    
    # Write the header with the relative file path
    echo "==== $file_path ====" >> "$OUTPUT_FILE"
    
    # Append the content of the file
    cat "$file_path" >> "$OUTPUT_FILE"
    
    # Add multiple newlines at the end for better separation
    echo -e "\n\n\n" >> "$OUTPUT_FILE"
}

# Loop through the configured list of files and directories
for item in "${FILES_TO_INCLUDE[@]}"; do
    if [ -f "$item" ]; then
        # If it's a single file, process it directly
        process_file "$item"
    elif [ -d "$item" ]; then
        # If it's a directory, find all relevant files inside it
        # - The `find` command is powerful.
        # - It searches for items of type 'f' (file).
        # - It ignores paths containing '__pycache__', '.git', '.vscode', etc.
        # - It only includes files ending in '.py' or other specified extensions.
        find "$item" -type f \( -name "*.py" -o -name "*.sh" \) \
        -not -path "*/__pycache__/*" \
        -not -path "*/.git/*" \
        -not -path "*/.venv/*" \
        -not -path "*/.vscode/*" \
        | sort | while read -r file; do
            process_file "$file"
        done
    else
        echo "Warning: Item '$item' not found. Skipping."
    fi
done

echo "========================================="
echo "✅ All done!"
echo "Combined code saved to: $OUTPUT_FILE"
echo "========================================="