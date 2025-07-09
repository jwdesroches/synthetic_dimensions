#! /bin/sh

# Ensure a file is provided as input
if [ -z "$1" ]; then
    echo "Usage: $0 <csv_file>"
    exit 1
fi

mkdir -p input
mkdir -p output

# Skip the first line (header) by starting from line 2
# We start the loop at 2 because line 1 is the header
for a in $(seq 2 1 $(wc -l < $1)); do
    # Generate the file name (e.g., input_1.py, input_2.py)
    FILEINDX=$((a - 1))  # Subtract 1 to get file names starting from input_1.py
    
    # Copy the template Python file to the 'input' folder
    cp template.py input/input_${FILEINDX}.py
    
    # Extract the number of columns (parameters) in the current line
    num_params=$(sed -n "${a}p" $1 | sed 's/[^,]//g' | wc -c)
    num_params=$((num_params + 1))  # Adjust for 1-based column count

    # Loop over the number of parameters in the current line
    for b in $(seq 1 1 $num_params); do
        # Extract the b-th parameter from the a-th line
        PARAMINP=$(sed -n "${a}p" $1 | cut -d, -f${b})
        
        # Replace PARAMb with the extracted parameter in the copied Python file
        sed -i "s/PARAM${b}/${PARAMINP}/" input/input_${FILEINDX}.py
        
        # Print a progress dot
        echo -n '.'
    done
    echo ""  # Newline after the dots for each line
done
