# run_job.py
import sys
import json

def example_function(input_1, input_2):
    output = input_1 + input_2
    return output

if __name__ == "__main__":
    input_file = sys.argv[1]
    output_file = sys.argv[2]

    with open(input_file, "r") as f:
        data = json.load(f)

    result = example_function(data["variable_1"], data["variable_2"])

    with open(output_file, "w") as f:
        f.write(str(result))
