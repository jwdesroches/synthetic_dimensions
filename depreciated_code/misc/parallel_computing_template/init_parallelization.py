import os
import pandas as pd
import numpy as np
import json

variable_1s = np.linspace(1, 10, 10)
variable_2s = np.linspace(1, 10, 10)

data = []

for variable_1 in variable_1s:
    for variable_2 in variable_2s:
        data.append({"variable_1": variable_1, "variable_2": variable_2})

df = pd.DataFrame(data)
df.to_csv("input_parameters.csv", index=False)

os.makedirs("inputs", exist_ok=True)
os.makedirs("outputs", exist_ok=True)