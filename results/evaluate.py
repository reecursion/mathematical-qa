import os
import pandas as pd
import numpy as np

# Define the base directory (update if needed)
base_dir = 'results/deepmind_math/flan-t5-xl'

# Subdirectories and scaling options
folders = ['encoder', 'decoder', 'both']
num_scalings = [0.6, 1.0, 1.4]
op_scalings = [0.6, 1.0, 1.4]

# Function to calculate accuracy and MSE
def calculate_metrics(df):
    accuracy = (df['correct'] == 1).mean()
    mse = np.mean((df['ground_truth'] - df['predicted'])**2)
    return accuracy, mse

# Dictionary to store results
results = {folder: {'accuracy': np.zeros((3, 3)), 'mse': np.zeros((3, 3))} for folder in folders}

# Process each folder
for folder in folders:
    folder_path = os.path.join(base_dir, folder)
    
    for i, num in enumerate(num_scalings):
        for j, op in enumerate(op_scalings):
            filename = f"inference_num{num}_op{op}.csv"
            file_path = os.path.join(folder_path, filename)
            
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                
                # Ensure numeric conversion for ground_truth and predicted
                df['ground_truth'] = pd.to_numeric(df['ground_truth'], errors='coerce')
                df['predicted'] = pd.to_numeric(df['predicted'], errors='coerce')
                
                acc, mse = calculate_metrics(df)
                results[folder]['accuracy'][i, j] = acc
                results[folder]['mse'][i, j] = mse

# Generate LaTeX tables
latex_tables = ""
for folder in folders:
    for metric in ['accuracy', 'mse']:
        latex_tables += f"\\begin{{table}}[h]\n\\centering\n\\caption{{{metric.capitalize()} for {folder}}}\n"
        latex_tables += "\\begin{tabular}{c|ccc}\n"
        latex_tables += "Num \\textbackslash Op & 0.6 & 1.0 & 1.4 \\\\\n\\hline\n"
        
        for i, num in enumerate(num_scalings):
            row = f"{num} & " + " & ".join([f"{results[folder][metric][i, j]:.4f}" for j in range(3)])
            latex_tables += row + " \\\\\n"
        
        latex_tables += "\\end{tabular}\n\\end{table}\n\n"

print(latex_tables)
