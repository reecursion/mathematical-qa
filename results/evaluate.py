import os
import pandas as pd
import numpy as np

# Define the base directory (update if needed)
base_dir = 'results/deepmind_math/flan-t5-xl'

# Subdirectories and scaling options
folders = ['encoder', 'decoder', 'both']
num_scalings = [0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
op_scalings = [0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]

# Function to calculate accuracy and MSE
def calculate_metrics(df):
    accuracy = (df['correct'] == 1).mean()
    mse = np.mean((df['ground_truth'] - df['predicted'])**2)
    return accuracy, mse

# Dictionary to store results
results = {
    folder: {
        'accuracy': np.zeros((len(num_scalings), len(op_scalings))),
        'mse': np.zeros((len(num_scalings), len(op_scalings)))
    } for folder in folders
}

# Process each folder
for folder in folders:
    folder_path = os.path.join(base_dir, folder)
    
    for i, num in enumerate(num_scalings):
        for j, op in enumerate(op_scalings):
            filename = f"inference_num{num:.1f}_op{op:.1f}.csv"
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
from textwrap import dedent

# LaTeX preamble with required packages
latex_preamble = """% Add these packages to your LaTeX document preamble:
% \\usepackage{xcolor}
% \\usepackage{colortbl}
% \\usepackage{array}

"""

latex_tables = latex_preamble
for folder in folders:
    for metric in ['accuracy', 'mse']:
        matrix = results[folder][metric]
        # Determine baseline indices corresponding to (1.0, 1.0)
        baseline_i = num_scalings.index(1.0) if 1.0 in num_scalings else len(num_scalings) // 2
        baseline_j = op_scalings.index(1.0) if 1.0 in op_scalings else len(op_scalings) // 2
        baseline = matrix[baseline_i, baseline_j]

        # Compute delta matrix where positive = better
        if metric == 'accuracy':
            delta_matrix = matrix - baseline
        else:  # mse (lower is better)
            delta_matrix = baseline - matrix
        max_abs_delta = float(np.nanmax(np.abs(delta_matrix))) if np.any(delta_matrix != 0) else 0.0
        
        latex_tables += f"\\begin{{table}}[h]\n\\centering\n"
        metric_label = 'MSE' if metric == 'mse' else 'Accuracy'
        latex_tables += f"\\caption{{{metric_label} for {folder}}}\n"
        col_spec = f"c|*{{{len(op_scalings)}}}{{c}}"
        latex_tables += f"\\begin{{tabular}}{{{col_spec}}}\n"
        
        # Fixed header row
        header_ops = " & ".join(f"{op:.1f}" for op in op_scalings)
        latex_tables += f"Num $\\backslash$ Op & {header_ops} \\\\\n\\hline\n"
        
        for i, num in enumerate(num_scalings):
            row_cells = []
            for j, op in enumerate(op_scalings):
                value = matrix[i, j]
                delta = delta_matrix[i, j]
                if max_abs_delta > 0:
                    intensity = int(round(min(100, max(0, (abs(delta) / max_abs_delta) * 80 + 10))))
                else:
                    intensity = 0
                # No arrow for baseline
                if i == baseline_i and j == baseline_j:
                    arrow = ""
                else:
                    arrow = " $\\uparrow$" if delta > 0 else (" $\\downarrow$" if delta < 0 else "")
                
                if i == baseline_i and j == baseline_j:
                    # Baseline: light gray background, bold text
                    cell = f"\\cellcolor{{gray!20}}\\textbf{{{value:.4f}}}"
                elif intensity > 0:
                    # Use much lighter colors for readability
                    if delta > 0:
                        # Light green for improvements
                        color_intensity = min(30, intensity // 3 + 10)
                        cell = f"\\cellcolor{{green!{color_intensity}}}{value:.4f}{arrow}"
                    else:
                        # Light red for degradations
                        color_intensity = min(30, intensity // 3 + 10)
                        cell = f"\\cellcolor{{red!{color_intensity}}}{value:.4f}{arrow}"
                else:
                    cell = f"{value:.4f}{arrow}"
                row_cells.append(cell)
            
            latex_tables += f"{num:.1f} & " + " & ".join(row_cells) + " \\\\\n"
        
        latex_tables += "\\end{tabular}\n\\end{table}\n\n"

print(latex_tables)