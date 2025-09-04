import pandas as pd
import re
from pylatexenc.latex2text import LatexNodes2Text

# Function to remove LaTeX
def latex_to_text(latex_str):
    text = LatexNodes2Text().latex_to_text(latex_str)
    text = re.sub(r'\s+', ' ', text)  # collapse multiple spaces/newlines
    return text.strip()

# Function to extract answer inside \boxed{...}
def extract_boxed_answer(latex_str):
    start = latex_str.find(r'\boxed{')
    if start == -1:
        return ""  # No boxed answer
    start += len(r'\boxed{')
    
    # Use a stack to find the matching closing brace
    stack = 1
    i = start
    while i < len(latex_str) and stack > 0:
        if latex_str[i] == '{':
            stack += 1
        elif latex_str[i] == '}':
            stack -= 1
        i += 1
    
    return latex_str[start:i-1].strip()

# Process CSV
def clean_math_dataset(input_csv, output_csv):
    # Load data
    df = pd.read_csv(input_csv)

    # Clean problem and solution
    df['problem'] = df['problem'].apply(latex_to_text)

    # Extract answer from original LaTeX solution and clean it
    df['answer'] = df['solution'].apply(lambda sol: latex_to_text(extract_boxed_answer(sol)))
    df['solution'] = df['solution'].apply(latex_to_text)


    # Save cleaned CSV
    df.to_csv(output_csv, index=False)
    print(f"Cleaned data saved to {output_csv}")

# Example usage
clean_math_dataset("raw_datasets/MATH/math_train.csv", "math_train_cleaned.csv")
