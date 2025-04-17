import pandas as pd
import re

df = pd.read_csv('mathematical-qa/results/llama_results_without_explanations.csv')

# Rename ground_truth to ground_Truth_Full
df = df.rename(columns={'ground_truth': 'ground_truth_full'})

# Extract ground truth using regex
def extract_answer(text):
    if isinstance(text, str):
        match = re.search(r'The answer is: (.+?)(?:$|")', text)
        if match:
            return match.group(1).strip()
    return None

# Create a new column with the extracted answers
df['ground_truth'] = df['ground_truth_full'].apply(extract_answer)

# Save the processed data to a new CSV file
df.to_csv('mathematical-qa/results/llama_results_without_explanations.csv', index=False)

print("CSV processing complete.")