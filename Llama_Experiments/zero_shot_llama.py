import re
import os
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
from huggingface_hub import InferenceClient
import argparse
import time

def extract_answer(response):
    match = re.search(r"The answer is:\s*(.*?)(?:\s*$|\n)", response)
    if match:
        return match.group(1).strip()
    return None

def is_correct(predicted, actual):
    if not predicted or not actual:
        return False
        
    # Clean up both answers for comparison
    predicted_clean = predicted.lower().strip()
    actual_clean = actual.lower().strip()
    
    # Try exact match first
    if predicted_clean == actual_clean:
        return True
    
    # Handle numeric answers with different formats
    try:
        # Try to convert to float for numeric comparison
        pred_num = float(predicted_clean.replace(',', ''))
        actual_num = float(actual_clean.replace(',', ''))
        return abs(pred_num - actual_num) < 1e-6
    except:
        pass
    
    # Handle fraction comparisons
    if '/' in predicted_clean and '/' in actual_clean:
        try:
            # Convert fractions to float
            pred_parts = predicted_clean.split('/')
            actual_parts = actual_clean.split('/')
            pred_frac = float(pred_parts[0]) / float(pred_parts[1])
            actual_frac = float(actual_parts[0]) / float(actual_parts[1])
            return abs(pred_frac - actual_frac) < 1e-6
        except:
            pass
    
    return False

def save_partial_results(results, output_file, suffix="_partial"):
    """Save partial results to avoid losing progress."""
    filename, ext = os.path.splitext(output_file)
    partial_file = f"{filename}{suffix}{ext}"
    
    # Create a DataFrame from current results
    if results:
        results_df = pd.DataFrame(results)
        results_df.to_csv("/home/gganeshl/GenAI/results/" + partial_file, index=False)
        print(f"Saved partial results to {partial_file}")

def main():
    # Set up command line arguments
    parser = argparse.ArgumentParser(description='Evaluate Llama on math problems')
    parser.add_argument('--provider', type=str, default="nebius",
                        help='API provider (nebius, huggingface, etc.)')
    parser.add_argument('--model', type=str, default="meta-llama/Llama-3.1-8B-Instruct",
                        help='Model ID to use')
    parser.add_argument('--limit', type=int, default=-1,
                        help='Limit to N examples (-1 for all)')
    parser.add_argument('--output', type=str, default="llama_results.csv",
                        help='Output CSV file')
    parser.add_argument('--temperature', type=float, default=0.3,
                        help='Temperature for generation')
    parser.add_argument('--save_interval', type=int, default=10,
                        help='Save partial results every N examples')
    parser.add_argument('--modification', type=str, default="\nGive concise explanation before 'The answer is: ', and give only numbers without symbols. If Latex is used, try to give Latex representation of answers.",
                        help='Modification of base prompt')
    args = parser.parse_args()
    
    # Get API key from environment variable
    api_key = os.environ.get("HF_API_KEY")
    if not api_key:
        raise ValueError("HF_API_KEY environment variable is not set")
    
    # Initialize the inference client
    client = InferenceClient(
        provider=args.provider,
        api_key=api_key,
    )
    
    print("Loading dataset...")
    dataset = load_dataset("reecursion/mmiqc-subset", split="test")
    
    if args.limit > 0:
        dataset = dataset.select(range(min(args.limit, len(dataset))))
    
    print(f"Dataset loaded with {len(dataset)} examples")
    
    # Evaluation loop
    results = []
    
    print("Starting evaluation...")
    for idx, example in enumerate(tqdm(dataset)):
        instruction = example["instruction"]
        output = example["output"]
        
        # Extract ground truth answer
        match = re.search(r"The answer is:\s*(.*?)(?:\s*$|\n)", output)
        if not match:
            print(f"Warning: Could not extract ground truth for example {idx}")
            # continue
            
        ground_truth = match.group(1).strip()
        
        # Modify the instruction to ask for concise explanation
        modified_instruction = instruction + args.modification
        
        # Generate response from model using Inference API
        try:
            completion = client.chat.completions.create(
                model=args.model,
                messages=[
                    {
                        "role": "user",
                        "content": modified_instruction
                    }
                ],
                max_tokens=512,
                temperature=args.temperature,
            )
            
            model_response = completion.choices[0].message.content
            
            # Extract predicted answer
            predicted_answer = extract_answer(model_response)
            if not predicted_answer:
                print(f"Warning: Could not extract prediction for example {idx}")
                # continue
            
            # Store results
            results.append({
                "idx": idx,
                "instruction": instruction,
                "ground_truth": ground_truth,
                "predicted": predicted_answer,
                'actual_full_response': output,
                "model_full_response": model_response,
            })
            
            # Save partial results periodically
            if (idx + 1) % args.save_interval == 0:
                save_partial_results(results, args.output)
                print(f"Progress: {idx+1}/{len(dataset)}")
                
        except Exception as e:
            print(f"Error on example {idx}: {e}")
            # Save partial results on error to avoid losing progress
            save_partial_results(results, args.output, suffix=f"_error_at_{idx}")
            continue
    
    # Calculate overall accuracy
    # accuracy = sum(r["correct"] for r in results) / len(results)
    # print(f"\nOverall accuracy: {accuracy:.4f}")
    
    # Save results to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv("/home/gganeshl/GenAI/results/" + args.output, index=False)
    print(f"Results saved to {args.output}")

if __name__ == "__main__":
    main()