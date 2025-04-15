import re
import os
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
from huggingface_hub import InferenceClient
import argparse
import time
from typing import List, Dict, Tuple, Any, Optional


class MathProcessor:
    """
    A class for processing math problem datasets from Hugging Face.
    
    This class takes a Hugging Face dataset containing mathematical problems with instructions and answers,
    and processes it to extract:
    1. Original instruction (with the "Please solve..." prefix)
    2. Clean instruction (without the "Please solve..." prefix)
    3. Full answer (including solution steps)
    4. Final answer (just the numerical or algebraic result)
    """
    
    def __init__(self, dataset_name: str = "reecursion/mmiqc-subset"):
        """
        Initialize the processor with the Hugging Face dataset name.
        
        Args:
            dataset_name: Name of the Hugging Face dataset
        """
        self.dataset_name = dataset_name
        self.dataset = None
        self.processed_dataset = None
        
    def load_dataset(self) -> None:
        """Load the dataset from Hugging Face."""
        self.dataset = load_dataset(self.dataset_name)
    
    def _extract_clean_instruction(self, instruction: str) -> str:
        """
        Extract the clean instruction by removing the standard prefix.
        
        Args:
            instruction: The original instruction with prefix
            
        Returns:
            The instruction without the standard prefix
        """
        # Fixed the prefix to match exactly what's in the dataset
        prefix = 'Please solve the following problem and put your answer at the end with "The answer is: ".\n'
        
        if instruction.startswith(prefix):
            return instruction[len(prefix):].strip()
        
        # Try alternative prefix formats if the main one doesn't match
        alt_prefix = 'Please solve the following problem and put your answer at the end with "The answer is: ". '
        if instruction.startswith(alt_prefix):
            return instruction[len(alt_prefix):].strip()
            
        # If no prefix matches, return the original instruction
        return instruction
    
    def _extract_final_answer(self, answer: str) -> str:
        """
        Extract just the final answer from the full answer.
        
        Args:
            answer: The full answer including steps
            
        Returns:
            Just the final numerical or algebraic result
        """
        final_answer = ""
        
        lines = answer.split('\n')
        for line in reversed(lines):
            if "The answer is:" in line:
                final_answer = line.split("The answer is:")[-1].strip()
                break
        
        return final_answer
    
    def process_dataset(self) -> Dict:
        """
        Process the dataset to extract clean instructions and final answers.
        
        Returns:
            The processed dataset with added columns
        """
        if not self.dataset:
            self.load_dataset()
        
        processed_splits = {}
        
        for split_name, split_data in self.dataset.items():
            # Create a new DataFrame from the split
            df = pd.DataFrame(split_data)
            
            # Add the clean instruction column
            df['clean_instruction'] = df['instruction'].apply(self._extract_clean_instruction)
            
            # Add the final answer column
            df['final_answer'] = df['output'].apply(self._extract_final_answer)
            
            # Store the processed split
            processed_splits[split_name] = df
        
        self.processed_dataset = processed_splits
        return self.processed_dataset
    
    def save_processed_dataset(self, output_dir: str) -> None:
        """
        Save the processed dataset splits to CSV files.
        
        Args:
            output_dir: Directory where the processed data will be saved
        """
        import os
        
        if self.processed_dataset is None:
            self.process_dataset()
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save each split to a separate CSV file
        for split_name, split_data in self.processed_dataset.items():
            output_path = os.path.join(output_dir, f"{split_name}.csv")
            split_data.to_csv(output_path, index=False)
            print(f"Processed {split_name} split saved to {output_path}")
    
    def get_sample(self, split: str = "train", n: int = 5) -> pd.DataFrame:
        """
        Get a sample of n processed entries from a specific split.
        
        Args:
            split: The dataset split to sample from
            n: Number of samples to return
            
        Returns:
            A pandas DataFrame with n samples
        """
        if self.processed_dataset is None:
            self.process_dataset()
        
        return self.processed_dataset[split].head(n)
    
    def to_huggingface_dataset(self) -> Any:
        """
        Convert the processed dataframes back to a Hugging Face Dataset object.
        
        Returns:
            A Hugging Face Dataset object with the additional columns
        """
        from datasets import Dataset, DatasetDict
        
        if self.processed_dataset is None:
            self.process_dataset()
        
        hf_datasets = {}
        for split_name, split_data in self.processed_dataset.items():
            hf_datasets[split_name] = Dataset.from_pandas(split_data)
        
        return DatasetDict(hf_datasets)

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
        results_df.to_csv("/home/gganeshl/mathematical-qa/results/" + partial_file, index=False)
        print(f"Saved partial results to {partial_file}")

def main():
    # Set up command line arguments
    parser = argparse.ArgumentParser(description='Evaluate Llama on math problems')
    parser.add_argument('--provider', type=str, default="nebius",
                        help='API provider (nebius, huggingface, etc.)')
    parser.add_argument('--model', type=str, default="meta-llama/Llama-3.1-8B-Instruct",
                        help='Model ID to use')
    parser.add_argument('--dataset', type=str, default="reecursion/mmiqc-subset",
                        help='Dataset ID to use')
    parser.add_argument('--limit', type=int, default=-1,
                        help='Limit to N examples (-1 for all)')
    parser.add_argument('--output', type=str, default="llama_results.csv",
                        help='Output CSV file')
    parser.add_argument('--temperature', type=float, default=0.3,
                        help='Temperature for generation')
    parser.add_argument('--save_interval', type=int, default=10,
                        help='Save partial results every N examples')
    parser.add_argument('--modification', type=str, default="Please solve the following problem and give a concise explanation with the answer at the end with \"The answer is: \". ",
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
    # dataset = load_dataset("reecursion/mmiqc-subset", split="test")

    processor = MathProcessor(dataset_name=args.dataset)
    dataset = processor.process_dataset()['test']
    dataset = processor.to_huggingface_dataset()['test']

    
    if args.limit > 0:
        dataset = dataset.select(range(min(args.limit, len(dataset))))
    
    print(f"Dataset loaded with {len(dataset)} examples")
    
    # Evaluation loop
    results = []
    
    print("Starting evaluation...")
    for idx, example in enumerate(tqdm(dataset)):
        instruction = example["clean_instruction"]
        output = example["output"]
        
        # Extract ground truth answer
        match = re.search(r"The answer is:\s*(.*?)(?:\s*$|\n)", output)
        if not match:
            print(f"Warning: Could not extract ground truth for example {idx}")
            # continue
            
        ground_truth = match.group(1).strip()
        
        # Modify the instruction to ask for concise explanation
        modified_instruction = args.modification + instruction 
        
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
                "question": instruction,
                "ground_truth": ground_truth,
                "prompt": modified_instruction,
                "predicted": predicted_answer,
                'ground_truth': output,
                "model_response": model_response,
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
    results_df.to_csv("/home/gganeshl/mathematical-qa/results/" + args.output, index=False)
    print(f"Results saved to {args.output}")

if __name__ == "__main__":
    main()