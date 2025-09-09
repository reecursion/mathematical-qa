import torch
import pandas as pd
import numpy as np
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import re
import argparse
from tqdm import tqdm
from typing import List, Dict, Tuple, Any, Optional
import os

class EnhancedFlanT5Inference:
    def __init__(self, model_name="google/flan-t5-base", prompt="", device=None, debug=True):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.debug = debug
        print(f"Using device: {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, device_map="cuda")
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="cuda")
        if self.debug:
            print("Model and tokenizer loaded.")
        
        # Store original weights for both encoder and decoder
        self.original_state_dict = {
            k: v.clone() for k, v in self.model.state_dict().items()
        }
        
        self.prompt = prompt
        if self.debug:
            print(f"Prompt being used is {self.prompt}")
            encoder_weights = sum(1 for k in self.original_state_dict if 'encoder' in k)
            decoder_weights = sum(1 for k in self.original_state_dict if 'decoder' in k)
            print(f"Original attention weights snapshot stored: {len(self.original_state_dict)} tensors "
                  f"({encoder_weights} encoder, {decoder_weights} decoder).")
        
        # Prepare enhanced mathematical word lists
        self._prepare_enhanced_math_word_lists()
        # Initialize token mappings from fixed lists
        self._prepare_enhanced_token_mappings()

    def _prepare_enhanced_math_word_lists(self):
        """Enhanced mathematical word lists with better categorization."""
        
        # Core number words (high confidence)
        self.core_number_words = {
            "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
            "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen",
            "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety",
            "hundred", "thousand", "million", "billion"
        }
        
        # Core operation words (high confidence)
        self.core_operation_words = {
            "plus", "minus", "times", "equals", "equal", "add", "subtract", "multiply", "divide",
            "sum", "total", "difference", "product"
        }
        
        # Mathematical symbols (very high confidence)
        self.math_symbols = {"+", "-", "*", "/", "=", "<", ">", "×", "÷", "±", "≠", "≤", "≥", "%"}
        
        # Contextual math words (only boost in mathematical contexts)
        self.contextual_math_words = {
            "quotient", "remainder", "average", "mean", "percent", "percentage", 
            "ratio", "proportion", "fraction", "decimal", "greater", "less", "than"
        }

    def _prepare_enhanced_token_mappings(self):
        """Enhanced token mappings with better classification."""
        
        # Core mathematical tokens (always boost)
        self.core_number_tokens = set()
        self.core_operator_tokens = set()
        
        # Process single digits
        for digit in "0123456789":
            tokens = self.tokenizer.encode(digit, add_special_tokens=False)
            self.core_number_tokens.update(tokens)
        
        # Process core number words
        for word in self.core_number_words:
            tokens = self.tokenizer.encode(word, add_special_tokens=False)
            self.core_number_tokens.update(tokens)
            # Also try capitalized
            tokens = self.tokenizer.encode(word.capitalize(), add_special_tokens=False)
            self.core_number_tokens.update(tokens)
        
        # Process mathematical symbols
        for symbol in self.math_symbols:
            tokens = self.tokenizer.encode(symbol, add_special_tokens=False)
            self.core_operator_tokens.update(tokens)
        
        # Process core operation words
        for word in self.core_operation_words:
            tokens = self.tokenizer.encode(word, add_special_tokens=False)
            self.core_operator_tokens.update(tokens)
            # Also try capitalized
            tokens = self.tokenizer.encode(word.capitalize(), add_special_tokens=False)
            self.core_operator_tokens.update(tokens)
        
        # Contextual tokens (for context-aware boosting)
        self.contextual_math_tokens = set()
        for word in self.contextual_math_words:
            tokens = self.tokenizer.encode(word, add_special_tokens=False)
            self.contextual_math_tokens.update(tokens)
        
        if self.debug:
            print(f"Core number tokens: {len(self.core_number_tokens)}")
            print(f"Core operator tokens: {len(self.core_operator_tokens)}")
            print(f"Contextual math tokens: {len(self.contextual_math_tokens)}")

    def classify_tokens_enhanced(self, question):
        """Enhanced token classification with context awareness."""
        
        tokens = self.tokenizer.encode(question, add_special_tokens=False)
        token_strs = self.tokenizer.convert_ids_to_tokens(tokens)
        
        # Always boost these (high confidence mathematical tokens)
        definite_number_tokens = []
        definite_operator_tokens = []
        
        # Context-dependent tokens (only boost if in mathematical context)
        contextual_tokens = []
        
        # Check overall mathematical density of the question
        math_density = self._calculate_math_density(tokens)
        is_math_heavy = math_density > 0.3  # At least 30% mathematical content
        
        for i, (token_id, token_str) in enumerate(zip(tokens, token_strs)):
            clean_token = token_str.lower().strip('▁')
            
            # Definite numbers: pure digits or core number words
            if re.match(r'^\d+$', clean_token) or token_id in self.core_number_tokens:
                definite_number_tokens.append(token_id)
            
            # Definite operators: mathematical symbols or core operation words
            elif clean_token in self.math_symbols or token_id in self.core_operator_tokens:
                definite_operator_tokens.append(token_id)
            
            # Contextual: mathematical words that depend on context
            elif token_id in self.contextual_math_tokens and is_math_heavy:
                contextual_tokens.append(token_id)
            
            # Additional context check: mathematical patterns
            elif self._has_mathematical_neighbors(i, tokens, token_strs):
                if clean_token in ["of", "by", "per", "each"]:  # Mathematical connectors
                    contextual_tokens.append(token_id)
        
        if self.debug:
            print(f"Question math density: {math_density:.3f}")
            print(f"Definite numbers: {len(definite_number_tokens)}")
            print(f"Definite operators: {len(definite_operator_tokens)}")
            print(f"Contextual: {len(contextual_tokens)}")
        
        return definite_number_tokens, definite_operator_tokens, contextual_tokens

    def _calculate_math_density(self, tokens):
        """Calculate the density of mathematical content in the question."""
        math_count = 0
        for token_id in tokens:
            if (token_id in self.core_number_tokens or 
                token_id in self.core_operator_tokens or 
                token_id in self.contextual_math_tokens):
                math_count += 1
        return math_count / len(tokens) if tokens else 0

    def _has_mathematical_neighbors(self, position, tokens, token_strs, window=2):
        """Check if a token has mathematical neighbors within a window."""
        start = max(0, position - window)
        end = min(len(tokens), position + window + 1)
        
        for i in range(start, end):
            if i != position:
                token_id = tokens[i]
                token_str = token_strs[i].lower().strip('▁')
                
                # Check for mathematical neighbors
                if (re.match(r'^\d+$', token_str) or 
                    token_id in self.core_number_tokens or
                    token_str in self.math_symbols or
                    token_id in self.core_operator_tokens):
                    return True
        return False

    def modify_attention_enhanced(self, question=None, num_scaling=1.02, op_scaling=1.03, 
                                contextual_scaling=1.01, model_part="both", layer_range=None):
        """
        Enhanced attention modification with better token selection and layer targeting.
        
        Args:
            question: The question text to analyze
            num_scaling: Scaling for definite number tokens
            op_scaling: Scaling for definite operator tokens  
            contextual_scaling: Scaling for contextual mathematical tokens
            model_part: "encoder", "decoder", or "both"
            layer_range: tuple (start, end) for layer range, or None for all layers
        """
        
        # Always reset weights before any modification
        self.reset_weights()
        
        if not question:
            return
        
        # Get enhanced token classification
        number_tokens, operator_tokens, contextual_tokens = self.classify_tokens_enhanced(question)
        
        modified_params = 0
        
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                # Check if we should modify this parameter
                should_modify = self._should_modify_parameter(name, model_part, layer_range)
                
                if should_modify and 'SelfAttention' in name and 'relative_attention_bias' not in name:
                    if 'q.weight' in name or 'k.weight' in name:
                        modified_param = param.clone()
                        
                        # Apply number scaling
                        for token_id in number_tokens:
                            if token_id < param.size(1):
                                modified_param[:, token_id] *= num_scaling
                                modified_params += 1
                        
                        # Apply operator scaling
                        for token_id in operator_tokens:
                            if token_id < param.size(1):
                                modified_param[:, token_id] *= op_scaling
                                modified_params += 1
                        
                        # Apply contextual scaling (more conservative)
                        for token_id in contextual_tokens:
                            if token_id < param.size(1):
                                modified_param[:, token_id] *= contextual_scaling
                                modified_params += 1
                        
                        param.copy_(modified_param)
        
        if self.debug:
            layer_info = f" (layers {layer_range})" if layer_range else ""
            print(f"Enhanced modification{layer_info}: {modified_params} parameter positions modified")

    def _should_modify_parameter(self, name, model_part, layer_range=None):
        """Enhanced parameter selection with layer range support."""
        
        # Check model part
        part_match = False
        if model_part == "encoder" and 'encoder' in name:
            part_match = True
        elif model_part == "decoder" and 'decoder' in name:
            part_match = True
        elif model_part == "both" and ('encoder' in name or 'decoder' in name):
            part_match = True
        
        if not part_match:
            return False
        
        # Check layer range if specified
        if layer_range is not None:
            start_layer, end_layer = layer_range
            
            # Extract layer number from parameter name
            layer_match = re.search(r'layer\.(\d+)', name)
            if layer_match:
                layer_num = int(layer_match.group(1))
                if not (start_layer <= layer_num < end_layer):
                    return False
            else:
                # If we can't extract layer number, skip when layer_range is specified
                return False
        
        return True

    def reset_weights(self):
        """Reset model weights to original state."""
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in self.original_state_dict:
                    param.copy_(self.original_state_dict[name])
        if self.debug:
            print("Model attention weights reset to original.")

    def process_dataset(self, dataset_name="deepmind_math"):
        """Load dataset from CSV file."""
        csv_path = f"processed_dataset/{dataset_name}/test.csv"
        
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Dataset file not found: {csv_path}")
        
        df = pd.read_csv(csv_path)
        return df

    def prepare_prompt(self, question):
        """Prepare prompt with question."""
        return f"{self.prompt}{question}"

    def extract_final_answer(self, text):
        """Extract final numerical answer from model output."""
        # Look for numbers at the end of the response
        number_match = re.search(r'(-?\d+(?:\.\d+)?)\s*$', text.strip())
        if number_match:
            return number_match.group(1)
        
        return ""

    def run_inference(self, df, batch_size=8, num_scaling=1.02, op_scaling=1.03, 
                     contextual_scaling=1.01, model_part="both", layer_range=None):
        """Run inference with enhanced attention modifications."""
        
        results = []
        if self.debug:
            layer_info = f", layers {layer_range}" if layer_range else ""
            print(f"Running inference with scaling: num={num_scaling}, op={op_scaling}, "
                  f"contextual={contextual_scaling}, part={model_part}{layer_info}")
        
        total_examples = len(df)
        
        for i in tqdm(range(0, total_examples, batch_size)):
            batch = df.iloc[i:i+batch_size] 
            prompts = [self.prepare_prompt(q) for q in batch["question"]]  
            batch_results = []
            
            for j, (idx, row) in enumerate(batch.iterrows()):
                question = row["question"]
                
                # Apply enhanced attention modification
                self.modify_attention_enhanced(
                    question=question,
                    num_scaling=num_scaling,
                    op_scaling=op_scaling,
                    contextual_scaling=contextual_scaling,
                    model_part=model_part,
                    layer_range=layer_range
                )
                
                # Generate answer
                single_input = self.tokenizer([prompts[j]], return_tensors="pt", padding=True, truncation=True).to(self.device)
                with torch.no_grad():
                    output = self.model.generate(
                        input_ids=single_input.input_ids,
                        attention_mask=single_input.attention_mask,
                        max_length=512,
                        num_beams=4,
                        early_stopping=True,
                        past_key_values=None
                    )
                prediction = self.tokenizer.decode(output[0], skip_special_tokens=True)
                final_answer = self.extract_final_answer(prediction)
                
                batch_results.append({
                    "idx": idx,
                    "question": row["question"],
                    "prompt": prompts[j],
                    "ground_truth": row["answer"],
                    "model_response": prediction,
                    "predicted": final_answer
                })
                
                # Reset weights for next question
                self.reset_weights()
            
            results.extend(batch_results)
        
        return pd.DataFrame(results)

    def save_results(self, results_df, output_path="inference_results.csv", dataset_name="deepmind_math", 
                     model_name="google/flan-t5-base", model_part="both", layer_range=None,
                     num_scaling=1.0, op_scaling=1.0, contextual_scaling=1.0):
        """Save results with enhanced directory structure."""
        
        results_df["correct"] = results_df.apply(
            lambda x: str(x["ground_truth"]).strip().lower() == str(x["predicted"]).strip().lower(), axis=1
        )
        
        # Extract model name for directory structure
        model_short = model_name.split("/")[-1] if "/" in model_name else model_name
        
        # Create layer range string for path
        if layer_range:
            layer_str = f"layers_{layer_range[0]}_{layer_range[1]}"
        else:
            layer_str = "all_layers"
        
        # Create structured output directory: results_enhanced/dataset/model/model_part/layer_range/
        output_dir = f"results_enhanced/{dataset_name}/{model_short}/{model_part}/{layer_str}"
        os.makedirs(output_dir, exist_ok=True)
        
        # Create structured filename with scaling values
        base_name = os.path.splitext(os.path.basename(output_path))[0]
        scaling_str = f"num{num_scaling}_op{op_scaling}_ctx{contextual_scaling}"
        structured_filename = f"{base_name}_{scaling_str}.csv"
        output_path = os.path.join(output_dir, structured_filename)
        
        results_df.to_csv(output_path, index=False)
        print(f"Results saved to {output_path}")
        accuracy = results_df["correct"].mean()
        print(f"Overall Accuracy: {accuracy:.4f}")
        
        return accuracy, output_path


def main():
    parser = argparse.ArgumentParser(description="Enhanced Flan-T5 inference experiments")
    parser.add_argument("--model", type=str, default="google/flan-t5-base", help="Model name or path")
    parser.add_argument("--dataset", type=str, default="deepmind_math", help="Dataset name")
    parser.add_argument("--output", type=str, default="inference_results.csv", help="Output CSV filename")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    
    # Enhanced scaling parameters
    parser.add_argument("--num_scaling", type=float, default=1.0, help="Number token scaling")
    parser.add_argument("--op_scaling", type=float, default=1.0, help="Operator token scaling")
    parser.add_argument("--contextual_scaling", type=float, default=1.0, help="Contextual token scaling")
    
    # Model part and layer targeting
    parser.add_argument("--model_part", type=str, choices=["encoder", "decoder", "both"], default="both", 
                        help="Which part of the model to modify")
    parser.add_argument("--layer_start", type=int, default=None, help="Start layer (inclusive)")
    parser.add_argument("--layer_end", type=int, default=None, help="End layer (exclusive)")
    
    # Other parameters
    parser.add_argument("--debug", action="store_true", help="Enable debug/verbose output")
    parser.add_argument("--modification", type=str, default="Solve ", help="Prompt modification")
    
    args = parser.parse_args()

    torch.cuda.empty_cache()

    # Determine layer range
    layer_range = None
    if args.layer_start is not None and args.layer_end is not None:
        layer_range = (args.layer_start, args.layer_end)

    inference = EnhancedFlanT5Inference(model_name=args.model, debug=args.debug, prompt=args.modification)
    df = inference.process_dataset(dataset_name=args.dataset)
    
    results = inference.run_inference(
        df,
        batch_size=args.batch_size,
        num_scaling=args.num_scaling,
        op_scaling=args.op_scaling,
        contextual_scaling=args.contextual_scaling,
        model_part=args.model_part,
        layer_range=layer_range
    )
    
    accuracy, saved_path = inference.save_results(
        results, 
        output_path=args.output, 
        dataset_name=args.dataset, 
        model_name=args.model, 
        model_part=args.model_part,
        layer_range=layer_range,
        num_scaling=args.num_scaling, 
        op_scaling=args.op_scaling,
        contextual_scaling=args.contextual_scaling
    )


if __name__ == "__main__":
    main()

# Example usage commands:

# Baseline (no modification):
# python enhanced_flan_t5_experiments.py --model google/flan-t5-base --num_scaling 1.0 --op_scaling 1.0 --contextual_scaling 1.0

# Conservative scaling (recommended):
# python enhanced_flan_t5_experiments.py --model google/flan-t5-base --num_scaling 1.02 --op_scaling 1.03 --contextual_scaling 1.01

# Early layers only (layers 0-3):
# python enhanced_flan_t5_experiments.py --model google/flan-t5-base --num_scaling 1.02 --op_scaling 1.03 --layer_start 0 --layer_end 3

# Late layers only (for base model with 12 layers, layers 9-12):
# python enhanced_flan_t5_experiments.py --model google/flan-t5-base --num_scaling 1.02 --op_scaling 1.03 --layer_start 9 --layer_end 12

# Encoder only with conservative scaling:
# python enhanced_flan_t5_experiments.py --model google/flan-t5-base --model_part encoder --num_scaling 1.015 --op_scaling 1.025

# Decoder only with different layer ranges:
# python enhanced_flan_t5_experiments.py --model google/flan-t5-base --model_part decoder --layer_start 6 --layer_end 12 --num_scaling 1.01 --op_scaling 1.02