import torch
import pandas as pd
import numpy as np
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_dataset
import re
from tqdm import tqdm
import argparse
from typing import List, Dict, Tuple, Any, Optional


class LatexAwareMathProcessor:
    def __init__(self, model_name="google/flan-t5-base", device=None, debug=False):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.debug = debug
        print(f"Using device: {self.device}")
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
        
        # Store original weights for reset
        self.original_state_dict = {
            k: v.clone() for k, v in self.model.state_dict().items()
            if 'encoder' in k and 'attention' in k and 'relative_attention_bias' not in k
        }
        
        # Initialize latex token detection
        self._initialize_latex_tokens()
        
    def _initialize_latex_tokens(self):
        """Initialize patterns and tokens for LaTeX math expressions"""
        # Common LaTeX commands
        self.latex_commands = {
            'operators': [r'\\cdot', r'\\times', r'\\div', r'\\pm', r'\\mp', r'\\leq', r'\\geq', 
                          r'\\neq', r'\\approx', r'\\equiv', r'\\sum', r'\\prod', r'\\frac'],
            'functions': [r'\\sqrt', r'\\log', r'\\ln', r'\\exp', r'\\sin', r'\\cos', 
                         r'\\tan', r'\\cot', r'\\sec', r'\\csc', r'\\arcsin', r'\\arccos', r'\\arctan'],
            'grouping': [r'\\left', r'\\right', r'\\begin', r'\\end', r'\\{', r'\\}', r'\\[', r'\\]'],
            'special': [r'\\pi', r'\\theta', r'\\alpha', r'\\beta', r'\\gamma', r'\\delta', 
                       r'\\epsilon', r'\\varepsilon', r'\\infty']
        }
        
        # Improved LaTeX patterns with enhanced number detection
        self.latex_patterns = {
            'variables': r'([a-zA-Z](?:_[a-zA-Z0-9]+)?)',  # Match a_1, x, \alpha, etc.
            'subscripted_var': r'([a-zA-Z])_([a-zA-Z0-9]+)',  # Specifically match a_1, x_i, etc.
            'numbers': r'(\d+(?:\.\d*)?(?:[eE][+-]?\d+)?)',  # Enhanced to match 5, 3.14, 1e-3, etc.
            'operators': r'([+\-*/=<>])',  # Basic operators
            'latex_cmd': r'(\\[a-zA-Z]+)',  # LaTeX commands like \frac, \cdot
            'subscripts': r'(_[a-zA-Z0-9]+)'  # Subscripts like _1, _i
        }
        
        # Create token mappings for LaTeX commands
        self.latex_token_ids = {}
        for category, commands in self.latex_commands.items():
            self.latex_token_ids[category] = []
            for cmd in commands:
                # Remove backslashes for tokenizer
                clean_cmd = cmd.replace('\\', '')
                tokens = self.tokenizer.encode(clean_cmd, add_special_tokens=False)
                self.latex_token_ids[category].extend(tokens)
        
        # Regular expression to identify LaTeX math expressions (within $ or $$)
        self.latex_math_pattern = re.compile(r'\$(.+?)\$|\$\$(.+?)\$\$')
        
        # Enhanced patterns for more complex LaTeX parsing
        self.math_delimiter_pattern = re.compile(r'(\$|\$\$)')
        self.variable_with_subscript_pattern = re.compile(r'([a-zA-Z])_([a-zA-Z0-9]+)')
        self.operator_pattern = re.compile(r'(\+|-|\*|/|=|<|>|\\cdot|\\times|\\div|\\leq|\\geq|\\neq)')
        
        # Additional patterns for better number detection
        self.number_pattern = re.compile(r'(\d+(?:\.\d*)?(?:[eE][+-]?\d+)?)')
        
    def reset_weights(self):
        """Reset model weights to original values"""
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in self.original_state_dict:
                    param.copy_(self.original_state_dict[name])
        if self.debug:
            print("Model weights reset to original values")
    
    def _extract_latex_elements(self, text):
        """
        Extract and classify elements in LaTeX math expressions
        Returns dictionary with identified elements
        """
        # Find all LaTeX math expressions in the text
        math_expressions = self.latex_math_pattern.findall(text)
        
        # Flatten the list of tuples from findall
        math_exprs = [expr for tup in math_expressions for expr in tup if expr]
        
        elements = {
            'variables': [],
            'subscripted_vars': [],
            'subscripts': [],
            'numbers': [],
            'operators': [],
            'latex_commands': [],
            'math_regions': []  # Store the span of math expressions
        }
        
        # Store the span of math expressions in the original text
        for match in self.latex_math_pattern.finditer(text):
            elements['math_regions'].append((match.start(), match.end()))
        
        # Process each math expression
        for expr in math_exprs:
            # Find variables with subscripts (a_1, etc.)
            for var_match in re.finditer(self.latex_patterns['subscripted_var'], expr):
                var = var_match.group(0)  # The whole match (a_1)
                base_var = var_match.group(1)  # Just the variable (a)
                subscript = var_match.group(2)  # Just the subscript (1)
                
                if var:
                    elements['subscripted_vars'].append(var)
                else:
                    elements['variables'].append(base_var)
            
            # Also find standalone variables
            for var_match in re.finditer(r'(?<![a-zA-Z\\])([a-zA-Z])(?![a-zA-Z_])', expr):
                if var_match.group(1) not in elements['variables']:
                    elements['variables'].append(var_match.group(1))
            
            # Find numbers with improved pattern
            for num_match in re.finditer(self.number_pattern, expr):
                elements['numbers'].append(num_match.group(1))
            
            # Find operators - improved to catch all operators including +
            for op_match in re.finditer(r'([+\-*/=<>])', expr):
                elements['operators'].append(op_match.group(1))
            
            # Find LaTeX commands
            for cmd_match in re.finditer(self.latex_patterns['latex_cmd'], expr):
                elements['latex_commands'].append(cmd_match.group(1))
        
        return elements
    
    def _tokenize_entities(self, entities: List[str]) -> Dict[str, List[int]]:
        """Tokenizes each entity and returns token ID mapping"""
        mapping = {}
        for ent in entities:
            clean = ent.replace("\\", "")
            token_ids = self.tokenizer.encode(clean, add_special_tokens=False)
            mapping[ent] = token_ids
        return mapping

    def _parse_math_structure(self, text):
        """
        Parse LaTeX math structure with improved handling of variables, numbers, and operators
        """
        # Tokenize the text
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        token_strs = self.tokenizer.convert_ids_to_tokens(tokens)

        token_types = ['regular'] * len(tokens)
        token_scaling = [1.0] * len(tokens)
        
        structure = {
            'math_regions': [],
            'latex_commands': [],
            'equal_signs': [],
            'variable_terms': [],
            'number_terms': [],  # New field to track number positions
            'operator_positions': {},
            'parentheses_pairs': []
        }

        # First, identify math regions in the original text
        math_regions_text = []
        for match in self.latex_math_pattern.finditer(text):
            math_regions_text.append((match.start(), match.end(), match.group(0)))
        
        # Convert text positions to token positions (approximate)
        math_delim_positions = []
        
        # Find all tokens that could be math delimiters ($)
        for i, token in enumerate(token_strs):
            if token in ['▁$', '$', '▁$$', '$$']:
                math_delim_positions.append(i)
        
        # Pair delimiters to create token-based math regions
        if len(math_delim_positions) >= 2:
            for i in range(0, len(math_delim_positions) - 1, 2):
                start = math_delim_positions[i]
                end = math_delim_positions[i + 1] if i + 1 < len(math_delim_positions) else -1
                if end != -1:
                    structure['math_regions'].append((start, end))
        
        # Create a string of the entire token sequence for regex matching
        token_text = ' '.join(token_strs)
        
        # Token classification loop with improved logic
        in_math_region = False
        current_region_start = -1
        current_region_end = -1
        
        for i, token in enumerate(token_strs):
            # Check if we're entering or leaving a math region
            for start, end in structure['math_regions']:
                if i == start:
                    in_math_region = True
                    current_region_start = start
                    current_region_end = end
                    break
                elif i == end + 1:
                    in_math_region = False
                    current_region_start = -1
                    current_region_end = -1
                    break
            
            # Skip space tokens for better pattern matching
            if token == '▁':
                continue
                
            # Handle math delimiters - don't upweight them
            if token in ['▁$', '$', '▁$$', '$$']:
                token_types[i] = 'math_delimiter'
                # Don't upweight the delimiters themselves
                token_scaling[i] = 1.0
                continue
            
            # Special handling inside math regions
            if in_math_region and current_region_start != -1 and current_region_end != -1:
                # We're inside a math region but not on a delimiter
                # Apply a base boost to all tokens within math regions
                token_scaling[i] *= 1.3
                
                # Remove leading space marker from token for cleaner matching
                clean_token = token.strip('▁')
                
                # Check for LaTeX commands
                for category, cmds in self.latex_commands.items():
                    for cmd in cmds:
                        clean_cmd = cmd.replace('\\', '')
                        if clean_token == clean_cmd:
                            token_types[i] = 'latex_command'
                            token_scaling[i] = 2.2
                            structure['latex_commands'].append(i)
                            break
                
                # Check for variables and subscripts
                # Variable pattern: a single letter possibly followed by _ and digits
                if re.match(r'^▁?[a-zA-Z]$', token):
                    # Check if it's followed by a subscript pattern
                    is_subscripted = False
                    if i + 2 < len(token_strs) and token_strs[i+1] == '_' and re.match(r'^\d+$', token_strs[i+2]):
                        # This is part of a subscripted variable like a_1
                        token_types[i] = token_types[i+1] = token_types[i+2] = 'variable'
                        token_scaling[i] = token_scaling[i+1] = token_scaling[i+2] = 1.8
                        structure['variable_terms'].extend([i, i+1, i+2])
                        is_subscripted = True
                    
                    if not is_subscripted:
                        # Single standalone variable
                        token_types[i] = 'variable'
                        token_scaling[i] = 1.8
                        structure['variable_terms'].append(i)
                
                # Enhanced check for numbers with scientific notation
                if re.match(r'^▁?(\d+(\.\d*)?([eE][+-]?\d+)?)$', clean_token):
                    token_types[i] = 'number'
                    token_scaling[i] = 1.8  # Increased from 1.6 to make numbers more prominent
                    structure['number_terms'].append(i)
                
                # Check for operators with enhanced detection
                if clean_token in ['+', '-', '*', '/', '=', '<', '>']:
                    token_types[i] = 'operator'
                    token_scaling[i] = 2.4 if clean_token in ['=', '<', '>'] else 2.0
                    structure['operator_positions'].setdefault(clean_token, []).append(i)
                    if clean_token == '=':
                        structure['equal_signs'].append(i)

        # Additional processing: boost attention for tokens around equals signs and other structures
        for eq_pos in structure['equal_signs']:
            # Boost nearby tokens
            context_radius = 5
            for i in range(max(0, eq_pos - context_radius), min(len(tokens), eq_pos + context_radius + 1)):
                if i != eq_pos:
                    dist_factor = 1 - abs(i - eq_pos) / context_radius
                    token_scaling[i] *= (1 + 0.5 * dist_factor)
        
        # Boost parentheses pairs and their contents
        parens_stack = []
        for i, token in enumerate(token_strs):
            if token in ['(', '\\{', '\\[', '\\left(']:
                parens_stack.append(i)
            elif token in [')', '\\}', '\\]', '\\right)'] and parens_stack:
                start = parens_stack.pop()
                structure['parentheses_pairs'].append((start, i))
                # Boost everything inside the parentheses
                for j in range(start, i+1):
                    token_scaling[j] *= 1.2

        return tokens, token_strs, structure, token_types, token_scaling

    def modify_attention_for_latex_math(self, text, base_scaling=1.5):
        """
        Modify attention weights with special handling for LaTeX math expressions
        """
        self.reset_weights()
        
        # Extract LaTeX elements first for deeper understanding
        latex_elements = self._extract_latex_elements(text)
        
        # Parse math structure with LaTeX awareness
        tokens, token_strs, structure, token_types, token_scaling = self._parse_math_structure(text)
        
        # Enhanced scaling based on LaTeX elements
        
        # 1. Special handling for variables with subscripts (a_1, etc.)
        for subscripted_var in latex_elements['subscripted_vars']:
            # Find token positions that represent this variable
            for i in range(len(token_strs) - 2):
                if token_strs[i].endswith(subscripted_var[0]) and token_strs[i+1] == '_' and token_strs[i+2].startswith(subscripted_var[2:]):
                    # This sequence represents a subscripted variable
                    token_scaling[i] *= 1.4
                    token_scaling[i+1] *= 1.3
                    token_scaling[i+2] *= 1.4
                    
                    # Add to structure if not already there
                    if i not in structure['variable_terms']:
                        structure['variable_terms'].extend([i, i+1, i+2])
        
        # 2. Enhanced scaling for operators
        for op in latex_elements['operators']:
            for i, token in enumerate(token_strs):
                if token.strip('▁') == op:
                    token_scaling[i] *= 1.5
                    # Add to structure if not already there
                    if op not in structure['operator_positions']:
                        structure['operator_positions'][op] = []
                    if i not in structure['operator_positions'][op]:
                        structure['operator_positions'][op].append(i)
        
        # 3. Enhanced scaling for numbers
        for num in latex_elements['numbers']:
            for i, token in enumerate(token_strs):
                clean_token = token.strip('▁')
                # Check if token represents this number
                # Try exact match first
                if clean_token == num:
                    token_scaling[i] *= 1.7
                    if i not in structure['number_terms']:
                        structure['number_terms'].append(i)
                # For multi-token numbers, try to match parts
                elif num.startswith(clean_token) and clean_token.isdigit():
                    token_scaling[i] *= 1.5
                    if i not in structure['number_terms']:
                        structure['number_terms'].append(i)
        
        # 4. Special handling for LaTeX commands
        for cmd in latex_elements['latex_commands']:
            clean_cmd = cmd.replace('\\', '')
            for i, token in enumerate(token_strs):
                if token.strip('▁') == clean_cmd:
                    token_scaling[i] *= 1.8
                    # Add to structure if not already there
                    if i not in structure['latex_commands']:
                        structure['latex_commands'].append(i)
                    
                    # Boost tokens that follow a command (likely its arguments)
                    for j in range(i+1, min(len(tokens), i+5)):
                        token_scaling[j] *= 1.3
        
        # Apply the calculated scaling factors to model parameters
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if ('encoder' in name and 'attention' in name and 
                    'relative_attention_bias' not in name and
                    ('q' in name.lower() or 'k' in name.lower())):
                    
                    modified_param = param.clone()
                    
                    # Apply the scaling to the token dimensions
                    # Make sure we're within dimension bounds
                    for token_id, scale in enumerate(token_scaling):
                        if token_id < param.size(1):  # Check dimension bounds
                            modified_param[:, token_id] *= scale * base_scaling
                    
                    # Copy the modified parameters back
                    param.copy_(modified_param)
        
        # Debug information
        if self.debug:
            self._print_latex_analysis(tokens, token_strs, structure, token_types, token_scaling, latex_elements)
    
    def _print_latex_analysis(self, tokens, token_strs, structure, token_types, token_scaling, latex_elements=None):
        """Print debug information about the LaTeX math parsing"""
        print("\n" + "="*70)
        print("LATEX MATH STRUCTURE ANALYSIS")
        print("="*70)
        
        # Print the tokens and their scaling
        print("\nTOKENS AND SCALING:")
        print("-" * 50)
        print(f"{'Index':<7}{'Token':<15}{'Token ID':<10}{'Scaling':<10}{'Type':<15}")
        print("-" * 50)
        
        for i, (token, token_id, token_type, scale) in enumerate(
                zip(token_strs, tokens, token_types, token_scaling)):
            print(f"{i:<7}{token:<15}{token_id:<10}{scale:<10.2f}{token_type:<15}")
        
        # Print extracted LaTeX elements if available
        if latex_elements:
            print("\nEXTRACTED LATEX ELEMENTS:")
            for category, items in latex_elements.items():
                if category != 'math_regions':  # Skip the spans for cleaner output
                    print(f"- {category}: {items}")
        
        # Print structural information
        print("\nSTRUCTURAL ELEMENTS:")
        print(f"- Math regions: {structure['math_regions']}")
        print(f"- LaTeX commands: {structure['latex_commands']}")
        print(f"- Equal signs: {structure['equal_signs']}")
        print(f"- Parentheses pairs: {structure['parentheses_pairs']}")
        print(f"- Variable terms: {structure['variable_terms']}")
        print(f"- Number terms: {structure['number_terms']}")  # Added number terms tracking
        
        # Print operator positions
        print("\nOPERATORS:")
        for op, positions in structure['operator_positions'].items():
            if positions:
                print(f"- '{op}': {positions}")
        
        print("="*70)

    def prepare_prompt(self, question):
        """Prepare the prompt for the model"""
        return f"Please solve the following problem and put your answer at the end with \"The answer is: \". {question}"

    def extract_answer(self, text):
        """Extract the final answer from the model output"""
        text = text.strip()
        if "The answer is:" in text:
            return text.split("The answer is:")[-1].strip()
        return text
    
    def process_dataset(self, dataset_name="Rith335/mmiqc_filtered_math_equation_problems", split="test"):
        """Process the dataset and return a DataFrame"""
        dataset = load_dataset(dataset_name)
        
        # Extract the relevant split
        split_data = dataset[split]
        
        # Process instructions and outputs
        questions = []
        answers = []
        
        for item in split_data:
            # Clean instruction by removing prompt prefixes
            instruction = item['instruction']
            prefix = 'Please solve the following problem and put your answer at the end with "The answer is: ".'
            if instruction.startswith(prefix):
                question = instruction[len(prefix):].strip()
            else:
                question = instruction
            
            # Extract final answer from output
            output = item['output']
            final_answer = ""
            lines = output.split('\n')
            for line in reversed(lines):
                if "The answer is:" in line:
                    final_answer = line.split("The answer is:")[-1].strip()
                    break
            
            questions.append(question)
            answers.append(final_answer)
        
        return pd.DataFrame({"question": questions, "answer": answers})
    
    def run_inference(self, df, batch_size=8, use_latex_awareness=True, base_scaling=1.5):
        """
        Run inference on the dataset with LaTeX-aware attention modification
        """
        results = []
        total_examples = len(df)
        
        for i in tqdm(range(0, total_examples, batch_size)):
            batch = df.iloc[i:i+batch_size]
            prompts = [self.prepare_prompt(q) for q in batch["question"]]
            batch_results = []
            
            if self.debug:
                print(f"\nProcessing batch {i//batch_size + 1}/{(total_examples+batch_size-1)//batch_size}")
            
            for j, (idx, row) in enumerate(batch.iterrows()):
                question = row["question"]
                if self.debug:
                    print(f"\nProcessing question {idx} ({j+1}/{len(batch)}):")
                    print(f"QUESTION: {question}")
                
                # Apply LaTeX-aware attention modification if enabled
                if use_latex_awareness:
                    self.modify_attention_for_latex_math(question, base_scaling)
                
                # Generate answer
                single_input = self.tokenizer([prompts[j]], return_tensors="pt", padding=True, truncation=True).to(self.device)
                with torch.no_grad():
                    output = self.model.generate(
                        input_ids=single_input.input_ids,
                        attention_mask=single_input.attention_mask,
                        max_length=150,
                        num_beams=4,
                        early_stopping=True
                    )
                prediction = self.tokenizer.decode(output[0], skip_special_tokens=True)
                final_answer = self.extract_answer(prediction)
                
                if self.debug:
                    print(f"Ground truth: {row['answer']}")
                    print(f"Predicted: {final_answer}")
                    print(f"Correct: {row['answer'].strip().lower() == final_answer.strip().lower()}")
                
                batch_results.append({
                    "idx": idx,
                    "prompt": prompts[j],
                    "ground_truth": row["answer"],
                    "predicted": final_answer,
                    "model_response": prediction,
                })
                
                # Reset weights for next question
                self.reset_weights()
            
            results.extend(batch_results)
                
        return pd.DataFrame(results)
    
    def save_results(self, results_df, output_path="latex_math_results.csv"):
        """Save results and calculate accuracy metrics"""
        results_df["correct"] = results_df.apply(
            lambda x: x["ground_truth"].strip().lower() == x["predicted"].strip().lower(), axis=1
        )
        results_df.to_csv(output_path, index=False)
        print(f"Results saved to {output_path}")
        
        # Calculate accuracy
        accuracy = results_df["correct"].mean()
        print(f"Overall Accuracy: {accuracy:.4f}")
        
        # Print incorrect examples for analysis
        if self.debug:
            incorrect_df = results_df[results_df["correct"] == False]
            print("\nSample of incorrect predictions:")
            for i, (_, row) in enumerate(incorrect_df.head(3).iterrows()):
                print(f"\nExample {i+1}:")
                print(f"Question: {row['prompt'][:100]}...")
                print(f"Ground truth: {row['ground_truth']}")
                print(f"Predicted: {row['predicted']}")


def main():
    parser = argparse.ArgumentParser(description="Run LaTeX-aware inference on Math QA datasets")
    parser.add_argument("--model", type=str, default="google/flan-t5-base", help="Model name or path")
    parser.add_argument("--dataset", type=str, default="Rith335/mmiqc_filtered_math_equation_problems", 
                        help="Dataset name on Hugging Face")
    parser.add_argument("--output", type=str, default="results/latex_math_results.csv", 
                        help="Output CSV path")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--no_latex", action="store_true", help="Disable LaTeX-aware attention")
    parser.add_argument("--base_scaling", type=float, default=1.5, help="Base scaling factor")
    parser.add_argument("--debug", action="store_true", help="Enable debug/verbose output")
    args = parser.parse_args()
    
    processor = LatexAwareMathProcessor(model_name=args.model, debug=args.debug)
    df = processor.process_dataset(dataset_name=args.dataset)
    
    results = processor.run_inference(
        df,
        batch_size=args.batch_size,
        use_latex_awareness=not args.no_latex,
        base_scaling=args.base_scaling
    )
    
    processor.save_results(results, output_path=args.output)


if __name__ == "__main__":
    main()