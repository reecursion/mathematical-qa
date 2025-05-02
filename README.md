# Modifying Attention for Mathematical Reasoning in Language Models

## 10623 Generative AI Final Project

##### Gayathri Ganesh Lakshmy, Rithvik Senthil, Rupsa Dhar

Large Language Models (LLMs) often struggle with complex mathematical reasoning due to inadequate handling of mathematical operators. This project explores modifying the attention mechanism to scale focus on operators and numbers, aiming to better capture mathematical structure. We evaluate this approach using Pass@1 and BERTScore on the MMIQC dataset.

## To run inference:

#### Baseline (set --baseline to True)

python src/flan_t5_attention_mod.py --model google/flan-t5-xl --output <OUTPUT_FILE_PATH> --num_scaling 1 --op_scaling 1 --modification "Please solve the following problem and only output the answer at the end with \"The answer is: \". " --baseline True

#### Attention Modification

##### Without explanation prompting:

python src/flan_t5_attention_mod.py --model google/flan-t5-xl --output <OUTPUT_FILE_PATH> --num_scaling 1 --op_scaling 0.7 --modification "Please solve the following problem and only output the answer at the end with \"The answer is: \". " --model_part <ENCODER/DECODER/BOTH>

##### With explanation (chain-of-thought) prompting:

python src/flan_t5_attention_mod.py --model google/flan-t5-xl --output <OUTPUT_FILE_PATH> --num_scaling 1 --op_scaling 0.7 --modification "Please solve the following problem and give a concise explanation with the answer at the end with \"The answer is: \"." --model_part <ENCODER/DECODER/BOTH>

## Evaluation

python utils/evaluate_using_llama.py --file <EVAL_OUTPUT_PATH>
