# VALIDATED TASKS
- ifeval
- gsm8k
- hellaswag_gen
- drop_new
- truthfulqa_gen
- mmlu_generative
- bbh_zeroshot
- bbh_fewshot
- swde

-----------------------------------------

# Update tasks - gsm8k, bbh_fewshot, swde: 20-8-2025
1. revert gsm8k doc_to_text prompt to original
2. update doc_to_text prompt for bbh_fewshot and include subset as bbh_fewshot_subset
3. update doc_to_text prompt for swde

# Initial commit: 19-8-2025
1. add model for openvino_genai
2. huggingface
3. extract thinking output for ifeval task
4. update doc_to_text prompt for mmlu_generative
5. create subset for bbh_zeroshot as bbh_zeroshot_subset
6. convert hellaswag to generate_until task as hellaswag_gen
7. update metrics for truthfulqa
8. fix process_function & doc_to_text & doc_to_target & generation_kwargs for drop & include 3 num_shots as drop_new 