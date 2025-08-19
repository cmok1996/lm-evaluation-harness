# VALIDATED TASKS
- ifeval
- gsm8k
- hellaswag_gen
- drop_new
- truthfulqa_gen
- mmlu_generative
- bbh_zeroshot

-----------------------------------------

# Initial commit: 19-8-2025
1. add model for openvino_genai
2. huggingface
3. extract thinking output for ifeval task
4. update doc_to_text prompt for mmlu_generative
5. create subset for bbh_zeroshot as bbh_zeroshot_subset
6. convert hellaswag to generate_until task as hellaswag_gen
7. update metrics for truthfulqa
8. fix process_function & doc_to_text & doc_to_target & generation_kwargs for drop & include 3 num_shots as drop_new 