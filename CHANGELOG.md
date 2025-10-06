# VALIDATED TASKS
- ifeval
- gsm8k
- hellaswag_gen
- drop_new
- truthfulqa_gen
- mmlu_generative
- mmlu_stem_gen
- mmlu_social_sciences_gen
- mmlu_humanities_gen
- mmlu_others_gen
- bbh_zeroshot
- bbh_zeroshot_subset
- bbh_fewshot
- bbh_fewshot_subset
- swde
- humaneval
- winogrande_gen
- wmt24
- wmt24_subset
- mmlu_generative_orig
- mmlu_stem_gen_orig
- mmlu_social_sciences_gen_orig
- mmlu_humanities_gen_orig
- mmlu_others_gen_orig
- bbh_fewshot_orig
- bbh_fewshot_subset_orig
- humaneval_orig
- swde_orig
- mgsm_direct_zh


-----------------------------------------
# update get_usecase_score api 6-10-2025
1. update get_usecase_score to be more flexible and return leaderboard pivot table directly

# leaderboard aggregate scores by use-cases 6-10-2025
1. separate routers into tasks and leaderboard
2. include leaderboard api
3. use case config
4. include mgsm_direct_zh as supported task

# generation kwargs for benchmarks: 2-10-2025
1. modify oepnai completion default generation kwargs for stop and max_gen_toks in create_payload
2. modify stop and max_gen_toks in bbh to allow for longer responses

# leaderboard UI with gradio: 2-10-2025
1. include gradio UI
2. include /leadeboard api
3. changes to endpoint prefix
4. fix bbh_fewshot_subset metric

# enhance api functionality: 1-10-2025
1. include /get_samples_path, /get_results_path
2. modify /results to take output_path as argument instead of results.json file
3. modify path structure

# update tasks config and include original tasks: 30-9-2025
1. modify mmlu subsets to include aggregate score
2. modify doc_to_text for mmlu to limit the choices responses
3. modify bbh filter to extract desired regex
4. modify humaneval doc_to_text and metric function to extract code
5. fix get_sample_size to only limit 1 sample if limit = 1
6. include original tasks - mmlu, bbh, humaneval, swde
7. fix truthfqulqa_gen metric to bleu_acc,none

# wrap task_stats functions into fastapi: 30-9-2025
1. wrap task_stats functions into fastapi
2. get overall and per-sample results from file
3. update requirements
4. fix /api/task/num_samples to include limit argument

# minor fixes: 15-9-2025
1. modify ifeval metrics in task config
2. remove eval_results directory
3. disable removal of thinking tags for ifeval as it will be handled in mep

# support wmt24 and wmt24_subset, reverse translation task to input foreign source and output english: 29-8-2025
1. support wmt24 and wmt24_subset, reverse translation task to input foreign source and output english: 29-8-2025
2. subset includes "en-ar_EG", "en-zh_CN", "en-de_DE", "en-es_MX", "en-fr_CA", "en-it_IT", "en-ja_JP", "en-ta_IN", "en-th_TH", "en-vi_VN"

# update get_sample_size code for limit: 28-8-2025
1. add more fields for task_config
2. get sample size from config instead of load_dataset

# update get_sample_size code for limit: 27-8-2025
1. treat limit=1.0 as 100% while limit=1 as 1 sample

# include metadata for supported tasks and script for task statistics: 26-8-2025
1. task_config.py to provide metadata for supported tasks
2. task_stats.py to generate count statistics for supported tasks

# support generate_until for winogrande: 26-8-2025
1. support winogrande to generate_until task as winogrande_gen

# Update humaneval: 25-8-2025
1. revert gsm8k doc_to_text prompt to original
2. update doc_to_text prompt for bbh_fewshot and include subset as bbh_fewshot_subset
3. update doc_to_text prompt for swde

# Update humaneval: 25-8-2025
1. revert gsm8k doc_to_text prompt to original
2. update doc_to_text prompt for bbh_fewshot and include subset as bbh_fewshot_subset
3. update doc_to_text prompt for swde

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