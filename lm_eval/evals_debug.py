from lm_eval import evaluator
import os
import json
from datetime import datetime
import numpy as np
import torch
import time
from abc import ABCMeta
import logging
import sys

logging.basicConfig(
    filename = 'eval_results/eval.log',
    level = logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)

logger = logging.getLogger(__name__)
logger.addHandler(handler)

TASKS = ['humaneval'] #['mgsm_direct_zh'] #['ifeval', 'gsm8k', 'bbh_zeroshot_subset', 'mmlu_generative',   'truthfulqa_gen', 'mutual'] #['ifeval'] #['mmlu_generative', 'bbh_zeroshot_subset', 'ifeval', 'gsm8k', 'race', 'truthfulqa_gen', 'mutual'] # #mutual 'race',  ['longbench_hotpotqa'] #['ifeval', 'mmlu_generative'] #
MODEL_BACKEND = 'local-completions' #hf #'openvino_genai' #'openvino_genai' # 'builtin_gguf' #'builtin_gguf' #'gguf' # #'local-completions' #
# MODELS = ['intel_sealion_v2.1_ipex_0307']
MODELS = [
    "gpt-oss:20b"
    # "qwen3:latest",
    # "Qwen/Qwen3-8B",
    # "meta-llama/Llama-3.1-8B-Instruct",
    # "meta-llama/Meta-Llama-3-8B-Instruct"
    # "mistralai/Mistral-7B-Instruct-v0.3",
    # "mistralai/Ministral-8B-Instruct-2410",
    # "microsoft/Phi-4-mini-instruct",
    # "microsoft/Phi-3-medium-4k-instruct",
    # "devstral:24b-small-2505-q4_K_M",
    # "ministral/Ministral-3b-instruct"

    # 'aisingapore/llama3-8b-cpt-sea-lionv2.1-instruct', 
        #   'microsoft/Phi-3.5-mini-instruct', 
        #   'qwen/qwen2.5-7B-Instruct', 

        #   'Qwen/Qwen2.5-7B-Instruct', 
        # 'aisingapore/llama3-8b-cpt-sea-lionv2.1-instruct'

        #   'qwen/qwen2-7B-Instruct',
        #   'qwen/qwen2.5-0.5B-Instruct', 
        #   'qwen/qwen2.5-1.5B-Instruct', 
        #   'qwen/qwen2.5-3B-Instruct', 
        #   'qwen/qwen2.5-7B-Instruct', 
        #   'qwen/qwen2.5-14B-Instruct', 
        #   'qwen/qwen2.5-7B-Instruct', 
        #   'qwen/qwen2-7B-Instruct', #/home/aicoe/lm-evaluation-harness/llm_inference/Phi-3.5-mini-instruct-gptqmodel-4bit
        # 'aisingapore/llama3-8b-cpt-sea-lionv2.1-instruct:Q4_K_M',
        # 'Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4',
        # 'Qwen/Qwen2-7B-Instruct-GPTQ-Int4',
        # 'llm_inference/Phi-3.5-mini-instruct-gptqmodel-4bit',
        # 'llm_inference/llama3-8b-cpt-sea-lionv2.1-instruct-gptqmodel-4bit',
        # 'llm_inference/qwen2.5-7b-instruct-q4_k_m/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf',
        # 'llm_inference/qwen2-7b-instruct-q4_k_m.gguf',
        # 'llm_inference/llama3-8b-cpt-sea-lionv2.1-instruct-Q4_K_M.gguf',
        # 'llm_inference/Phi-3.5-mini-instruct-Q4_K_M.gguf',
        # 'deepseek-r1:1.5b',
        # 'deepseek-r1:7b',
        # 'deepseek-r1:8b',
        # 'deepseek-r1:14b',
        # 'deepseek-r1:32b',
        # 'hf.co/unsloth/DeepSeek-R1-Distill-Qwen-1.5B-GGUF:Q8_0',
        # 'hf.co/unsloth/DeepSeek-R1-Distill-Qwen-7B-GGUF:Q8_0',
        # 'hf.co/unsloth/DeepSeek-R1-Distill-llama-8B-GGUF:Q8_0',
        # 'hf.co/unsloth/DeepSeek-R1-Distill-Qwen-14B-GGUF:Q8_0'
        # 'llama3.1:8b-instruct-q4_K_M',
        # 'qwen2.5:7b-instruct-q4_K_M',
        # 'phi3.5:3.8b-mini-instruct-q4_K_M',
        # 'mistral:7b-instruct-q4_K_M',
        # 'hf.co/mradermacher/Ministral-3b-instruct-GGUF:q4_k_m',
        # 'hf.co/unsloth/Phi-4-mini-instruct-GGUF:q4_k_m'
        # 'qwen2.5:1.5b-instruct-q4_K_M',
        # 'llama3.1:latest',
        # 'qwen2:7b-instruct-q4_K_M'
          ]

LIMIT = 10
DEVICE = 'cuda:0' #'cuda:0' #'cpu'
BATCH_SIZE = 'auto'
LOG_SAMPLES = True
NUM_FEWSHOT = None
CONFIRM_RUN_UNSAFE_CODE = False
APPLY_CHAT_TEMPLATE = True
SYSTEM_INSTRUCTION = None

start_time = datetime.now()
start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

os.makedirs("eval_results", exist_ok=True)

def convert_types(obj):
    if isinstance(obj, ABCMeta):  # Handle abstract base classes
        return str(obj)
    elif isinstance(obj, np.generic):  # Convert NumPy types
        return obj.item()
    elif callable(obj):  # Filters out functions
        return str(obj)
    elif isinstance(obj, torch.dtype):  # Convert torch.dtype to string
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

for TASK in TASKS:
    for MODEL in MODELS:
        logger.info(f'Task - {TASK}, Model - {MODEL}, Model Backend - {MODEL_BACKEND}, Limit - {LIMIT}, Batch size - {BATCH_SIZE}')
        task_start_time = time.time()
        if TASK in ('gsm8k', 'ifeval', 'bbh_zeroshot_sample', 'mutual'): #and MODEL_BACKEND == 'hf':
            APPLY_CHAT_TEMPLATE =  True #True
        elif TASK in ('humaneval', 'humaneval_instruct'):
            os.environ["HF_ALLOW_CODE_EVAL"] = "1"
            CONFIRM_RUN_UNSAFE_CODE = True
        elif TASK == 'mmlu_generative':
            NUM_FEWSHOT = 5
            APPLY_CHAT_TEMPLATE = False
        else:
            APPLY_CHAT_TEMPLATE =  True
        OUTPUT_PATH = f'./eval_results/{TASK}_eval'

        if MODEL in ('Qwen/Qwen3-8B', "qwen3:latest"):
            SYSTEM_INSTRUCTION = "You are a helpful assistant. /no_think."

        if MODEL_BACKEND == 'vllm':
            model_args = {'pretrained': MODEL,
                        'dtype': 'half',
                        'gpu_memory_utilization': 0.7,
                        'max_num_seqs': 1,
                        'max_model_len': 4096}
        elif MODEL_BACKEND in ('hf', 'hf-chat'):
            model_args = {'pretrained' : MODEL}
            if TASK  in ('humaneval', 'humaneval-instruct'):
                model_args['parallelize'] = False
        elif MODEL_BACKEND == 'local-completions':
             APPLY_CHAT_TEMPLATE = False
             model_args = {
            'pretrained' : MODEL, 
                        'base_url':  'http://localhost:11434/v1/completions', #'http://localhost:8000' , #'http://localhost:11434/v1/completions',# #'http://localhost:11434/v1/completions', #'http://localhost:8000', #'http://10.121.177.99:8000/completions', #'http://localhost:11434/v1/completions', #'http://localhost:11434/v1/completions', 
                        # 'tokenizer': MODEL,
                        'tokenized_requests': False,
                        'tokenizer_backend': None #'huggingface' #'huggingface'
                        }
        elif MODEL_BACKEND == 'local-chat-completions':
             APPLY_CHAT_TEMPLATE = True
             model_args = {
            'pretrained' : MODEL, 
                        'base_url':  'http://localhost:11434/v1/chat/completions', #'http://localhost:8000' , #'http://localhost:11434/v1/completions',# #'http://localhost:11434/v1/completions', #'http://localhost:8000', #'http://10.121.177.99:8000/completions', #'http://localhost:11434/v1/completions', #'http://localhost:11434/v1/completions', 
                        # 'tokenizer': MODEL,
                        'tokenized_requests': False,
                        'tokenizer_backend': None #'huggingface' #'huggingface'
                        }
        elif MODEL_BACKEND == 'gguf':
             model_args = {
            'pretrained' : MODEL, 
                        'base_url': 'http://localhost:8000',# ' #'http://localhost:11434/v1/completions', #'http://localhost:8000', #'http://10.121.177.99:8000/completions', #'http://localhost:11434/v1/completions', #'http://localhost:11434/v1/completions', 
                        }
        elif MODEL_BACKEND == 'builtin_gguf':
            model_args = {'model_path': f"/home/aicoe/lm-evaluation-harness/{MODEL}",
                        'n_gpu_layers' : 50}
            APPLY_CHAT_TEMPLATE =  False #model will internally call create_chat_completions for generate tasks
        elif MODEL_BACKEND == 'openvino':
            model_args = {'pretrained': MODEL,
                        #   'bit':4
                        }
            DEVICE = 'cpu'
        elif MODEL_BACKEND == 'openvino_genai':
            model_args = {'model_path': f'/home/aicoe/lm-evaluation-harness/llm_inference/models_ov/{MODEL}-4bit-sym-ov',
                          'model_id': MODEL}
            DEVICE = 'cpu'
            APPLY_CHAT_TEMPLATE= False

        results = evaluator.simple_evaluate(
                model = MODEL_BACKEND,
                model_args =model_args,
                tasks = TASK,
                limit = LIMIT,
                device = DEVICE,  
                batch_size =BATCH_SIZE,
                log_samples = LOG_SAMPLES,
                apply_chat_template = APPLY_CHAT_TEMPLATE,
                num_fewshot = NUM_FEWSHOT,
                confirm_run_unsafe_code=CONFIRM_RUN_UNSAFE_CODE,
                # fewshot_as_multiturn=True
                # model=args.model,
                # model_args=args.model_args,
                # tasks=task_names,
                # num_fewshot=args.num_fewshot,
                # batch_size=args.batch_size,
                # max_batch_size=args.max_batch_size,
                # device=args.device,
                # use_cache=args.use_cache,
                # limit=args.limit,
                # check_integrity=args.check_integrity,
                # write_out=args.write_out,
                # log_samples=args.log_samples,
                # evaluation_tracker=evaluation_tracker,
                # system_instruction=args.system_instruction,
                # apply_chat_template=args.apply_chat_template,
                # fewshot_as_multiturn=args.fewshot_as_multiturn,
                # gen_kwargs=args.gen_kwargs,
                # task_manager=task_manager,
                # predict_only=args.predict_only,
                # random_seed=args.seed[0],
                # numpy_random_seed=args.seed[1],
                # torch_random_seed=args.seed[2],
                # fewshot_random_seed=args.seed[3],
                # confirm_run_unsafe_code=args.confirm_run_unsafe_code,
                # **request_caching_args,
            )
        task_end_time = time.time()
        task_time_taken = task_end_time - task_start_time
        MODEL_PATH = os.path.join(OUTPUT_PATH, MODEL)
        results['model_backend'] = MODEL_BACKEND
        results['start_time'] = task_start_time 
        results['end_time'] = task_end_time 
        results['total_evaluation_time_seconds'] = task_time_taken
        if not os.path.exists(MODEL_PATH):
            os.makedirs(MODEL_PATH)  # Creates folder if it doesn't exist
        
        if 'samples' in results.keys():
            samples = results.pop('samples')
            samples_file = f'samples_{TASK}_{start_time}.json'
            SAMPLES_PATH = os.path.join(MODEL_PATH, samples_file)
            with open (SAMPLES_PATH, "w") as json_file:
                json.dump(samples, json_file, indent=4, default = convert_types)

        results_file =  f'results_{TASK}_{start_time}.json'
        RESULTS_PATH = os.path.join(MODEL_PATH, results_file)
        with open(RESULTS_PATH, "w") as json_file:
            json.dump(results, json_file, indent=4, default=convert_types)


