from datasets import load_dataset, get_dataset_config_names, concatenate_datasets
from task_config import TASK_CONFIG
from typing import Optional, Union
import math
import logging
import re
import os
from datetime import datetime
import json
from pydantic import BaseModel, Field
from typing import List, Optional
import pandas as pd
from pymongo import MongoClient
from dotenv import load_dotenv

logging.basicConfig(
    level=logging.INFO,                     # minimum log level
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

load_dotenv()

MONGODB_URL = os.getenv("MONGODB_URL")

def get_supported_tasks():
     supported_tasks = list(TASK_CONFIG.keys())
     return supported_tasks

def connect_mongodb():
    print(MONGODB_URL)
    client = MongoClient(MONGODB_URL)  # update your URI if needed
    return client

def get_clean_model_name_from_mongodb(client, model_name):
    db = client["mep"]
    collection = db["task"]

    query = {"task_id": model_name}
    result = list(collection.find(query))
    if result and 'model_card_name' in result:
        return result['model_card_name']
    else:
        return model_name  # return original name if no mapping found

def _load_dataset(tasks):
    datasets = dict()
    if isinstance(tasks, str):
        tasks = [tasks]
    supported_tasks = get_supported_tasks()
    for task in tasks:
        logger.info(f'Processing task - {task}')
        if task not in supported_tasks:
            raise ValueError(f'{task} not supported, must be one of the following - {supported_tasks}')
        task_config = TASK_CONFIG[task]

        if task_config['dataset_name'] is None:
            dataset_names = get_dataset_config_names(task_config['dataset_path'])
            dataset_names = [n for n in dataset_names if n not in ('default', 'all')] if len(dataset_names) > 1 else dataset_names #remove default for benchmarks with multiple subsets, eg bbh
            dataset_len = len(dataset_names)
            logger.info(f'Dataset name is not provided. Task {task} has {dataset_len} subsets')
        else:
            dataset_names = task_config['dataset_name']
            dataset_len = len(dataset_names)
            if isinstance(dataset_names, str):
                dataset_names = [dataset_names]
            logger.info(f'Task {task} has {dataset_len} subsets')

        if len(dataset_names) == 1:
            dataset_name = dataset_names[-1]
            logger.info(f'Loading {dataset_name}')
            dataset = load_dataset(path = task_config['dataset_path'], name = dataset_name, split = task_config['split'])
            datasets[task] = dataset

        else:
            
            task_dataset = []
            for dataset_name in dataset_names:
                logger.info(f'Loading {dataset_name}')
                ds = load_dataset(path = task_config['dataset_path'], name = dataset_name, split = task_config['split'])
                ds = ds.add_column(task, [dataset_name]*len(ds))
                task_dataset.append(ds)
            # task_combined = concatenate_datasets(task_dataset)
            # logger.info(f'Finished combining all subsets into a single task - {task}')
            datasets[task] = task_dataset

    return datasets

def get_sample_size(original_size, limit: Optional[int]) -> Union[int, None]:
    
    if limit is not None:
        cnt = (
            int(math.ceil(original_size * limit)) if ((limit <= 1.0) and (isinstance(limit, float))) else int(limit) #1.0 is treated as 100% of samples, while 1 is treated as 1 sample
        )
    else:
        cnt = original_size
    cnt = min(cnt, original_size)
    cnt = max(0, cnt)
    return cnt

def get_task_num_samples(tasks: Union[str, list], limits: Union[int, float, list, None]):
    try:
        if isinstance(tasks, str):
            tasks = [tasks]
        datasets = _load_dataset(tasks)
        task_config = dict()

        if isinstance(limits, list):
            assert len(tasks) == len(limits), f"Error: Number of tasks ({len(tasks)}) does not match number of limits ({len(limits)})"
        if not isinstance(limits, list):
            limits = [limits for i in range(len(tasks))]

        for (task_name, dataset), limit in zip(datasets.items(), limits):
            if isinstance(dataset, list):
                original_size = 0
                effective_size = 0
                for ds in dataset:
                    original_size += ds.num_rows
                    effective_size += get_sample_size(ds.num_rows, limit)    
            else:
                original_size = dataset.num_rows
                effective_size = get_sample_size(original_size, limit)    
            
            task_config[task_name] = {'num_samples': original_size,
                                    'effective': effective_size}
    
    except Exception as e:
        logger.error(f'Error in processing dataset - {e}')
        raise e
    
    return task_config

def get_task_num_samples_from_config(tasks, limit = None):
    task_stats = dict()
    for task in tasks:
        supported_tasks = get_supported_tasks()
        if task not in supported_tasks:
            raise ValueError(f'{task} not supported, must be one of the following - {supported_tasks}')
        task_config = TASK_CONFIG[task]
        original_size = task_config['n_samples']
        effective_size = get_sample_size(original_size, limit)
        task_stats[task] = {'num_samples': original_size,
                'effective': effective_size}

    return task_stats

def get_accuracy_results_score(task, output_path, model):
    results_paths =  get_results_path_by_task(output_path, model)
    results_paths = [s[-1] for s in results_paths]
    scores = []
    for results_path in results_paths:

        with open(results_path, 'r', encoding='utf-8') as f:
            # data = [json.loads(line) for line in f]
            data = json.load(f)

        assert task in TASK_CONFIG.keys(), f'Task {task} not found in TASK_CONFIG'
        
        metric = TASK_CONFIG[task]['metrics']
        if task in data['results']:
            accuracy = data['results'][task][metric]
            num_samples = sum(d["effective"] for d in data['n-samples'].values())
        elif task in data['groups']:
            accuracy = data['groups'][task][metric]
            num_samples = sum(d["effective"] for d in data['n-samples'].values())
        else:
            raise ValueError(f'Task {task} not found in results or groups in {results_path}')
        
        score = {'results_path': results_path,
                'accuracy': accuracy,
                'num_samples': num_samples,
                'metric': metric}
        scores.append(score)
        
    return scores

def get_results_path_by_task(task_dir, model):
    pattern = re.compile(r"^results_(\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}\.\d+)\.json$")

    path_dir = os.path.join(task_dir,  model)
    matched_files = []
    for filename in os.listdir(path_dir):
        
        match = pattern.match(filename)
        if match:
            timestamp_str = match.group(1)
            # Convert to datetime object
            timestamp = datetime.strptime(timestamp_str, "%Y-%m-%dT%H-%M-%S.%f")
            file_path = os.path.join(path_dir, filename)
            matched_files.append((filename, timestamp, timestamp_str, file_path))

    #sort by timestamp descending
    matched_files.sort(key=lambda x: x[1], reverse=True)

    # latest_timestamp = matched_files[0][1] if matched_files else None

    # Get the latest file
    # if matched_files:
    #     latest_file = matched_files[0][0]
    #     latest_file_path = os.path.join(path_dir, latest_file)
    #     print("Latest results file:", latest_file)
    # else:
    #     print("No matching results files found.")

    return matched_files #latest_file_path, matched_files

def get_sample_path_by_task(task_dir, model):
    pattern = re.compile(r"^samples_(.*)_(\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}\.\d+)\.jsonl?$")

    path_dir = os.path.join(task_dir,  model)
    matched_files = []
    for filename in os.listdir(path_dir):
        
        match = pattern.match(filename)
        if match:
            subtask = match.group(1)
            timestamp_str = match.group(2)
            # Convert to datetime object
            timestamp = datetime.strptime(timestamp_str, "%Y-%m-%dT%H-%M-%S.%f")
            file_path = os.path.join(path_dir, filename)
            matched_files.append((filename, subtask, timestamp, timestamp_str, file_path))

    #sort by timestamp descending
    matched_files.sort(key=lambda x: x[2], reverse=True)

    # latest_timestamp = matched_files[0][2] if matched_files else None

    # Get the latest file
    # if matched_files:
    #     latest_file = matched_files[0][0]
    #     latest_file_path = os.path.join(path_dir, latest_file)
    #     print("Latest sample file:", latest_file)
    # else:
    #     print("No matching sample files found.")

    return matched_files #latest_file_path, matched_files

class SampleResponse(BaseModel):
    model: Optional[str] = Field(default=None, description="Model name")
    task: Optional[str] = Field(default=None, description="Task name")
    subtask: Optional[str] = Field(default=None, description="Subtask Dataset name")
    prompt_idx: Optional[int] = Field(default=None, description="Benchmark prompt index")
    prompt: Optional[str] = Field(default=None, description="Task name")
    # full_prompt: str
    response: str
    filtered_response: str
    gold: Union[str, List[str], int, List[int], None]
    metric: Optional[str] = None
    is_correct: Optional[bool] = Field(default=None, description="accuracy or correctness of the response")
    timestamp: Optional[str] =  Field(default=None, description="Timestamp string when the sample was generated")

def get_response_df(TASK, sample_path, model, timestamp_str, subtask = None):
    # sample_path = get_latest_sample_path(eval_dir, model)

    with open(sample_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]

    def standardize_is_correct(value):
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return value == 1
        if isinstance(value, str):
            value_lower = value.strip().lower()
            if value_lower in ['1', 'true', 'yes']:
                return True
            elif value_lower in ['0', 'false', 'no']:
                return False
        # If value is invalid or unrecognized
        return None

    samples = []
    for sample_data in data:
        prompt_idx = sample_data['doc_id']
        prompt = sample_data['arguments']['gen_args_0']['arg_0']
        response = sample_data['resps'][0][0]
        filtered_response = ",".join(sample_data['filtered_resps']) if isinstance(sample_data['filtered_resps'][0], str)  else ",".join(sample_data['filtered_resps'][0]) 
        gold = sample_data['target']
        metric = TASK_CONFIG[TASK]['metrics'].split(',')[0] #sample_data['metrics']
        is_correct = standardize_is_correct(sample_data[metric])

        sample = SampleResponse(
            model=model,
            task=TASK,
            subtask = subtask,
            prompt_idx=prompt_idx,
            prompt=prompt,
            response=response,
            filtered_response=filtered_response,
            gold=gold,
            metric=metric,
            is_correct=is_correct,
            timestamp=timestamp_str
        )

        samples.append(sample)
    df_model = pd.DataFrame([s.dict() for s in samples])
    return df_model

def compare_responses_across_models(TASK, task_dir, model_names = None, prompt = None):
    # task_dir structure:
    # task_dir
    #   {model_name}/
    #       samples_{subtask}_{timestamp}.jsonl
    #       results_{timestamp}.json
    if model_names is None:
        model_names = os.listdir(task_dir)

    if isinstance(model_names, str):
        model_names = [model_names]

    df_all = pd.DataFrame()
    for model in model_names:
        # if is_mep:
        #     # if is_mep, then model name is stored as task_id in mongodb, need to convert to model_card_name
        #     client = connect_mongodb()
        #     model = get_clean_model_name_from_mongodb(client, model)
        matched_files = get_sample_path_by_task(task_dir, model)
        for filename, subtask, timestamp, timestamp_str, file_path in matched_files:
            
            # sample_model_path = get_sample_path_by_task(task_dir, model)
            df_model = get_response_df(TASK, file_path, model, timestamp_str, subtask)
            df_all = pd.concat([df_all, df_model], ignore_index=True)

    # get latest timestamp for each prompt_idx and model
    df_pivot = df_all.pivot_table(index=['task', 'subtask', 'prompt_idx', 'prompt', 'gold'], columns='model', values='is_correct', aggfunc='max').reset_index()

    if prompt is not None:
        try:
            #if prompt is an idx
            df_pivot = df_pivot[df_pivot['prompt_idx']==int(prompt)]
            df_all = df_all[df_all['prompt_idx']==int(prompt)]
        except Exception as e:
            #if prompt is the prompt string
            df_pivot = df_pivot[df_pivot['prompt']==prompt]
            df_all = df_all[df_all['prompt']==prompt]
    return df_pivot, df_all
    


if __name__ == '__main__':
    # dataset = _load_dataset('ifeval')
    # tasks =  get_supported_tasks() #['ifeval', 'bbh_fewshot_subset']
    # limits = None # [1.0, 0.05] or 5
    # task_config = get_task_num_samples_from_config(tasks, limits)
    # print(task_config)
    # prompt = """
    # Question: Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?\nAnswer:
    # """
    df_pivot, df_responses = compare_responses_across_models('gsm8k', 'eval_results', model_names = ['llama3.1'])
    print(df_responses)

    