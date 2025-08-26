from datasets import load_dataset, get_dataset_config_names, concatenate_datasets
from task_config import TASK_CONFIG
from typing import Optional, Union
import math
import logging

logging.basicConfig(
    level=logging.INFO,                     # minimum log level
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

def get_supported_tasks():
     supported_tasks = list(TASK_CONFIG.keys())
     return supported_tasks

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

if __name__ == '__main__':
    # dataset = _load_dataset('ifeval')
    tasks = get_supported_tasks() #['ifeval', 'bbh_fewshot_subset']
    limits = None # [1.0, 0.05] or 5
    task_config = get_task_num_samples(tasks, limits)
    print(task_config)