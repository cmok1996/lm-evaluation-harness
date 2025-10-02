from fastapi import APIRouter, Query
from task_config import TASK_CONFIG
from task_stats import _load_dataset, get_supported_tasks, get_task_num_samples_from_config, get_accuracy_results_score, get_sample_path_by_task, get_results_path_by_task, compare_responses_across_models, connect_mongodb, prepare_leaderboard_data
from typing import Union, List

router = APIRouter(prefix="/tasks", tags=["tasks"])


@router.get("/health")
def health_check():
    return {"status": "ok"}

@router.get("/config")
def get_task_config():
    return {"task_config": TASK_CONFIG}

@router.get("/tasks")
def list_tasks():
    tasks = get_supported_tasks()
    return {"tasks": tasks}

@router.get("/num_samples")
def get_task_samples(task_name: str, limit: Union[int, float, None]=None):
    try:
        if isinstance(task_name, str):
            task_name = [task_name]
        for task in task_name:
            if task not in TASK_CONFIG:
                return {"error": "Task not found"}
        num_samples = get_task_num_samples_from_config(task_name, limit)
        return {"task": task_name, "num_samples": num_samples}
    except Exception as e:
        return {"error": str(e)}

@router.get("/connect_db")
def connect_db():
    try:
        client = connect_mongodb()
        # Attempt a simple operation to verify the connection
        client.admin.command('ping')
        return {"status": "Connected to MongoDB successfully"}
    except Exception as e:
        return {"error": str(e)}
    
@router.get("/get_samples_path")
def get_samples_path(output_path: str, model_name: str):
    try:
        sample_path_ = get_sample_path_by_task(output_path, model_name)
        sample_path = [s[-1] for s in sample_path_]
        if sample_path is None:
            return {"error": "Sample path not found for the specified task"}
    except Exception as e:
        return {"error": str(e)}
    return {"sample_path": sample_path}

@router.get("/get_results_path")
def get_results_path(output_path: str, model_name: str):
    try:
        results_path_ = get_results_path_by_task(output_path, model_name)
        results_path = [s[-1] for s in results_path_]
        if results_path is None:
            return {"error": "Results path not found for the specified task"}
    except Exception as e:
        return {"error": str(e)}
    return { "results_path": results_path}

@router.get("/results")
def get_results(task, output_path:str, model):
    try:
        if task not in TASK_CONFIG:
            return {"error": "Task not found"}
        results = get_accuracy_results_score(task, output_path, model)
        if not results:
            return {"error": "Could not compute accuracy. Check results path and format."}
    except Exception as e:
        return {"error": str(e)}
    return results

@router.get("/accuracy_per_sample")
def get_accuracy_per_sample(task, output_path, model_names: List[str] = Query(None), prompt: Union[int, str, None] = None):
    """ Get accuracy per sample and detailed responses for a given task.
    
    Parameters:
    - task: Task name (must be in TASK_CONFIG)
    - output_path: Path to the LM eval harness results directory as per --output_path argument
    - model_names: List of model names or task_ids in output_path directory to filter results (optional)
    - prompt: Specific prompt idx or prompt str to filter results (optional)
    
    """
    try:

        
        if task not in TASK_CONFIG:
            return {"error": "Task not found"}
        df_pivot, df_responses = compare_responses_across_models(task, output_path, model_names, prompt)


        if df_pivot is None or df_responses is None:
            return {"error": "Could not compute accuracy per sample. Check eval directory and format."}
        return {
            "pivot_table": df_pivot,
            "responses": df_responses
        }
    except Exception as e:
        return {"error": str(e)}

@router.get("/leaderboard")
def get_leaderboard(eval_dir: str, tasks: List[str] = Query(None), models: List[str] = Query(None), min_num_samples: int = 0):
    """ Get leaderboard of models across specified tasks.
    
    Parameters:
    - eval_dir: Parent path to the LM eval harness results directory, one path before as per --output_path argument
    - tasks: List of task names (must be in TASK_CONFIG). If None, includes all supported tasks.
    - models: List of model names or task_ids in output_path directory to filter results (optional)
    - min_num_samples: Minimum number of samples a model must have evaluated on a task to be included (default=0)
    
    """
    try:
        supported_tasks = get_supported_tasks()
        if tasks is None:
            tasks = supported_tasks
        else:
            for task in tasks:
                if task not in supported_tasks:
                    return {"error": f"Task '{task}' not found"}
        df_leaderboard, leaderboard_data = prepare_leaderboard_data(eval_dir, tasks, models, min_num_samples)
        if df_leaderboard is None:
            return {"error": "Could not prepare leaderboard. Check eval directory and format."}

        return {
            "leaderboard": df_leaderboard,
            "detailed_data": leaderboard_data
        }
    except Exception as e:
        return {"error": str(e)}