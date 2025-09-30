from fastapi import APIRouter, Query
from app.task_config import TASK_CONFIG
from app.task_stats import _load_dataset, get_supported_tasks, get_task_num_samples_from_config, get_accuracy_results_score, compare_responses_across_models, connect_mongodb
from typing import Union, List

router = APIRouter(prefix="/api", tags=["API"])

@router.get("/health")
def health_check():
    return {"status": "ok"}

@router.get("/tasks/config")
def get_task_config():
    return {"task_config": TASK_CONFIG}

@router.get("/tasks")
def list_tasks():
    tasks = get_supported_tasks()
    return {"tasks": tasks}

@router.get("/tasks/samples")
def get_task_samples(task_name: str):
    if isinstance(task_name, str):
        task_name = [task_name]
    for task in task_name:
        if task not in TASK_CONFIG:
            return {"error": "Task not found"}
    num_samples = get_task_num_samples_from_config(task_name)
    return {"task": task_name, "num_samples": num_samples}

@router.get("/tasks/connect_db")
def connect_db():
    try:
        client = connect_mongodb()
        # Attempt a simple operation to verify the connection
        client.admin.command('ping')
        return {"status": "Connected to MongoDB successfully"}
    except Exception as e:
        return {"error": str(e)}

@router.get("/tasks/results")
def get_results(task, results_path):
    if task not in TASK_CONFIG:
        return {"error": "Task not found"}
    accuracy, metric = get_accuracy_results_score(task, results_path)
    if accuracy is None:
        return {"error": "Could not compute accuracy. Check results path and format."}
    return {"task": task, "accuracy": accuracy, "metric": metric}

@router.get("/tasks/accuracy_per_sample")
def get_accuracy_per_sample(task, eval_dir, model_names: List[str] = Query(...), prompt: Union[int, str, None] = None):
    if task not in TASK_CONFIG:
        return {"error": "Task not found"}
    df_pivot, df_responses = compare_responses_across_models(task, eval_dir, model_names, prompt)
    if df_pivot is None or df_responses is None:
        return {"error": "Could not compute accuracy per sample. Check eval directory and format."}
    return {
        "pivot_table": df_pivot.to_dict(orient='records'),
        "responses": df_responses.to_dict(orient='records')
    }

