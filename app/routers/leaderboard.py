from fastapi import APIRouter, Query
from task_config import TASK_CONFIG
from task_stats import _load_dataset, sanitize_floats, get_supported_tasks, prepare_leaderboard_data, get_supported_use_cases, get_use_case_weights, calculate_use_case_score
from typing import Union, List, Dict
from pydantic import BaseModel
import pandas as pd

router = APIRouter(prefix="/leaderboard", tags=["leaderboard"])


# Define the request model
class UseCaseRequest(BaseModel):
    leaderboard_data: List[Dict]  # JSON list of records
    usecase: str

@router.get("/health")
def health_check():
    return {"status": "ok"}

   
@router.get("/use_cases")
def get_usecases():
    use_cases = get_supported_use_cases()

    return {"use_cases": use_cases}

@router.get('/use_cases_weights')
def get_usecases_weights(usecase):
    weights = get_use_case_weights(usecase)
    return {"weights": weights}

@router.post('/use_case_score')
def get_usecase_score(request: UseCaseRequest):
    
    """
    Calculate use-case aggregated score

    Parameters:
    - leaderboard_data (list): each element is a row that corresponds to the accuracy for each specified tasks together with the aggregated usecase score of a model as columns
    - usecase (str): supported use case

    Returns:
    - usecase_score (list)
    """
    leaderboard_data = pd.DataFrame(request.leaderboard_data)
    df_usecase_score = calculate_use_case_score(leaderboard_data, request.usecase)
    usecase_score = sanitize_floats(df_usecase_score.to_dict(orient='records'))
    return {"result": usecase_score}

@router.get("/leaderboard")
def get_leaderboard(eval_dir: str, tasks: List[str] = Query(None), use_case:str = None, models: List[str] = Query(None), min_num_samples: int = 0):
    """ Get leaderboard of models across specified tasks.
    
    Parameters:
    - eval_dir (str): Parent path to the LM eval harness results directory, one path before as per --output_path argument
    - tasks (list, None): List of task names (must be in TASK_CONFIG). If None, includes all supported tasks.
    - models (list, None): List of model names or task_ids in output_path directory to filter results (optional)
    - min_num_samples (int): Minimum number of samples a model must have evaluated on a task to be included (default=0)
    
    Returns:
    - leaderboard (list): each element is a row that corresponds to the accuracy for each specified tasks together with the aggregated usecase score of a model as columns
    - detailed_data (list): each element is a row that corresponds to the accuracy of a single task with its corresponding weights accoridng to the usecase
    """
    try:
        supported_tasks = get_supported_tasks()
        if tasks is None:
            tasks = supported_tasks
        else:
            for task in tasks:
                if task not in supported_tasks:
                    return {"error": f"Task '{task}' not found"}
        df_leaderboard, leaderboard_data = prepare_leaderboard_data(eval_dir, tasks, use_case, models, min_num_samples)
        if df_leaderboard is None:
            return {"error": "Could not prepare leaderboard. Check eval directory and format."}

        return {
            "leaderboard": df_leaderboard,
            "detailed_data": leaderboard_data
        }
    except Exception as e:
        return {"error": str(e)}