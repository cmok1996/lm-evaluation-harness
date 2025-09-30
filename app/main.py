from fastapi import FastAPI
from app.routes import router
from app.task_config import TASK_CONFIG
from app.task_stats import _load_dataset, get_supported_tasks, get_task_num_samples_from_config

app = FastAPI(
    title="LM Eval Harness API",
    version="1.0.0",
    description="LM Eval Harness API"
)

# Include routes
app.include_router(router)

@app.get("/")
def root():
    return {"message": "Welcome to LM Eval Harness API"}

