from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers.tasks import router as task_router
from routers.leaderboard import router as leaderboard_router
import uvicorn

app = FastAPI(
    title="LM Eval Harness API",
    version="1.0.0",
    description="LM Eval Harness API"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],  
)
 
# Include routes
app.include_router(task_router)
app.include_router(leaderboard_router)

@app.get("/")
def root():
    return {"message": "Welcome to LM Eval Harness API"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

