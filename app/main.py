from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes import router

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
app.include_router(router)

@app.get("/")
def root():
    return {"message": "Welcome to LM Eval Harness API"}

