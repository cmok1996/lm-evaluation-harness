from pydantic import BaseModel, Field
from typing import List, Optional, Union

class SampleResponse(BaseModel):
    model: Optional[str] = Field(default=None, description="Model name")
    # platform: Optional[str] = Field(default = None, description = "Platform type: tower, AIPC, etc")
    # device: Optional[str] = Field(default = None, description = "CPU/ GPU/ NPU")
    # precision: Optional[str] = Field(default = "4bit", description = "Precision")
    # quantization : Optional[str] = Field(default = None, description = "Quantization method" )
    # inference_service: Optional[str] = Field(default = None, description = "Inference service")
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