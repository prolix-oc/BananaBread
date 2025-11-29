from typing import Optional, List, Union, Dict
from pydantic import BaseModel

# ----- Request Schemas -----

class RerankRequest(BaseModel):
    query: str
    documents: list[str]
    top_k: int = 3
    return_documents: bool = False
    task_description: Optional[str] = None

# Embedding request schema matching OpenAI's format:
class EmbeddingRequest(BaseModel):
    model: str = "mixedbread-ai/mxbai-embed-large-v1"
    input: Union[str, List[str], List[int], List[List[int]]]
    encoding_format: str = "float" # float or base64
    user: Optional[str] = None

class ClassificationRequest(BaseModel):
    input: str
    top_k: int | None = None   # Optional; if provided, only the top K results will be returned.
    sorted: bool = False       # If True, sort results from highest to lowest score.

# ----- Ollama Compatible Schemas -----

class OllamaMessage(BaseModel):
    role: str
    content: str

class OllamaRequest(BaseModel):
    model: str
    messages: list[OllamaMessage]
    stream: bool = False
    options: dict = {}
    format: str = None
    keep_alive: str = None

class OllamaResponse(BaseModel):
    model: str = "mixedbread-ai/mxbai-embed-large-v1"
    created_at: str = ""
    message: OllamaMessage
    done_reason: str = "stop"
    done: bool = True
    total_duration: int = 0
    load_duration: int = 0
    prompt_eval_count: int = 0
    prompt_eval_duration: int = 0
    eval_count: int = 0
    eval_duration: int = 0

class OllamaEmbeddingRequest(BaseModel):
    model: str
    prompt: Optional[str] = None
    input: Optional[Union[str, List[str]]] = None
    truncate: bool = True
    options: dict = {}
    keep_alive: str = "5m"

class OllamaEmbeddingResponse(BaseModel):
    embedding: list[float]
    model: str

# ----- Llama.cpp Compatible Schemas -----

class LlamaCppEmbeddingRequest(BaseModel):
    content: Union[str, List[str]]
    model: str = "mixedbread-ai/mxbai-embed-large-v1"
    normalize: bool = True
    truncate: bool = True

class LlamaCppEmbeddingResponse(BaseModel):
    embedding: Union[list[float], list[list[float]]]
    model: str
