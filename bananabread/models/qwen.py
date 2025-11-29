import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from typing import List, Optional

from bananabread.config import logger

# ----- Qwen Raw Model Class -----

class QwenRawModel:
    """
    Qwen model implementation using raw transformers (AutoModel/AutoTokenizer).
    Implements embedding and reranking logic using last token pooling and specific prompt formatting.
    """
    def __init__(self, model_name: str, device_arg: str = "cpu", use_flash_attention: bool = False):
        """
        Initialize Qwen model using raw transformers.
        """
        self.model_name = model_name
        self.device_arg = device_arg
        
        logger.info(f"Loading Qwen raw model: {model_name}")
        
        # Initialize tokenizer with left padding as required for last token pooling
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
        
        # Determine device settings
        kwargs = {}
        if use_flash_attention:
            kwargs["attn_implementation"] = "flash_attention_2"
            
        # Handle device mapping logic
        if device_arg.lower() == "cpu":
            device_map = None
        elif device_arg.lower() in ["auto", "cuda"]:
            device_map = "auto"
        else:
            # Specific cuda device like "cuda:0"
            device_map = None # We will manually .to() later if needed
            
        # Load model
        if device_map:
            self.model = AutoModel.from_pretrained(
                model_name,
                dtype=torch.bfloat16,
                device_map=device_map,
                **kwargs
            )
            self.device = self.model.device
        else:
            self.model = AutoModel.from_pretrained(
                model_name,
                dtype=torch.bfloat16,
                **kwargs
            )
            if device_arg.lower() != "cpu":
                self.model.to(device_arg)
            self.device = self.model.device
            
        self.model.eval()
        logger.info(f"Qwen raw model initialized on {self.device} with padding_side='left'")

    def last_token_pool(self, last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Pooling strategy for Qwen embedding models:
        Use the embedding of the last token (eos or before padding).
        """
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

    def get_detailed_instruct(self, task_description: str, query: str) -> str:
        """Format query with task instruction"""
        return f'Instruct: {task_description}\nQuery:{query}'

    def get_embeddings(self, texts: List[str], batch_size: int = 8) -> torch.Tensor:
        """Generate embeddings for a list of texts using last token pooling"""
        embeddings = []
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            inputs = self.tokenizer(
                batch, 
                padding=True, 
                truncation=True, 
                return_tensors="pt", 
                max_length=8192
            ).to(self.model.device)
            
            with torch.inference_mode():
                outputs = self.model(**inputs)
                # Use last token pooling
                batch_embeddings = self.last_token_pool(outputs.last_hidden_state, inputs['attention_mask'])
                embeddings.append(batch_embeddings)
                
        if not embeddings:
            return torch.tensor([])
            
        all_embeddings = torch.cat(embeddings, dim=0)
        # Normalize embeddings
        return F.normalize(all_embeddings, p=2, dim=1)

    def encode(self, sentences: List[str], prompt_name: str = None, batch_size: int = 8, **kwargs) -> torch.Tensor:
        """
        Compatibility wrapper for SentenceTransformer's encode.
        Ignores prompt_name as per user's simplified example (or adds if strictly needed).
        Returns torch tensor (endpoints expect .tolist() or something convertible).
        """
        return self.get_embeddings(sentences, batch_size=batch_size)

    def rank(self, query: str, documents: list[str], return_documents: bool = False, top_k: int = None, task_description: str = None) -> dict:
        """
        Rerank candidates based on similarity to query using Qwen task instructions.
        """
        # Format query with instruction
        task = task_description if task_description else 'Given a web search query, retrieve relevant passages that answer the query'
        formatted_query = self.get_detailed_instruct(task, query)
        
        query_emb = self.get_embeddings([formatted_query])
        candidate_embs = self.get_embeddings(documents)
        
        # Cosine similarity (inputs are already normalized, so this works as dot product)
        # query_emb is (1, D), candidate_embs is (N, D)
        scores = F.cosine_similarity(query_emb, candidate_embs)
        
        # Sort by score
        # torch.argsort sorts ascending by default, so we use descending=True
        if top_k is not None:
            top_indices = torch.argsort(scores, descending=True)[:top_k]
        else:
            top_indices = torch.argsort(scores, descending=True)
            
        results = []
        scores_list = scores.tolist()
        top_indices_list = top_indices.tolist()
        
        for idx in top_indices_list:
            result = {
                "index": idx,
                "score": float(scores_list[idx])
            }
            if return_documents:
                result["document"] = documents[idx]
            results.append(result)
            
        return {"results": results}
