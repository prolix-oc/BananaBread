import time
import threading
import torch
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from mxbai_rerank import MxbaiRerankV2

from bananabread.config import logger, args
from bananabread.models.qwen import QwenRawModel

# Global model references
embedding_model = None
embedding_model_pool = None
rerank_model = None
rerank_model_pool = None
embedding_model_name = ""

# Lazy-loaded classifier
classifier = None

# ----- Model Pool for Concurrent GPU Execution -----

class ModelPool:
    """
    Pool of model instances for concurrent GPU execution.
    Each thread in the threadpool gets its own model instance to avoid CUDA context issues.
    """
    
    def __init__(self, num_instances: int, model_loader_func, model_name: str):
        """
        Initialize model pool
        
        Args:
            num_instances: Number of model instances to create
            model_loader_func: Function that loads and returns a model instance
            model_name: Name of the model (for logging)
        """
        self.num_instances = num_instances
        self.model_name = model_name
        self.models = []
        self.model_index = 0
        self.lock = threading.Lock()
        
        logger.info(f"🔄 Creating model pool for {model_name} with {num_instances} instance(s)...")
        
        # Load multiple model instances
        for i in range(num_instances):
            logger.info(f"  Loading model instance {i+1}/{num_instances}...")
            model = model_loader_func()
            self.models.append(model)
            
        logger.info(f"✅ Model pool initialized with {len(self.models)} instance(s)")
    
    def get_model(self):
        """
        Get next model from pool using round-robin selection.
        Thread-safe.
        """
        with self.lock:
            model = self.models[self.model_index]
            self.model_index = (self.model_index + 1) % self.num_instances
            return model
    
    def get_all_models(self):
        """Get all models in the pool (for cleanup, etc.)"""
        return self.models

# ----- Compilation and Warmup -----

def compile_model_if_enabled(model, model_name: str):
    """
    Apply torch.compile() to a model if enabled via CLI flags.
    """
    if not args.enable_torch_compile:
        return model
    
    # Check PyTorch version
    torch_version = torch.__version__.split('+')[0]  # Remove any +cu118 suffix
    major, minor, patch = torch_version.split('.')[:3]
    
    if int(major) < 2:
        logger.warning(f"⚠️  torch.compile() requires PyTorch 2.0+, current version: {torch.__version__}")
        logger.warning(f"⚠️  Skipping compilation for {model_name}")
        return model
    
    try:
        logger.info(f"🔥 Compiling {model_name} with torch.compile()")
        logger.info(f"   Mode: {args.torch_compile_mode}")
        logger.info(f"   Backend: {args.torch_compile_backend}")
        
        # For SentenceTransformer models, compile the underlying encode method
        if hasattr(model, 'encode'):
            # Check if it's our QwenRawModel wrapper - compilation support for raw HF models
            if isinstance(model, QwenRawModel):
                logger.info(f"   Compiling underlying HF model for QwenRawModel...")
                model.model = torch.compile(
                    model.model,
                    mode=args.torch_compile_mode,
                    backend=args.torch_compile_backend
                )
                logger.info(f"✅ {model_name} compiled successfully")
                return model

            # Compile the model's encode function for SentenceTransformers
            compiled_model = model
            original_encode = model.encode
            
            # Create a wrapper that compiles the internal operations
            def compiled_encode(*args, **kwargs):
                return original_encode(*args, **kwargs)
            
            # Note: torch.compile works best on the underlying transformer model
            # For SentenceTransformer, we compile the first module (transformer)
            if hasattr(model, '_first_module'):
                try:
                    transformer = model._first_module()
                    if hasattr(transformer, 'auto_model'):
                        logger.info(f"   Compiling underlying transformer model...")
                        transformer.auto_model = torch.compile(
                            transformer.auto_model,
                            mode=args.torch_compile_mode,
                            backend=args.torch_compile_backend
                        )
                        logger.info(f"✅ {model_name} compiled successfully")
                    else:
                        logger.warning(f"⚠️  Could not access auto_model for compilation")
                except Exception as e:
                    logger.warning(f"⚠️  Failed to compile transformer: {e}")
            
            return compiled_model
        
        else:
            # Direct compilation
            compiled = torch.compile(
                model,
                mode=args.torch_compile_mode,
                backend=args.torch_compile_backend
            )
            logger.info(f"✅ {model_name} compiled successfully")
            return compiled
            
    except Exception as e:
        logger.error(f"❌ Failed to compile {model_name}: {e}")
        logger.warning(f"⚠️  Continuing with uncompiled model")
        return model

def warmup_model(model, model_type: str, model_name: str, num_samples: int = 3):
    """
    Perform warmup inference on a model to trigger compilation and cache warming.
    """
    if args.disable_warmup:
        logger.info(f"⏭️  Skipping warmup for {model_name} (disabled via --disable-warmup)")
        return
    
    logger.info(f"🔥 Warming up {model_name} ({model_type})...")
    logger.info(f"   Running {num_samples} warmup inference(s)...")
    
    start_time = time.time()
    
    try:
        with torch.inference_mode():
            for i in range(num_samples):
                if model_type == 'embedding':
                    # Warmup with dummy text of varying lengths
                    dummy_texts = [
                        "This is a warmup sample.",
                        "This is a longer warmup sample with more tokens to test batching behavior.",
                        "Short warmup."
                    ]
                    
                    if hasattr(model, 'encode'):
                        # SentenceTransformer or QwenRawModel
                        _ = model.encode([dummy_texts[i % len(dummy_texts)]])
                    else:
                        logger.warning(f"⚠️  Model does not have encode method, skipping warmup")
                        return
                
                elif model_type == 'rerank':
                    # Warmup with dummy query and documents
                    dummy_query = "What is machine learning?"
                    dummy_docs = [
                        "Machine learning is a subset of artificial intelligence.",
                        "Deep learning uses neural networks with multiple layers.",
                        "Python is a popular programming language for data science."
                    ]
                    
                    if hasattr(model, 'rank'):
                        _ = model.rank(dummy_query, dummy_docs, top_k=2)
                    else:
                        logger.warning(f"⚠️  Model does not have rank method, skipping warmup")
                        return
                
                elif model_type == 'classification':
                    # Warmup with dummy text
                    dummy_text = "This is a test sentence for classification warmup."
                    
                    if callable(model):
                        _ = model(dummy_text)
                    else:
                        logger.warning(f"⚠️  Model is not callable, skipping warmup")
                        return
        
        # Clear CUDA cache after warmup if using GPU
        if torch.cuda.is_available():
            device_str = ""
            if hasattr(model, 'device'):
                device_str = str(model.device)
            elif model_type == 'embedding':
                device_str = args.embedding_device
            elif model_type == 'rerank':
                device_str = args.rerank_device
            
            if device_str and (device_str.startswith('cuda') or device_str == 'mps'):
                torch.cuda.empty_cache()
        
        elapsed_time = time.time() - start_time
        logger.info(f"✅ Warmup completed for {model_name} in {elapsed_time:.2f}s")
        
    except Exception as e:
        logger.error(f"❌ Warmup failed for {model_name}: {e}")
        logger.warning(f"⚠️  Continuing anyway, but first inference may be slower")

class PooledQwenReranker:
    """Wrapper for Qwen reranker that uses a model pool"""
    def __init__(self, model_pool):
        self.model_pool = model_pool
        logger.info(f"♻️  Qwen reranker initialized using shared model pool")
    
    def rank(self, query: str, documents: list[str], return_documents: bool = False, top_k: int = None, task_description: str = None) -> dict:
        """Rank using a model from the shared pool"""
        # Get a model from the pool
        model = self.model_pool.get_model()
        # Use the rank method of the pooled QwenRawModel
        return model.rank(query, documents, return_documents=return_documents, top_k=top_k, task_description=task_description)

# ----- Initialization Logic -----

def load_embedding_model_instance():
    """Load a single embedding model instance based on args"""
    if args.embedding_model == 'qwen':
        qwen_model_name = f"Qwen/Qwen3-Embedding-{args.qwen_size}"
        # Use the raw transformer implementation for Qwen
        model = QwenRawModel(
            qwen_model_name,
            device_arg=args.embedding_device,
            use_flash_attention=args.qwen_flash_attention
        )
    else:
        # MixedBread model
        model = SentenceTransformer(
            "mixedbread-ai/mxbai-embed-large-v1",
            truncate_dim=args.embedding_dim,
            device=args.embedding_device
        )
    
    # Apply torch.compile() if enabled
    model = compile_model_if_enabled(model, f"Embedding-{args.embedding_model}")
    
    return model

def initialize_models():
    global embedding_model, embedding_model_pool, rerank_model, rerank_model_pool, embedding_model_name
    
    logger.info("Initializing models...")
    
    # Detect if we'll be using GPU
    using_gpu_embedding = args.embedding_device != "cpu"
    using_gpu_rerank = args.rerank_device != "cpu"
    
    # Determine reranking model choice
    if args.reranking_model is None:
        if args.embedding_model == 'qwen':
            reranking_model_choice = 'qwen'
            logger.info("No reranking model specified, using qwen (same as embedding model)")
        else:
            reranking_model_choice = 'mixedbread'
    else:
        reranking_model_choice = args.reranking_model

    # Determine if we use shared pool
    use_shared_qwen_pool = (
        args.embedding_model == 'qwen' and
        reranking_model_choice == 'qwen' and
        args.embedding_device == args.rerank_device and
        (using_gpu_embedding or using_gpu_rerank) and
        (args.num_concurrent_embedding > 1 or args.num_concurrent_rerank > 1)
    )
    
    # ----- Embedding Model Init -----
    logger.info(f"Loading embedding model on device: {args.embedding_device}")
    logger.info(f"Using embedding model: {args.embedding_model}")
    
    shared_qwen_pool = None
    
    if use_shared_qwen_pool:
        shared_pool_size = max(args.num_concurrent_embedding, args.num_concurrent_rerank)
        qwen_model_name = f"Qwen/Qwen3-Embedding-{args.qwen_size}"
        embedding_model_name = qwen_model_name
        
        logger.info(f"♻️  Creating SHARED model pool for both embedding and reranking")
        logger.info(f"   Using {shared_pool_size} Qwen instance(s) for both tasks")
        
        shared_qwen_pool = ModelPool(
            shared_pool_size,
            load_embedding_model_instance,
            embedding_model_name
        )
        embedding_model_pool = shared_qwen_pool
        embedding_model = None
        
    elif using_gpu_embedding and args.num_concurrent_embedding > 1:
        if args.embedding_model == 'qwen':
            embedding_model_name = f"Qwen/Qwen3-Embedding-{args.qwen_size}"
        else:
            embedding_model_name = "mixedbread-ai/mxbai-embed-large-v1"
        
        embedding_model_pool = ModelPool(
            args.num_concurrent_embedding,
            load_embedding_model_instance,
            embedding_model_name
        )
        embedding_model = None
        
    else:
        # Single instance
        if args.embedding_model == 'qwen':
            embedding_model_name = f"Qwen/Qwen3-Embedding-{args.qwen_size}"
            embedding_model = QwenRawModel(
                embedding_model_name,
                device_arg=args.embedding_device,
                use_flash_attention=args.qwen_flash_attention
            )
        else:
            embedding_model_name = "mixedbread-ai/mxbai-embed-large-v1"
            embedding_model = SentenceTransformer(embedding_model_name, truncate_dim=args.embedding_dim, device=args.embedding_device)
        
        # Apply compile to single instance
        embedding_model = compile_model_if_enabled(embedding_model, f"Embedding-{args.embedding_model}")
        embedding_model_pool = None

    # Warmup embedding
    if embedding_model_pool:
        logger.info(f"🔥 Warming up embedding model pool...")
        for i, model in enumerate(embedding_model_pool.get_all_models()):
            warmup_model(model, 'embedding', f"{embedding_model_name}-{i+1}", num_samples=args.warmup_samples)
    elif embedding_model:
        warmup_model(embedding_model, 'embedding', embedding_model_name, num_samples=args.warmup_samples)

    # ----- Rerank Model Init -----
    logger.info(f"Using reranking model: {reranking_model_choice}")
    
    if reranking_model_choice == 'qwen':
        if use_shared_qwen_pool:
            logger.info("♻️  Reranker using SHARED pool with embedding model")
            rerank_model = PooledQwenReranker(shared_qwen_pool)
            rerank_model_pool = None
        elif (args.embedding_model == 'qwen' and 
              args.embedding_device == args.rerank_device and
              args.num_concurrent_embedding == 1 and
              args.num_concurrent_rerank == 1):
             logger.info("♻️  Reusing Qwen embedding model for reranking")
             rerank_model = embedding_model
             rerank_model_pool = None
        else:
            qwen_reranker_model_name = f"Qwen/Qwen3-Embedding-{args.qwen_size}"
            if using_gpu_rerank and args.num_concurrent_rerank > 1:
                def load_qwen_reranker():
                    return QwenRawModel(
                        qwen_reranker_model_name,
                        device_arg=args.rerank_device,
                        use_flash_attention=args.qwen_flash_attention
                    )
                rerank_model_pool = ModelPool(
                    args.num_concurrent_rerank,
                    load_qwen_reranker,
                    qwen_reranker_model_name
                )
                rerank_model = None
            else:
                rerank_model = QwenRawModel(
                    qwen_reranker_model_name,
                    device_arg=args.rerank_device,
                    use_flash_attention=args.qwen_flash_attention
                )
                rerank_model_pool = None
    else:
        # MixedBread
        rerank_model = MxbaiRerankV2("mixedbread-ai/mxbai-rerank-base-v2", device=args.rerank_device)
        rerank_model_pool = None

    # Apply compile to single rerank instance
    if rerank_model is not None and not rerank_model_pool:
        if not use_shared_qwen_pool and rerank_model != embedding_model: # Don't re-compile if shared
            name = f"Rerank-{reranking_model_choice}" if reranking_model_choice != 'qwen' else "Rerank-Qwen"
            rerank_model = compile_model_if_enabled(rerank_model, name)

    # Warmup reranker
    if rerank_model_pool:
        logger.info(f"🔥 Warming up reranking model pool...")
        for i, model in enumerate(rerank_model_pool.get_all_models()):
            warmup_model(model, 'rerank', f"{reranking_model_choice}-{i+1}", num_samples=args.warmup_samples)
    elif rerank_model and not use_shared_qwen_pool and rerank_model != embedding_model:
        warmup_model(rerank_model, 'rerank', f"Rerank-{reranking_model_choice}", num_samples=args.warmup_samples)
    
    logger.info("Models initialized successfully")

def get_classifier():
    """Lazy-load the classification pipeline on first use"""
    global classifier
    if classifier is None:
        logger.info("🔄 Loading classification model (first use)...")
        classifier = pipeline(
            'text-classification',
            model='SamLowe/roberta-base-go_emotions',
            top_k=None
        )
        logger.info("✅ Classification model loaded")
    return classifier
