from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, snapshot_download


EMBEDDING_PIPELINE_TAGS = {"feature-extraction", "sentence-similarity"}
EMBEDDING_TAGS = {
    "sentence-transformers",
    "sentence-similarity",
    "feature-extraction",
    "text-embeddings-inference",
    "embeddings",
}

STANDARD_MODEL_REPOS = {
    "mixedbread": "mixedbread-ai/mxbai-embed-large-v1",
    "mxbai-embed-large-v1": "mixedbread-ai/mxbai-embed-large-v1",
}


def resolve_model_repo_id(
    *,
    author: str | None = None,
    path: str | None = None,
    model_name: str | None = None,
    size: str | None = None,
) -> str:
    """Resolve supported model selectors to a Hugging Face repo id."""
    clean_author = author.strip("/ ") if author else None
    clean_path = path.strip("/ ") if path else None
    clean_model_name = model_name.strip("/ ") if model_name else None
    clean_size = size.strip() if size else None

    if clean_author and clean_path:
        return f"{clean_author}/{clean_path}"

    if clean_path and "/" in clean_path:
        return clean_path

    if clean_model_name:
        model_key = clean_model_name.lower()
        if model_key == "qwen":
            return f"Qwen/Qwen3-Embedding-{clean_size or '0.6B'}"
        if clean_size:
            repo_name = f"{clean_model_name}-{clean_size}"
            return f"{clean_author}/{repo_name}" if clean_author else repo_name
        if model_key in STANDARD_MODEL_REPOS:
            return STANDARD_MODEL_REPOS[model_key]

    if clean_path:
        return clean_path

    raise ValueError("Provide author/path, a Hugging Face repo id in path, or model_name.")


def _safe_local_name(repo_id: str, revision: str | None = None) -> str:
    name = repo_id.replace("/", "--")
    if revision:
        name = f"{name}@{revision.replace('/', '--')}"
    return name


def inspect_hf_model(
    repo_id: str,
    *,
    revision: str | None = None,
    token: str | None = None,
) -> dict[str, Any]:
    api = HfApi(token=token)
    info = api.model_info(repo_id, revision=revision, files_metadata=False)
    tags = set(info.tags or [])
    library_name = getattr(info, "library_name", None)
    pipeline_tag = getattr(info, "pipeline_tag", None)
    sibling_names = {s.rfilename for s in info.siblings or []}

    is_sentence_transformer = (
        library_name == "sentence-transformers"
        or "sentence-transformers" in tags
        or "modules.json" in sibling_names
        or "sentence_bert_config.json" in sibling_names
    )
    is_embedding_capable = (
        is_sentence_transformer
        or pipeline_tag in EMBEDDING_PIPELINE_TAGS
        or bool(tags & EMBEDDING_TAGS)
    )

    return {
        "repo_id": info.id,
        "sha": info.sha,
        "revision": revision,
        "library_name": library_name,
        "pipeline_tag": pipeline_tag,
        "tags": sorted(tags),
        "is_sentence_transformer": is_sentence_transformer,
        "is_embedding_capable": is_embedding_capable,
    }


def download_hf_model(
    repo_id: str,
    *,
    storage_dir: str | Path,
    revision: str | None = None,
    token: str | None = None,
    allow_patterns: str | list[str] | None = None,
    ignore_patterns: str | list[str] | None = None,
) -> str:
    storage_path = Path(storage_dir).expanduser().resolve()
    storage_path.mkdir(parents=True, exist_ok=True)
    local_dir = storage_path / _safe_local_name(repo_id, revision)

    return snapshot_download(
        repo_id=repo_id,
        revision=revision,
        token=token,
        local_dir=str(local_dir),
        allow_patterns=allow_patterns,
        ignore_patterns=ignore_patterns,
    )
