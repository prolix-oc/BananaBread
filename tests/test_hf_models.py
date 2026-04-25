import pytest

import bananabread.hf_models as hf_models
from bananabread.hf_models import resolve_model_repo_id


def test_resolves_author_and_path_to_repo_id():
    assert (
        resolve_model_repo_id(author="sentence-transformers", path="all-MiniLM-L6-v2")
        == "sentence-transformers/all-MiniLM-L6-v2"
    )


def test_resolves_direct_hf_slug_from_path():
    assert (
        resolve_model_repo_id(path="mixedbread-ai/mxbai-embed-large-v1")
        == "mixedbread-ai/mxbai-embed-large-v1"
    )


def test_resolves_standard_qwen_selector():
    assert resolve_model_repo_id(model_name="qwen", size="4B") == "Qwen/Qwen3-Embedding-4B"


def test_requires_model_selector():
    with pytest.raises(ValueError, match="Provide author/path"):
        resolve_model_repo_id()


def test_inspect_hf_model_passes_token_to_api(monkeypatch):
    captured = {}

    class Sibling:
        rfilename = "modules.json"

    class Info:
        id = "private/model"
        sha = "abc123"
        tags = ["sentence-transformers"]
        library_name = "sentence-transformers"
        pipeline_tag = "sentence-similarity"
        siblings = [Sibling()]

    class FakeHfApi:
        def __init__(self, token=None):
            captured["token"] = token

        def model_info(self, repo_id, revision=None, files_metadata=False):
            captured["repo_id"] = repo_id
            captured["revision"] = revision
            return Info()

    monkeypatch.setattr(hf_models, "HfApi", FakeHfApi)

    metadata = hf_models.inspect_hf_model("private/model", revision="main", token="hf_test")

    assert captured == {"token": "hf_test", "repo_id": "private/model", "revision": "main"}
    assert metadata["is_embedding_capable"] is True


def test_download_hf_model_passes_token_to_snapshot_download(monkeypatch, tmp_path):
    captured = {}

    def fake_snapshot_download(**kwargs):
        captured.update(kwargs)
        return kwargs["local_dir"]

    monkeypatch.setattr(hf_models, "snapshot_download", fake_snapshot_download)

    local_path = hf_models.download_hf_model(
        "private/model",
        storage_dir=tmp_path,
        revision="main",
        token="hf_test",
    )

    assert captured["repo_id"] == "private/model"
    assert captured["revision"] == "main"
    assert captured["token"] == "hf_test"
    assert local_path.endswith("private--model@main")
