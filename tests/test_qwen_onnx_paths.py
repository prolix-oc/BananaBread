import pytest


def qwen_onnx_model(monkeypatch, tmp_path):
    monkeypatch.setenv("BANANABREAD_CONFIG", str(tmp_path / "config.json"))
    from bananabread.models.qwen import QwenOnnxModel

    return QwenOnnxModel


def test_missing_onnx_directory_is_created(tmp_path, monkeypatch):
    QwenOnnxModel = qwen_onnx_model(monkeypatch, tmp_path)
    model_dir = tmp_path / "models" / "qwen3-embedding-0.6b-int8-onnx"

    with pytest.raises(FileNotFoundError, match="No \\.onnx model found"):
        QwenOnnxModel._resolve_onnx_path(str(model_dir))

    assert model_dir.is_dir()


def test_missing_onnx_file_parent_is_created(tmp_path, monkeypatch):
    QwenOnnxModel = qwen_onnx_model(monkeypatch, tmp_path)
    model_file = tmp_path / "models" / "qwen3-embedding-0.6b-int8-onnx" / "model.onnx"

    with pytest.raises(FileNotFoundError, match="ONNX model file does not exist"):
        QwenOnnxModel._resolve_onnx_path(str(model_file))

    assert model_file.parent.is_dir()


def test_optimized_onnx_path_creates_output_directory(tmp_path, monkeypatch):
    QwenOnnxModel = qwen_onnx_model(monkeypatch, tmp_path)
    model_file = tmp_path / "models" / "qwen3-embedding-0.6b-int8-onnx" / "model.onnx"
    model_file.parent.mkdir(parents=True)
    model_file.write_bytes(b"placeholder")

    optimized_path = QwenOnnxModel._optimized_onnx_path(model_file)

    assert optimized_path == model_file.parent / "optimized" / "model.onnx"
    assert optimized_path.parent.is_dir()


def test_directory_prefers_model_onnx_when_multiple_files_exist(tmp_path, monkeypatch):
    QwenOnnxModel = qwen_onnx_model(monkeypatch, tmp_path)
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    (model_dir / "z.onnx").write_bytes(b"placeholder")
    preferred = model_dir / "model.onnx"
    preferred.write_bytes(b"placeholder")

    assert QwenOnnxModel._resolve_onnx_path(str(model_dir)) == preferred
