"""CPU-only ONNX serving adapter; the evaluated FP32 development path stays available."""

import json
from pathlib import Path

import numpy as np

from .api import create_app
from .generation import AnsweringService, ROOT
from .retrieval import Retriever, token_count


class OnnxEncoder:
    def __init__(self, config, model_dir):
        import onnxruntime as ort
        from transformers import AutoTokenizer

        self.config = config
        self.model_dir = Path(model_dir)
        self.metadata = json.loads((self.model_dir / "runtime.json").read_text())
        for key in ("model_id", "model_revision", "dimensions", "max_tokens"):
            if self.metadata[key] != config[key]:
                raise ValueError(f"ONNX model/index mismatch: {key}")
        self.device = "cpu"
        self.precision = self.metadata["precision"]
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir, local_files_only=True, token=False)
        options = ort.SessionOptions()
        options.intra_op_num_threads = config["cpu_threads"]
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(self.model_dir / "model.onnx"), options,
                                            providers=["CPUExecutionProvider"])

    def encode(self, texts, *, query=False):
        prefix = self.config["query_prefix"] if query else self.config["passage_prefix"]
        inputs = [prefix + text for text in texts]
        if not inputs:
            return np.empty((0, self.config["dimensions"]), dtype=np.float32)
        if any(token_count(self.tokenizer, text) > self.config["max_tokens"] for text in inputs):
            raise ValueError("Input exceeds embedding budget; truncation is forbidden")
        vectors = []
        for start in range(0, len(inputs), self.config["batch_size"]):
            batch = self.tokenizer(inputs[start:start + self.config["batch_size"]],
                                   padding=True, truncation=False, return_tensors="np")
            vectors.append(self.session.run(["embeddings"], {
                name: batch[name].astype(np.int64) for name in ("input_ids", "attention_mask")
            })[0])
        result = np.concatenate(vectors).astype(np.float32)
        if result.shape != (len(inputs), self.config["dimensions"]) or not np.isfinite(result).all():
            raise ValueError("Invalid ONNX embedding output")
        if not np.allclose(np.linalg.norm(result, axis=1), 1, atol=1e-5):
            raise ValueError("ONNX embeddings must be normalized")
        return result


class DeploymentService(AnsweringService):
    def __init__(self):
        retriever = Retriever(ROOT / "data/retrieval/chroma")
        retriever.encoder = OnnxEncoder(retriever.config, ROOT / "model")
        super().__init__(retriever=retriever)

    def ask(self, *args, **kwargs):
        result = super().ask(*args, **kwargs)
        result["configuration"]["embedding_runtime"] = self.retriever.encoder.metadata
        return result


app = create_app(service_factory=DeploymentService)
