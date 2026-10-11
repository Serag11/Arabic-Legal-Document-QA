"""Export the cached pinned E5 checkpoint. No model downloads or Google calls."""

import argparse
import gc
from pathlib import Path
import shutil

from .retrieval import load_config, load_tokenizer, write_json


def export(output):
    import torch
    from transformers import AutoModel

    config = load_config()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "model.onnx").exists():
        raise ValueError("Export destination already contains a model")
    torch.set_num_threads(config["cpu_threads"])
    model = AutoModel.from_pretrained(config["model_id"], revision=config["model_revision"],
                                      local_files_only=True, token=False, attn_implementation="eager").eval()

    class Embeddings(torch.nn.Module):
        def __init__(self, encoder):
            super().__init__()
            self.encoder = encoder

        def forward(self, input_ids, attention_mask):
            hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
            mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            return torch.nn.functional.normalize(pooled, p=2, dim=1)

    tokenizer = load_tokenizer(config)
    tokenizer.save_pretrained(output)
    sample = tokenizer(["query: What is a lease?", "query: ما تعريف عقد الإيجار؟"],
                       padding=True, return_tensors="pt")
    with torch.no_grad():
        torch.onnx.export(Embeddings(model), (sample["input_ids"], sample["attention_mask"]),
                          str(output / "model.onnx"), dynamo=False, opset_version=17,
                          input_names=["input_ids", "attention_mask"], output_names=["embeddings"],
                          dynamic_axes={"input_ids": {0: "batch", 1: "sequence"},
                                        "attention_mask": {0: "batch", 1: "sequence"},
                                        "embeddings": {0: "batch"}})
    del model
    gc.collect()
    write_metadata(output, config, "float32")


def write_metadata(output, config, precision):
    import importlib.metadata
    from .runtime_artifacts import file_sha256

    write_json(output / "runtime.json", {
        **{key: config[key] for key in ("model_id", "model_revision", "dimensions", "max_tokens")},
        "backend": "onnxruntime", "precision": precision, "pooling": "attention_mask_mean",
        "normalize_embeddings": True, "opset": 17,
        "weights_sha256": file_sha256(output / "model.onnx"),
        "model_bytes": (output / "model.onnx").stat().st_size,
        "onnxruntime_version": importlib.metadata.version("onnxruntime"),
    })


def quantize(source, output):
    from onnxruntime.quantization import QuantType, quantize_dynamic

    source, output = Path(source), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "model.onnx").exists():
        raise ValueError("Quantization destination already contains a model")
    for path in source.iterdir():
        if path.is_file() and path.name not in ("model.onnx", "runtime.json"):
            shutil.copy2(path, output / path.name)
    quantize_dynamic(source / "model.onnx", output / "model.onnx", weight_type=QuantType.QInt8,
                     op_types_to_quantize=["MatMul", "Gather"], per_channel=True,
                     reduce_range=True, extra_options={"MatMulConstBOnly": True})
    write_metadata(output, load_config(), "int8")


def compact(source, output):
    """Store FP16 weights, with explicit FP32 casts for portable CPU computation."""
    import onnx
    from onnx import TensorProto, helper, numpy_helper
    import numpy as np

    source, output = Path(source), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "model.onnx").exists():
        raise ValueError("Compact destination already contains a model")
    for path in source.iterdir():
        if path.is_file() and path.name not in ("model.onnx", "runtime.json"):
            shutil.copy2(path, output / path.name)
    model = onnx.load(source / "model.onnx")
    initializers, casts = [], []
    for tensor in model.graph.initializer:
        if tensor.data_type == TensorProto.FLOAT:
            name = tensor.name + "_fp16_storage"
            initializers.append(numpy_helper.from_array(numpy_helper.to_array(tensor).astype(np.float16), name))
            casts.append(helper.make_node("Cast", [name], [tensor.name], to=TensorProto.FLOAT))
        else:
            initializers.append(tensor)
    del model.graph.initializer[:]
    model.graph.initializer.extend(initializers)
    nodes = list(model.graph.node)
    del model.graph.node[:]
    model.graph.node.extend(casts + nodes)
    onnx.checker.check_model(model)
    onnx.save(model, output / "model.onnx")
    write_metadata(output, load_config(), "float16-storage-float32-compute")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fp32 = commands.add_parser("export")
    fp32.add_argument("--output", type=Path, required=True)
    int8 = commands.add_parser("quantize")
    int8.add_argument("--source", type=Path, required=True)
    int8.add_argument("--output", type=Path, required=True)
    half = commands.add_parser("compact")
    half.add_argument("--source", type=Path, required=True)
    half.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "export":
        export(args.output)
    elif args.command == "quantize":
        quantize(args.source, args.output)
    else:
        compact(args.source, args.output)


if __name__ == "__main__":
    main()
