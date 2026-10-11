"""Exercise the actual image's encoder/index without credentials or network access."""

import argparse
import importlib.util
import json
from pathlib import Path

from .deployment import OnnxEncoder
from .generation import build_context, load_generation_config
from .retrieval import Retriever


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, required=True)
    args = parser.parse_args()
    if importlib.util.find_spec("torch") is not None:
        raise ValueError("The compact runtime must not contain PyTorch")
    retriever = Retriever()
    retriever.encoder = OnnxEncoder(retriever.config, "model")
    generation = load_generation_config()
    records = json.loads(args.evaluation.read_text())["records"]
    if len(records) != 60:
        raise ValueError("Smoke comparison requires the complete fixed evaluation")
    for row in records:
        hits = retriever.search(row["question"], k=generation["retrieval_top_k"])
        context = build_context(hits, generation["max_context_characters"])
        if json.loads(json.dumps(context)) != row["retrieved_contexts"]:
            raise ValueError(f"Image retrieval changed for {row['id']}")
    try:
        retriever.encoder.encode(["lease " * 600], query=True)
    except ValueError as exc:
        if "budget" not in str(exc):
            raise
    else:
        raise ValueError("Oversized input was silently truncated")
    print(json.dumps({"torch_installed": False, "identical_contexts": len(records),
                      "indexed_documents": retriever.collection.count(), "truncation_rejected": True}))


if __name__ == "__main__":
    main()
