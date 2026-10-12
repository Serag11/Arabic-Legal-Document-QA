"""Exercise the actual image's encoder/index without credentials or network access."""

import argparse
import importlib.util
import json
from pathlib import Path

from .deployment import OnnxEncoder
from .generation import build_context, load_generation_config
from .retrieval import Retriever


def equivalent(row, context):
    """Does the image's retrieval for `row` match the evaluated one?

    Supported questions must match exactly. For unsupported (off-topic) probes the 4th-8th articles
    have nearly tied scores (measured gaps as small as 1e-5), so float differences between CPUs can
    swap the boundary article; those probes may differ by one article if the top hit is unchanged.
    """
    current = json.loads(json.dumps(context))
    if current == row["retrieved_contexts"]:
        return True
    if row["expected_article_numbers"]:
        return False
    now = list(dict.fromkeys(p["article_number"] for p in current))
    before = list(dict.fromkeys(p["article_number"] for p in row["retrieved_contexts"]))
    return bool(now) and now[0] == before[0] and len(set(now) & set(before)) >= len(before) - 1


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
    tolerated = []
    for row in records:
        hits = retriever.search(row["question"], k=generation["retrieval_top_k"])
        context = build_context(hits, generation["max_context_characters"])
        if not equivalent(row, context):
            raise ValueError(f"Image retrieval changed for {row['id']}")
        if json.loads(json.dumps(context)) != row["retrieved_contexts"]:
            tolerated.append(row["id"])
    try:
        retriever.encoder.encode(["lease " * 600], query=True)
    except ValueError as exc:
        if "budget" not in str(exc):
            raise
    else:
        raise ValueError("Oversized input was silently truncated")
    print(json.dumps({"torch_installed": False, "identical_contexts": len(records) - len(tolerated),
                      "tolerated_unsupported_probe_differences": tolerated,
                      "indexed_documents": retriever.collection.count(), "truncation_rejected": True}))


if __name__ == "__main__":
    main()
