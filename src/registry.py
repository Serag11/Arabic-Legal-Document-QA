"""Register the measured winning RAG configuration as a reloadable MLflow model."""

import importlib.metadata
import json
from threading import Lock

import mlflow
from mlflow import MlflowClient
from mlflow.models import ModelSignature
from mlflow.types import ColSpec, Schema

from .generation import ARABIC, AnsweringService, ROOT
from .quality_gate import check_current
from .retrieval import write_json


class RegisteredLegalQA(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        specification = json.loads(open(context.artifacts["configuration"]).read())
        self.language_mode = specification["variant"]["document_language"]
        self.service = AnsweringService(config=specification["generation_config"],
                                        index=context.artifacts["index"], prompt_path=context.artifacts["prompt"])
        self.lock = Lock()

    def predict(self, context, model_input, params=None):
        answers = []
        for row in model_input.to_dict(orient="records"):
            question = row["question"]
            language = "ar" if ARABIC.search(question) else "en"
            filters = {"language": "en" if language == "ar" else "ar"} if self.language_mode == "opposite" else {}
            with self.lock:
                answers.append(self.service.ask(question, language=language, filters=filters))
        return answers


def register_best():
    gate = check_current()
    summary = json.loads((ROOT / "reports/experiments/summary.json").read_text())
    report = json.loads((ROOT / summary["validation_file"]).read_text())
    specification = {key: report[key] for key in ("variant", "generation_config", "retrieval_config", "pipeline_fingerprint")}
    configuration_path = ROOT / "reports/experiments/selected-configuration.json"
    write_json(configuration_path, specification)
    mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'data/mlflow/tracking.db'}")
    mlflow.set_experiment("CivilCode/registered-champion")
    requirements = ["--extra-index-url https://download.pytorch.org/whl/cpu"] + [
        f"{name}=={importlib.metadata.version(name)}" for name in (
            "torch", "sentence-transformers", "transformers", "chromadb", "google-genai", "python-dotenv", "pydantic", "pymupdf", "pyarabic",
        )]
    with mlflow.start_run(run_name="register-selected-configuration") as run:
        mlflow.log_dict(specification, "selected-configuration.json")
        mlflow.log_metrics({key: value for key, value in report["metrics"].items() if value is not None})
        info = mlflow.pyfunc.log_model(
            name="legal-qa", python_model=RegisteredLegalQA(),
            artifacts={"configuration": str(configuration_path), "index": str(ROOT / "data/retrieval/chroma"),
                       "prompt": str(ROOT / "prompts/answer_v1.txt")},
            code_paths=[str(ROOT / "src")], pip_requirements=requirements,
            signature=ModelSignature(inputs=Schema([ColSpec("string", "question")])),
            registered_model_name="ArabicLegalDocumentQA",
        )
        version = info.registered_model_version
        MlflowClient().set_registered_model_alias("ArabicLegalDocumentQA", "champion", version)
        result = {"name": "ArabicLegalDocumentQA", "version": version, "alias": "champion",
                  "model_uri": info.model_uri, "registration_run_id": run.info.run_id,
                  "evaluation_run_id": summary["validation_run_id"], "gate": gate,
                  "note": "Registered RAG pipeline/configuration and local index; Google generation weights and API credentials are not stored."}
    write_json(ROOT / "reports/experiments/registry.json", result)
    return result
