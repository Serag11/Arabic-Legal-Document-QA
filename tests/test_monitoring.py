"""Dashboard and alert definitions must reference metrics the service really exports."""

import json
from pathlib import Path
import re

from prometheus_client import CollectorRegistry
import yaml

from src.drift import DriftBaseline
from src.metrics import Metrics

MONITORING = Path(__file__).resolve().parents[1] / "monitoring"
NAME = re.compile(r"\blegal_qa_[a-z0-9_]+")


def exported_names():
    metrics = Metrics(CollectorRegistry(), pricing={"input": 1, "output": 1},
                      drift=DriftBaseline({"centroid": [1.0], "cosine_mean": 1, "cosine_std": 0}))
    names = set()
    for family in metrics.registry.collect():
        # Labelled families have no samples until used, so derive the exposed names from the type.
        suffixes = {"counter": ("_total",), "histogram": ("_bucket", "_sum", "_count"), "gauge": ("",)}[family.type]
        names |= {family.name + suffix for suffix in suffixes}
    return names


def test_dashboard_queries_use_exported_metrics():
    dashboard = json.loads((MONITORING / "grafana/dashboards/legal-qa.json").read_text())
    used = {name for panel in dashboard["panels"] for target in panel["targets"] for name in NAME.findall(target["expr"])}
    assert used and used <= exported_names(), sorted(used - exported_names())


def test_alert_rules_use_exported_metrics_and_are_documented():
    rules = yaml.safe_load((MONITORING / "alerts.yml").read_text())["groups"][0]["rules"]
    assert len(rules) >= 6
    for rule in rules:
        assert rule["labels"]["severity"] in {"warning", "critical"} and rule["annotations"]["summary"]
        missing = set(NAME.findall(rule["expr"])) - exported_names()
        assert not missing, (rule["alert"], missing)


def test_every_alert_has_a_unit_test():
    rules = {rule["alert"] for rule in yaml.safe_load((MONITORING / "alerts.yml").read_text())["groups"][0]["rules"]}
    tested = {case["alertname"] for test in yaml.safe_load((MONITORING / "tests/alerts_test.yml").read_text())["tests"]
              for case in test["alert_rule_test"]}
    assert rules <= tested, sorted(rules - tested)
