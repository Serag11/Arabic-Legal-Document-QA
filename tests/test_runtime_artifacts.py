"""Delivery checks: pinned artifacts and equivalence must fail closed."""

import hashlib
import io
import json
import tarfile

import pytest

from src.deployment_gate import check_equivalence, context_hash
from src.runtime_artifacts import bundle, download_archive, file_sha256, unpack, verify_pinned_directory


def test_bundle_roundtrip_and_tamper_detection(tmp_path):
    model, index = tmp_path / "model", tmp_path / "index"
    model.mkdir()
    index.mkdir()
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text("{}\n")
    (model / "model.onnx").write_bytes(b"model")
    metadata = {"precision": "test", "weights_sha256": file_sha256(model / "model.onnx")}
    (model / "runtime.json").write_text(json.dumps(metadata))
    (index / "manifest.json").write_text(json.dumps({
        "corpus_sha256": file_sha256(corpus), "chunks_sha256": "chunks",
    }))
    archive, descriptor = tmp_path / "runtime.tar.gz", tmp_path / "config.json"
    config = bundle(model, index, archive, descriptor, "runtime-v1", "owner/repo", corpus)
    destination = tmp_path / "unpacked"
    unpack(archive, destination, config["sha256"])
    assert verify_pinned_directory(destination, config)["model"] == metadata
    (destination / "model/model.onnx").write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum mismatch"):
        verify_pinned_directory(destination, config)


@pytest.mark.parametrize("kind", ["traversal", "symlink", "duplicate"])
def test_archive_rejects_unsafe_entries(tmp_path, kind):
    path = tmp_path / "bad.tar.gz"
    with tarfile.open(path, "w:gz") as archive:
        entry = tarfile.TarInfo("../escape" if kind == "traversal" else "model.onnx")
        if kind == "symlink":
            entry.type = tarfile.SYMTYPE
            entry.linkname = "../escape"
            archive.addfile(entry)
        else:
            entry.size = 1
            archive.addfile(entry, io.BytesIO(b"x"))
            if kind == "duplicate":
                archive.addfile(entry, io.BytesIO(b"y"))
    with pytest.raises(ValueError, match="Invalid runtime archive"):
        unpack(path, tmp_path / "out", file_sha256(path))


def test_archive_checksum_checked_before_extraction(tmp_path):
    archive = tmp_path / "not-a-tar.gz"
    archive.write_bytes(b"not an archive")
    destination = tmp_path / "out"
    with pytest.raises(ValueError, match="bundle checksum mismatch"):
        unpack(archive, destination, hashlib.sha256(b"other").hexdigest())
    assert not destination.exists()


def test_runtime_gate_rejects_changed_context_missing_rows_and_stale_model():
    context = [{"source_text": "source", "article_number": 1}]
    baseline = {"records": [{"id": "q-ar", "retrieved_contexts": context,
                             "retrieved_article_numbers": [1]}]}
    model = {"precision": "test"}
    record = {"id": "q-ar", "context_identical": True,
              "context_sha256": context_hash(context), "retrieved_article_numbers": [1]}
    report = {"runtime_fingerprint": "current", "model": model, "records": [record]}
    assert check_equivalence(report, baseline, "current", model) == 1
    with pytest.raises(ValueError, match="stale"):
        check_equivalence(report, baseline, "changed", model)
    with pytest.raises(ValueError, match="every fixed question"):
        check_equivalence(dict(report, records=[]), baseline, "current", model)
    with pytest.raises(ValueError, match="fresh answer evaluation"):
        check_equivalence(dict(report, records=[dict(record, context_sha256="changed")]),
                          baseline, "current", model)


def test_private_release_download_uses_cli_without_token_in_arguments(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setenv("GH_TOKEN", "ci-token")
    monkeypatch.setenv("GH_BINARY", "/usr/bin/gh")
    monkeypatch.setattr("src.runtime_artifacts.subprocess.run", lambda args, **kwargs: calls.append(args))
    download_archive({"release": "runtime-v1", "repository": "owner/repo", "asset": "runtime.tar.gz"},
                     tmp_path / "runtime.tar.gz")
    assert calls[0][:4] == ["/usr/bin/gh", "release", "download", "runtime-v1"]
    assert "ci-token" not in calls[0]


def test_authenticated_download_requires_cli(monkeypatch, tmp_path):
    monkeypatch.setenv("GH_TOKEN", "ci-token")
    monkeypatch.delenv("GH_BINARY", raising=False)
    monkeypatch.setattr("src.runtime_artifacts.shutil.which", lambda name: None)
    with pytest.raises(ValueError, match="Install GitHub CLI"):
        download_archive({}, tmp_path / "runtime.tar.gz")


def test_local_gh_login_failure_falls_back_to_anonymous_public_download(monkeypatch, tmp_path):
    import io
    import subprocess

    monkeypatch.delenv("GH_TOKEN", raising=False)
    monkeypatch.setenv("GH_BINARY", "/usr/bin/gh")

    def fake_run(args, **kwargs):
        if args[1:3] == ["auth", "status"]:
            return subprocess.CompletedProcess(args, 0)
        raise subprocess.CalledProcessError(1, args)

    urls = []
    monkeypatch.setattr("src.runtime_artifacts.subprocess.run", fake_run)
    monkeypatch.setattr("src.runtime_artifacts.urllib.request.urlopen",
                        lambda url, timeout: urls.append(url) or io.BytesIO(b"archive-bytes"))
    archive = tmp_path / "runtime.tar.gz"
    download_archive({"release": "runtime-v1", "repository": "owner/repo", "asset": "runtime.tar.gz"}, archive)
    assert archive.read_bytes() == b"archive-bytes"
    assert urls == ["https://github.com/owner/repo/releases/download/runtime-v1/runtime.tar.gz"]


def test_ci_token_download_failure_is_not_silently_retried_anonymously(monkeypatch, tmp_path):
    import subprocess

    import pytest

    monkeypatch.setenv("GH_TOKEN", "ci-token")
    monkeypatch.setenv("GH_BINARY", "/usr/bin/gh")
    monkeypatch.setattr("src.runtime_artifacts.subprocess.run",
                        lambda args, **kwargs: (_ for _ in ()).throw(subprocess.CalledProcessError(1, args)))
    with pytest.raises(subprocess.CalledProcessError):
        download_archive({"release": "r", "repository": "o/r", "asset": "a"}, tmp_path / "a")
