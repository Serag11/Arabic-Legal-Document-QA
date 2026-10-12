"""Build and verify pinned runtime bundles with standard-library integrity checks."""

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import time
import urllib.error
import urllib.request


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def directory_sha256(path):
    root = Path(path)
    files = {p.relative_to(root).as_posix(): file_sha256(p)
             for p in sorted(root.rglob("*")) if p.is_file()}
    return hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()


def download_archive(config, archive):
    # Actions supplies its built-in read token; gh handles private release asset
    # authentication and redirects without exposing credentials in command args.
    gh = os.environ.get("GH_BINARY") or shutil.which("gh")
    token_given = bool(os.environ.get("GH_TOKEN"))
    authenticated = token_given
    if gh and not authenticated:
        authenticated = subprocess.run([gh, "auth", "status"], capture_output=True).returncode == 0
    if authenticated:
        if not gh:
            raise ValueError("Install GitHub CLI to download authenticated release assets")
        try:
            subprocess.run([gh, "release", "download", config["release"], "--repo", config["repository"],
                            "--pattern", config["asset"], "--output", str(archive), "--clobber"], check=True)
            return
        except subprocess.CalledProcessError:
            # CI must fail loudly. A merely logged-in local gh (for example an old version without
            # --clobber) should not block downloading a public release, which needs no login.
            if token_given:
                raise
    url = (f"https://github.com/{config['repository']}/releases/download/"
           f"{config['release']}/{config['asset']}")
    download_public(url, archive)


def download_public(url, archive, attempts=6, timeout=60):
    """Anonymous download that resumes a partial file and retries stalled or dropped connections.

    The release is large and connections can be slow; the checksum is verified before extraction,
    so a wrong or partial file can never be used.
    """
    archive = Path(archive)
    for attempt in range(1, attempts + 1):
        offset = archive.stat().st_size if archive.exists() else 0
        request = urllib.request.Request(url, headers={"Range": f"bytes={offset}-"} if offset else {})
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                resumed = offset > 0 and getattr(response, "status", 200) == 206
                with archive.open("ab" if resumed else "wb") as handle:
                    shutil.copyfileobj(response, handle)
            return
        except urllib.error.HTTPError as exc:
            if exc.code == 416:  # the file on disk is already complete
                return
            if exc.code < 500 or attempt == attempts:
                raise
        except (TimeoutError, ConnectionError, urllib.error.URLError, OSError):
            if attempt == attempts:
                raise
        time.sleep(min(2 ** attempt, 15))


def verify_directory(root, manifest):
    root = Path(root)
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()
              and p.name != "bundle-manifest.json"}
    if actual != set(manifest["files"]):
        raise ValueError("Runtime bundle file set differs from its manifest")
    for name, digest in manifest["files"].items():
        if file_sha256(root / name) != digest:
            raise ValueError(f"Runtime artifact checksum mismatch: {name}")


def verify_pinned_directory(root, config):
    root = Path(root)
    path = root / "bundle-manifest.json"
    if file_sha256(path) != config["bundle_manifest_sha256"]:
        raise ValueError("Runtime manifest differs from pinned configuration")
    manifest = json.loads(path.read_text())
    if manifest["model"] != config["model"]:
        raise ValueError("Runtime model differs from pinned configuration")
    verify_directory(root, manifest)
    return manifest


def unpack(archive, destination, expected_sha256):
    archive, destination = Path(archive), Path(destination)
    if file_sha256(archive) != expected_sha256:
        raise ValueError("Runtime bundle checksum mismatch")
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("Runtime destination must be empty")
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as handle:
        members = handle.getmembers()
        names = set()
        for member in members:
            path = Path(member.name)
            if (not member.isfile() or path.is_absolute() or ".." in path.parts
                    or member.name in names or "\\" in member.name):
                raise ValueError("Invalid runtime archive entry")
            names.add(member.name)
        handle.extractall(destination, filter="data")
    manifest = json.loads((destination / "bundle-manifest.json").read_text())
    verify_directory(destination, manifest)
    return manifest


def bundle(model, index, output, descriptor, tag, repository, corpus="data/egyptian_civil_code_articles.jsonl"):
    model, index, output, descriptor = map(Path, (model, index, output, descriptor))
    files = {}
    for source, prefix in ((model, "model"), (index, "data/retrieval/chroma")):
        for path in sorted(source.rglob("*")):
            if path.is_file():
                if path.is_symlink():
                    raise ValueError("Runtime bundle cannot contain symlinks")
                files[f"{prefix}/{path.relative_to(source).as_posix()}"] = path
    metadata = json.loads((model / "runtime.json").read_text())
    index_manifest = json.loads((index / "manifest.json").read_text())
    if file_sha256(model / "model.onnx") != metadata["weights_sha256"]:
        raise ValueError("Model weights do not match their runtime metadata")
    if file_sha256(corpus) != index_manifest["corpus_sha256"]:
        raise ValueError("Corpus does not match the bundled index")
    # A small corpus copy allows CI corpus-contract tests without DVC or a PDF.
    # Docker copies only the model/index, so it is not duplicated in the image.
    files["data/egyptian_civil_code_articles.jsonl"] = Path(corpus)
    manifest = {"schema_version": 1, "model": metadata,
                "corpus_sha256": index_manifest["corpus_sha256"],
                "chunks_sha256": index_manifest["chunks_sha256"],
                "files": {name: file_sha256(path) for name, path in files.items()}}
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = output.parent / "bundle-manifest.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    files["bundle-manifest.json"] = manifest_path
    with output.open("wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as gz:
        with tarfile.open(fileobj=gz, mode="w") as archive:
            for name, path in sorted(files.items()):
                entry = archive.gettarinfo(str(path), arcname=name)
                entry.uid = entry.gid = entry.mtime = 0
                entry.uname = entry.gname = ""
                entry.mode = 0o644
                with path.open("rb") as handle:
                    archive.addfile(entry, handle)
    config = {"schema_version": 1, "repository": repository, "release": tag,
              "asset": output.name, "sha256": file_sha256(output),
              "bundle_manifest_sha256": file_sha256(manifest_path),
              "model_directory_sha256": directory_sha256(model),
              "index_directory_sha256": directory_sha256(index),
              "model": metadata, "corpus_sha256": manifest["corpus_sha256"],
              "chunks_sha256": manifest["chunks_sha256"]}
    descriptor.parent.mkdir(parents=True, exist_ok=True)
    descriptor.write_text(json.dumps(config, indent=2) + "\n")
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("bundle")
    for name in ("model", "index", "output", "descriptor", "tag", "repository"):
        create.add_argument(f"--{name}", required=True)
    create.add_argument("--corpus", default="data/egyptian_civil_code_articles.jsonl")
    extract = commands.add_parser("unpack")
    extract.add_argument("--archive", required=True)
    extract.add_argument("--output", required=True)
    extract.add_argument("--config", default="config/runtime-artifacts.json")
    fetch = commands.add_parser("fetch")
    fetch.add_argument("--output", default="runtime")
    fetch.add_argument("--config", default="config/runtime-artifacts.json")
    ensure = commands.add_parser("ensure")
    ensure.add_argument("--output", default="runtime")
    ensure.add_argument("--config", default="config/runtime-artifacts.json")
    args = parser.parse_args()
    if args.command == "bundle":
        result = bundle(args.model, args.index, args.output, args.descriptor, args.tag, args.repository, args.corpus)
    else:
        config = json.loads(Path(args.config).read_text())
        if args.command == "ensure" and Path(args.output).exists():
            result = verify_pinned_directory(args.output, config)
            print(json.dumps({"status": "verified cached runtime", "model": result["model"]}, indent=2))
            return
        archive = Path(args.archive) if args.command == "unpack" else Path(args.output).with_suffix(".tar.gz")
        if args.command in ("fetch", "ensure"):
            archive.parent.mkdir(parents=True, exist_ok=True)
            download_archive(config, archive)
        result = unpack(archive, args.output, config["sha256"])
        verify_pinned_directory(args.output, config)
    print(json.dumps({key: value for key, value in result.items() if key != "files"}, indent=2))


if __name__ == "__main__":
    main()
