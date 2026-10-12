# Stage 6 — CI and compact, self-contained delivery

## Current status

Implementation, local image verification, and GitHub Actions delivery are complete.
Commit `efef5f368e46e5b714bc03dffb9eed9aef630922` passed delivery run
https://github.com/Serag11/Arabic-Legal-Document-QA/actions/runs/38151794649
in 4m37s. The measured-quality workflow also passed (run `38151794656`).
The delivery runner verified the full 60-context comparison, oversized-input
rejection, HTTP readiness, publication, and anonymous pull from Docker Hub.

Published image: `serag11/arabic-legal-document-qa:latest` and
`serag11/arabic-legal-document-qa:sha-efef5f368e46e5b714bc03dffb9eed9aef630922`.
Published manifest digest:
`sha256:c0e7d883d1752f3c9b724f978bd960443f0da1e98737bab7a0de646f57efdb0b`.
No image was downloaded locally for this verification.
The `runtime-v1` release is published; GitHub reports the archive as uploaded,
529,911,987 bytes, with SHA256 matching `config/runtime-artifacts.json`:
`eed251f9176e9ad497382a1185a6fe3ff0c77216681858ae0b61e7aecac2c81a`.
Verified through GitHub's API without downloading another local copy.
Release: https://github.com/Serag11/Arabic-Legal-Document-QA/releases/tag/runtime-v1

GitHub CLI authentication now works, and repository Actions secrets
`DOCKERHUB_USERNAME` and `DOCKERHUB_TOKEN` were confirmed through GitHub's API.
The repository is private; the workflow uses its built-in token to access assets.
A draft `runtime-v1` release was created (release ID `409634368`) and its small
`bundle-manifest.json` asset uploaded successfully. Uploading the 529,911,987-byte
archive repeatedly stalled with the installed GitHub CLI 2.4.0, including an
HTTP/1.1 retry. Incomplete `starter` assets were removed. The release remains a
draft until the user completed browser upload/publication; publication and
archive integrity are now confirmed.

Delivery target: **`serag11/arabic-legal-document-qa`**, public, Linux AMD64 CPU.
The existing development Dockerfile/Compose setup remains available.

## Model selection

All candidates use the pinned E5 revision
`d128750597153bb5987e10b1c3493a34e5a4502a`, the existing FP32 passage index,
the same 512-token limit, E5 prefixes, attention-mask mean pooling, normalized
768-dimensional embeddings, and top-k 5 article-grouped retrieval.

| ONNX model | Model bytes | Expected article hits | Identical generator contexts |
| --- | ---: | ---: | ---: |
| FP32 | 1,110,014,565 | 55/56 | 60/60 |
| Dynamic INT8, MatMul + Gather | 278,611,767 | 53/56 | 11/60 |
| **FP16 storage, FP32 computation** | **555,132,709** | **55/56** | **60/60** |

INT8 was rejected because it lost two Arabic retrieval hits. The selected graph
stores floating-point initializers in FP16 and explicitly casts them to FP32
before computation. This is a disk/image-size optimization, not a claim of
half-precision CPU execution or a 50% reduction in runtime memory. ONNX Runtime
can expand the stored weights during graph optimization.

Comparison reports: `reports/deployment/{fp32,int8,compact}-retrieval.json`.
The selected report binds the deployment code/dependency lock, complete model
directory, index directory, and existing full answer-evaluation report by hashes.

## Quality verification

The source-faithful passages **and their order** supplied to generation match
the selected Stage 5 evaluation for all 60 questions, including refusal probes.
The generation configuration and original pipeline fingerprint remain current.
Consequently the measured 0.9250 coverage-adjusted faithfulness score is reused
for identical generator inputs; no new Google generation or RAGAS calls were made.
This validates equivalence on the fixed dataset, not universal equivalence for
every possible question or independent legal correctness.

`python -m src.deployment_gate` rejects changed deployment code/dependencies,
model/index files, baseline reports, incomplete comparisons, or any changed
generator context. A changed context requires fresh answer evaluation before
this delivery path can be promoted.

Local checks:

- 77 offline tests passed, including archive integrity, private-release download, and promotion-gate tests.
- Project correctness lint and full unused/undefined-name lint for new modules passed.
- Original measured quality gate passed.
- Self-contained image built successfully; uncompressed Docker image size:
  **about 1.28 GB / 1.19 GiB** (registry transfer size is not measured).
- Actual image, with networking disabled and no PyTorch installed, reproduced
  all 60 evaluated contexts with its bundled model/index.
- FastAPI startup and `/health` passed: 2,242 indexed documents, 1,149 articles.
- An unmatched article filter returned localized insufficient evidence and
  reported the ONNX embedding runtime without contacting Google.

The local Docker login was rejected while fetching a public base image. Building
with an isolated anonymous Docker configuration succeeded; the user's existing
Docker login/configuration was not modified.

## Runtime bundle and CI

Local bundle: `data/deployment/releases/runtime-v1.tar.gz` (Git-ignored).
Pin: `config/runtime-artifacts.json`.

The bundle contains the ONNX model/tokenizer, Chroma index, an integrity manifest,
and a small corpus copy for CI corpus-contract tests. The image copies only the
model/index and integrity manifest; it excludes the duplicate corpus copy,
PDF, visual QA crops, evaluation tooling, PyTorch, and local MLflow artifacts.

`.github/workflows/delivery.yml`:

1. Checks baseline measured quality and compact-runtime provenance/equivalence.
2. Runs pinned Ruff lint.
3. Downloads or verifies the cached, checksum-pinned GitHub Release bundle.
4. Restores the corpus copy for tests and installs frozen development dependencies.
5. Runs offline tests and builds `Dockerfile.runtime`.
6. Exercises the actual image's full retrieval comparison with networking disabled,
   then checks HTTP readiness with a dummy API key. No Google API calls occur.
7. On successful `main` builds, logs in using `DOCKERHUB_USERNAME` and
   `DOCKERHUB_TOKEN`, then publishes `latest` and `sha-<full-commit>` tags with
   provenance/SBOM attestations. Pull requests do not publish.

Neither Google Drive credentials nor a Google API key is needed in CI.
DVC stays the local versioning/reproduction tool.
Private release assets are downloaded through GitHub CLI with the workflow's
built-in `GH_TOKEN` and `contents: read` permission. Public assets can also be
downloaded anonymously. No additional GitHub token secret is needed. Anonymous
Docker Hub pull is checked after publication to verify public image access.

## Bootstrap the first publication

The bundle must exist on GitHub before the delivery workflow can download it.
Use GitHub CLI with write access to `Serag11/Arabic-Legal-Document-QA`:

```bash
gh auth login --web
gh release create runtime-v1 data/deployment/releases/runtime-v1.tar.gz --repo Serag11/Arabic-Legal-Document-QA --target main --latest=false --title "Runtime artifacts v1" --notes "Pinned compact ONNX E5 model, tokenizer, Chroma index, and corpus-test copy. Checksums are recorded in config/runtime-artifacts.json."
```

Alternatively, create a release tagged `runtime-v1` in GitHub's web UI and upload
the exact local archive. Do not rename or rebuild the asset without updating its
pin. The release is an artifact release; its tag is not the Docker image version.

The installed GitHub CLI is authenticated but is version 2.4.0 and does not
support `--latest=false`. Since the draft already exists, use GitHub's web UI to
edit it, upload the exact archive, and publish it instead of creating another
release. This browser upload/publication step is now complete.

The implementation was committed and pushed to `main`, and the delivery workflow
and public Docker Hub tags were verified as recorded above.

## Rebuild the artifact locally

Start with the DVC-tracked corpus/index and the already cached pinned E5 model.
Export tooling is deliberately outside the production dependency lock:

```bash
uv run --frozen --with onnx==1.20.1 python -m src.export_runtime export --output data/deployment/fp32
uv run --frozen --with onnx==1.20.1 python -m src.export_runtime compact --source data/deployment/fp32 --output data/deployment/compact
uv run --frozen python -m src.benchmark_runtime --model data/deployment/compact --output reports/deployment/compact-retrieval.json
uv run --frozen python -m src.runtime_artifacts bundle --model data/deployment/compact --index data/retrieval/chroma --output data/deployment/releases/runtime-v1.tar.gz --descriptor config/runtime-artifacts.json --tag runtime-v1 --repository Serag11/Arabic-Legal-Document-QA
python -m src.deployment_gate
```

Export commands require new/empty model destinations; existing models are not
overwritten. For a later artifact update use a **new release tag and archive
name**, update the descriptor, and rerun validation before publishing.

To build from a published bundle:

```bash
python -m src.runtime_artifacts ensure --output runtime
docker build -f Dockerfile.runtime -t serag11/arabic-legal-document-qa:local .
```

`ensure` validates cached files, not just their presence. An invalid cache fails
rather than being silently reused. Local unpacking of the prepared archive is:

```bash
python -m src.runtime_artifacts unpack --archive data/deployment/releases/runtime-v1.tar.gz --output runtime
```

Use an empty destination. The descriptor's archive digest is checked before
extraction; links, path traversal, duplicate entries, and unexpected file sets
are rejected.
For private release downloads, authenticate `gh` locally first. If the CLI is
outside PATH, set `GH_BINARY` to its executable path.

Intermediate export/quantization models, expanded bundle copies, and temporary
Stage 6 images are removed after verification to limit local storage use. The
compressed release archive is retained for upload. Further image builds should
run on GitHub rather than accumulating local copies.

## Remaining verification

- Perform the separately deferred clean-checkout `dvc pull` / `dvc repro` check.
  This is independent of the verified CI release-bundle and container delivery.


## CI smoke-test tolerance for unsupported probes (2026-10-12)

The offline image smoke test (`src/runtime_smoke.py`) compares the image's retrieved contexts with the
evaluation's recorded contexts. One run failed with "Image retrieval changed for s2-u02-ar" on a commit
whose only change was the README; re-running the same commit passed. The cause is nearly tied scores for
off-topic queries: for the unsupported probes the 5th and 6th articles differ by as little as 0.00001
(`s2-u01-en`) or 0.00009 (`s2-u02-ar`), so floating-point differences between GitHub runner CPUs can swap
the boundary article. Supported questions have clear winners and must still match exactly; unsupported
probes may now differ by one article provided the top hit is unchanged (`equivalent()`, unit-tested). Any
tolerated difference is listed in the smoke output (`tolerated_unsupported_probe_differences`).
