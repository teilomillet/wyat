# Preparing a public Wyat release

Release preparation covers two public surfaces: the pip distributions and the
public source repository `teilomillet/wyat`. Both must contain only explicitly
approved files. Ignored files in the maintainer workspace (research, data,
models, receipts) remain private and are never committed or exported.

Version **0.1.0** is the first release of this API. The existing PyPI `0.0.1`
release contains a different implementation; never reuse or overwrite a
published version. Nothing is published merely by building, exporting or
checking it.

## Export a clean source candidate

The release file allowlist defines the approved runtime, public examples,
documentation, tests and check tooling. Private datasets, models, operational
receipts, benchmark and research tooling are excluded. Keep their original
files in the private workspace; an export is a copy of the selected public
files, not a move or deletion.

From the maintainer workspace, choose a fresh destination:

```sh
.venv/bin/python scripts/package/export.py artifacts/public-release/source
```

The exporter rejects an existing destination and writes a checksum inventory
of the selected files. The exported tree has no `.git` directory, ignored
artifacts or credentials. Git tracks only the allowlisted public files; keep
private work in ignored paths and review `git diff --cached` before each commit.
The export is a check of that boundary, not a separate publication channel.

Inspect the source inventory and the content of every selected file. An
allowlisted filename does not establish that its contents are suitable for the
public audience. Review documentation links, examples, local paths, attribution,
model licenses and workflow destinations as well as Python code.

The exporter, build hook and archive checker run the same content privacy policy
on the selected bytes. Known credential formats and private filesystem paths
are rejected with redacted errors. CI checks source before tests and checks
test/coverage reports before retaining them; coverage paths are relative.
These checks complement independent review and cannot recognize all private
natural-language content. Preserve private originals and transformation evidence
in access-controlled storage outside this repository before removing them from
a release surface. Bind the archive and cleaned release through hashes and a
private change receipt; do not publish internal archive paths.

## Check and build from the exported source

Run the public checks inside the exported directory:

```sh
cd artifacts/public-release/source
uv sync --locked --extra inference --extra server --group examples
uv run --no-sync python scripts/checks/main.py
uv build --no-build-isolation
uv run --no-sync python scripts/package/check.py dist
```

The public check command covers public code, examples and tests. Optional
capability skips remain visible. On Apple Silicon, install `--extra mlx` and use
`--require-mlx` when MLX availability is required; use `--require-metal` for a
required Apple GPU check.

The package checker inspects metadata and exact archive contents, rebuilds from
the source distribution, exercises the installed core and CLI in isolation, and
writes distribution checksums. Inspect both archives and keep their hashes
with the source inventory. The wheel contains the approved runtime; the source
archive also includes the approved public source, examples, documentation,
tests and check tooling. Neither archive contains private research material.

This local build is a pre-check. The distributions uploaded to PyPI are rebuilt
by `pypi.yml` from the tagged commit and checked again there; compare their
`SHA256SUMS` with your local build.

Then install the candidate wheel into a separate application environment and
exercise the [installation guide](install.md), including no-weight first use.
Run trained-model checks separately with an explicitly selected supported
artifact and runtime. A fixture check establishes API mechanics, not trained
quality, calibration, device coverage or performance.

## Pin the model release

`wyat download` and the [standalone verifier](../examples/model/verify.py)
refuse to run until a release is pinned. After publishing the model to
`teilomillet/wyat-1-600m`, copy its commit hash into `REVISION`, the SHA-256 of
`model/native_manifest.json` into `MANIFEST_SHA256` and the digests of
`LICENSE` and `NOTICE` into `LICENSE_FILES` in `wyat/download.py`, and the same
`REVISION` and `MANIFEST_SHA256` into the verifier; a test keeps the two equal.
Then run `wyat download` into an empty directory, load the result and make one
decision before tagging. A model only loads when its manifest records the
artifact contract of the version being released (`artifact_contract()` in
`wyat/loading/native_artifact.py`, versioned by `wyat.tasks.RECIPE` and
`PREDICTOR`).

Every code file bundled with a model release must carry the attribution
`Teïlo Millet - teilomillet` and its SPDX license header, and third-party
notices are retained. The maintainer model-release tooling, kept outside the
public source, adds and checks these headers before generating manifests and
checksums, and before upload.

## Prepare publication

Set the version in `pyproject.toml`, run `uv lock`, and update the minimum
version in `llms.txt` if it changes. The tag must be `v<version>`.

Before publishing, confirm the unused version, reviewed source snapshot,
license and attribution, complete check results, dependency installation and
release notes. Inspect the repository's actual contents and access settings. Publication changes the audience; preparation
does not authorize that external action by itself.

For GitHub Actions publication, configure the PyPI trusted publisher for
repository `teilomillet/wyat` (the workflow refuses any other) with workflow
`pypi.yml` and environment `pypi`. The
repository owner, name, workflow filename and environment must match the trusted
publisher exactly.

`.github/workflows/pypi.yml` publishes when a `v<version>` tag is pushed on a
commit already on `main`: it rejects a tag that differs from the project version,
runs the full CI matrix on that commit, verifies the built distributions' checksums,
refuses a version already on PyPI and uploads through trusted publishing.
Upload runs only when the repository variable `PYPI_PUBLISH_ENABLED` is `true`;
otherwise a tag push checks and builds but uploads nothing. Create the `pypi`
environment with required reviewers before enabling the variable: GitHub creates
a missing environment without protection on first use. Protect `v*` tags, and
verify that the controls apply to the repository and plan. These administrative
settings are not established by this source tree.

Dispatching the workflow manually on `main` is the dry run: the same checks and
build for main's version, never an upload. Run it on the reviewed commit, review
every CI lane, including failures and skipped capabilities, and retain the checked
distributions with their checksums. Pushing the tag is the publication request, so
push an immutable `v<version>` tag only on that dry-run commit after explicit
approval. Trusted publishing and environment protection remain unverified
until the first tagged upload.

## Publish the checked candidate

After explicit publication approval, push the tag on the reviewed commit. The
workflow publishes only its checked wheel and source archive. Do not
rebuild during upload or substitute artifacts with different hashes. PyPI
trusted publishing uses short-lived OIDC credentials; no publishing token
belongs in the source tree.

After upload, verify the registry files and hashes, then install the exact
published version in a clean consumer environment and repeat first use. Only
then describe `pip install wyat` as providing this API.

If an upload fails, retain its logs and compare recorded hashes with any files
already present on PyPI. Uploads can partially succeed. Never move the release
tag or replace a published version; recovery requires a deliberate maintainer
decision, usually a new version.
