# Preparing a public wyat release

Release preparation covers two public surfaces: the pip distributions and the
public source repository `teilomillet/wyat`. Both must contain only explicitly
approved files. Ignored files in the maintainer workspace (research, data,
models, receipts) remain private and are never committed or exported.

Version **0.2.0** loads wyat-2 1.4B (Ouro) decision artifacts, with exact
CUDA speed work, bounded-memory streaming on macOS and memory warnings when a
Mac is short of headroom. Version **0.1.2** added MLX inference for the 800M
artifact, quieter PyTorch loading and updated examples. Version **0.1.1** introduced the option-fork
loader and pinned 800M download. Version **0.1.0** introduced this API. The old PyPI `0.0.1`
release contains a different implementation; never reuse or overwrite a
published version. Nothing is published merely by building, exporting or
checking it.

Pushing `main` prepares the source. Creating the version tag is a separate
publication step.

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

The default for `wyat.load()` and `wyat download` is the pinned
`teilomillet/wyat-1-800m` release. Its canonical descriptor in
[`wyat/download.py`](../wyat/download.py) records the immutable Hub commit,
SHA-256 of `model/fork_manifest.json`, digests of `LICENSE` and `NOTICE`, all
eight payload filenames and total download size. Keep those pins aligned with
the reviewed upload whenever the model release changes; never copy the 600M
revision, manifest or runtime contract into the 800M descriptor.

Before tagging, run bare `wyat download` into an empty directory, verify its
result against that descriptor, load it with PyTorch and make a decision.
Exercise `wyat.load()` from an empty cache as well as a verified cache. The
manifest must match `contract()` in
[`wyat/loading/fork_artifact.py`](../wyat/loading/fork_artifact.py), and the
downloader must preserve its eight payload files, including `vocab.json` and
`merges.txt`. Also check the MLX runtime on Apple Silicon; it loads the same
800M artifact without conversion.

The 600M compatibility artifact keeps a separate native descriptor and the
[standalone 600M verifier](../examples/model/verify.py). Keep their immutable
revision and manifest hash consistent when updating that artifact; neither
its native manifest nor its release pins apply to 800M.

Append benchmark results only after the selected evaluation has complete,
validated outputs for the exact model artifact. Identify the suite version,
split, model identity, task denominators and scoring method. Keep Choice,
Score and Noul separate, retain unsupported or failed rows, and identify
partial comparisons as incomplete. A benchmark result does not remove a
separate model-release requirement.

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
otherwise a tag push checks and builds but uploads nothing. The `pypi`
environment requires a maintainer's approval before the upload job runs and
accepts deployments only from `v*` tags, and a repository ruleset blocks
updating or deleting `v*` tags. These are GitHub settings, not files in this
tree: check them under the repository's Environments and Rulesets before
enabling the variable, and keep the environment in place, because GitHub
recreates a missing environment without protection.

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
