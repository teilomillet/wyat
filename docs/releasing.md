# Preparing a public Wyat release

Release preparation covers two public surfaces: the pip distributions and the
source code. Both must contain only explicitly approved files. The mixed
maintainer workspace and its Git history remain private.

For model releases, every bundled code file must also carry the attribution
`Teïlo Millet - teilomillet` and its appropriate SPDX license header. Add headers
before generating source-bound manifests and checksums, retain third-party
notices, and reject missing headers before upload. This applies to every future
model release, including its examples and bundled runtime source.

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

The exporter must reject an existing destination and produce an inventory of
selected files. The public source tree must have no inherited `.git` directory,
commit history, ignored artifacts or credentials. Do not make the mixed private
repository public. Git history pruning is deferred until V1; before changing
repository visibility, review the pruned history as well as the current files.
The clean snapshot provides the candidate file boundary for that review.

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
uv sync --locked --extra inference --extra server
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
encoding recipe (`wyat.tasks.RECIPE`) and head version (`PREDICTOR` in
`wyat/loading/native_artifact.py`) of the version being released.

## Prepare publication

Before publishing, confirm the unused version, reviewed source snapshot,
license and attribution, complete check results, dependency installation and
release notes. Choose the public source destination and inspect its actual
contents and access settings. Publication changes the audience; preparation
does not authorize that external action by itself.

For GitHub Actions publication, configure the workflow and PyPI trusted publisher
for the approved source repository. The repository owner, name, workflow filename
and environment must match the trusted publisher exactly. Keep publication
disabled until those settings and a dry run have been reviewed.

The supplied release workflow is manually dispatched. Its `publish` input
defaults to `false`; publication additionally requires
`PYPI_PUBLISH_ENABLED=true` and the `pypi` environment's protection rules.
Protect the release branch and immutable tags, configure required environment
reviewers where supported, and verify that the controls apply to the chosen
repository and plan. These administrative settings are not established by this
source tree.

Create an immutable `v<version>` tag only on the reviewed public commit. A dry
run must check that exact commit and retain checked distributions with their
checksums. Review every CI lane, including failures and skipped capabilities,
before requesting publication. CI execution, trusted publishing and environment
protection remain unverified until exercised on the destination repository.

## Publish the checked candidate

After explicit publication approval, use the reviewed source destination and
release workflow. Publish only its checked wheel and source archive. Do not
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
