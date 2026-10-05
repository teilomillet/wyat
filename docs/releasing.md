# Releasing Wyat

Releases are deliberate. [The release workflow](../.github/workflows/release.yml)
accepts an existing tag through **Run workflow** on `main`; ordinary pushes,
tag pushes, and GitHub release events do not publish anything. Its `publish`
input defaults to `false`. Publication additionally requires the repository
variable `PYPI_PUBLISH_ENABLED` to equal `true` and the `pypi` environment's
protection rules to pass.

The existing [PyPI version 0.0.1](https://pypi.org/project/wyat/0.0.1/) differs
from this implementation. Do not publish current code under that version.
Choose a new, unused version in a separate reviewed change; this infrastructure
does not change the package version or publish a release.

## One-time maintainer setup

These are administrative actions, not settings this checkout has configured.
Keep `PYPI_PUBLISH_ENABLED` unset or `false` until all are complete.
PyPI makes the distributions public. Review that audience for the runtime and
the source archive, which retains historical research evidence, before enabling
publication from this private repository.

1. Protect `main` with reviewed changes and required CI checks. Protect release
   tags from deletion or movement, and restrict who can create them. Review
   workflow changes as executable release infrastructure.
2. Create the GitHub environment **`pypi`**. Restrict its deployment branch to
   **`main`**, because the manual workflow runs on that branch even though it
   tests a selected tag. Configure required reviewers, prevent self-review,
   and disable administrator bypass where the repository's plan permits those
   controls. Confirm the controls actually apply to this repository's visibility
   and plan before enabling publishing. GitHub describes their availability and
   setup in [Managing environments](https://docs.github.com/en/actions/how-tos/deploy/configure-and-manage-deployments/manage-environments).
3. As an owner of the existing `wyat` PyPI project, add its GitHub trusted
   publisher with **owner `teilomillet`**, **repository `wyat`**, **workflow
   filename `release.yml`**, and **environment `pypi`**. Those values must match
   exactly; follow [PyPI's configuration guide](https://docs.pypi.org/trusted-publishers/adding-a-publisher/).
4. After a successful dry run and review of those settings, set the GitHub
   repository Actions variable **`PYPI_PUBLISH_ENABLED=true`**. This variable
   is an enable switch, not a credential. Remove it or set it to `false` to
   disable subsequent publication requests.

No PyPI API token or GitHub personal token belongs in this workflow. Only the
publishing job receives `id-token: write`; PyPI exchanges that OIDC identity for
short-lived publishing credentials. The `pypi` environment supplies the review
boundary. See [Using a trusted publisher](https://docs.pypi.org/trusted-publishers/using-a-publisher/).

## Prepare and inspect a candidate

1. Choose an unused version, update `pyproject.toml` and refresh `uv.lock` in a
   reviewed change, and merge it with passing CI. Check the release notes and
   installed-package examples against that implementation.
2. Create an immutable tag **`v<version>`** on the reviewed commit reachable
   from `main`. For example, version `0.0.2` would require tag `v0.0.2`; this is
   an illustration, not a selected next version or an instruction to tag now.
3. Open **Actions → Release → Run workflow**, select branch **`main`**, enter
   that existing tag, and leave **publish unchecked**.
4. Inspect every CI lane, including failures and skips, then download the
   **`ci-distributions`** artifact and its **`SHA256SUMS`**. Inspect the wheel,
   source archive, metadata, and installed-package behavior before publication.

The workflow resolves the tag once to a commit, rejects tags outside `main`'s
history, and requires its version to match `pyproject.toml`. It calls the same
[CI workflow](../.github/workflows/ci.yml) for that exact commit. CI runs locked
dependency installation, types, formatting, tests, and package checks; only
the Linux Python 3.12 lane builds the wheel and source archive with
`uv build --no-build-isolation`, using the build backend installed from
`uv.lock`. The package
checker validates the release version, package contents, and isolated core/CLI
imports, then writes the checksums. [uv's GitHub Actions guide](https://docs.astral.sh/uv/guides/integration/github/)
describes the underlying locked-environment workflow.

The other lanes exercise Python compatibility and the MLX dependency path.
They do not establish deployment performance, downloaded-model quality, CUDA
coverage, or Metal availability on a particular hosted runner. Read the actual
test results and capability skips.

## Publish the checked artifact

After the administrative setup and explicit release approval, dispatch **Release**
again on `main` with the same immutable tag and **publish checked**. This run
rechecks the tag and reruns CI; it publishes that run's checked artifact, not
an artifact from the earlier dry run. Approve the `pypi` environment only after
reviewing its selected commit and complete CI results.

The publishing job downloads the same run's `ci-distributions`, verifies its
exact checksum inventory, and rejects a version already present on PyPI. It
does not check out project code, install project dependencies, or rebuild.
It sends only the verified wheel and source archive to PyPI, with trusted
publishing and provenance attestations. External actions use reviewed full
commit hashes; see [GitHub's secure-use guidance](https://docs.github.com/en/actions/reference/security/secure-use).

If uploading fails, retain the workflow logs and compare the recorded hashes
with any files already visible on PyPI before deciding how to recover. Uploads
can partially succeed: the workflow deliberately does not skip existing files
or cancel a publishing run to start another one. Never move the tag or silently
replace a published version. A partial release needs an explicit maintainer
decision, usually a new version.

CI and release execution, PyPI trust, environment approval, and registry
publication remain unverified until they are exercised on GitHub. Local checks
of these files do not establish that those administrative controls exist.
