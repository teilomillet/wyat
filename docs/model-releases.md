# Model release attribution and privacy

Every code file bundled with a Wyat model release must carry attribution for
**Teïlo Millet - teilomillet**, including examples and vendored Wyat runtime code.
Preserve existing third-party notices. The current release tooling supports
UTF-8 Python (`.py` and `.pyi`); other code kinds need a reviewed header rule
before inclusion. Unknown file kinds are rejected instead of being treated as
data automatically. Supported non-code formats and filenames are explicitly
listed in the release tool. Do not relabel code with incompatible third-party
licensing.

Keep the author attribution in code headers and `NOTICE`. Do not add an
explanation of attribution headers, signing changes or their verification
receipts to the model README; its license links and third-party notices suffice.

The canonical Python header is:

```python
# SPDX-FileCopyrightText: 2026 Teïlo Millet - teilomillet
# SPDX-License-Identifier: Apache-2.0
```

Run preparation on a fresh staging tree immediately after copying the code:

```sh
.venv/bin/python scripts/package/model_release.py prepare artifacts/model-release
```

This preserves executable shebangs, encoding cookies, module docstrings and
existing attribution. Preparation is idempotent. It refuses to change unsigned
code after manifests or checksums have been created: source hashes are part of
the native loader contract, so signing must happen **before** constructing
`native_manifest.json`, `runtime/SOURCE.json`, `RELEASE.json` and verification
receipts. Generate source-bound contracts from the signed bundled runtime,
rather than the unsigned maintainer checkout. Run the loader and inference
checks with that bundled runtime first on the import path, then create
`FILES.sha256` and freeze the full file inventory.

The reviewed inventory is JSON with `repo_id`, `parent_commit` and a `files`
mapping from each relative filename to its `bytes` count and `sha256`. Include
all release files, including `FILES.sha256`; keep the review inventory outside
the upload tree. Have the exact bundle and inventory independently reviewed.

Every release also runs the canonical content policy in
`scripts/package/privacy.py`. The model profile admits reviewed artifact paths,
rejects known credential/private-path patterns and raw-record fields, excludes
candidate/class/small-cell diagnostics and confidential cohort commitments,
and checks safetensors headers for arbitrary metadata and unreferenced bytes.
Changing an allowed artifact or adding another format requires policy review.
The checks cannot recognize all confidential prose or measure model memorization.

Before reducing a release's disclosure, preserve the exact private originals,
source pins, scripts, raw reports and transformation records outside the source
repository in access-controlled storage. Verify every copy by size and SHA-256
before retiring a remote artifact. Keep a manifest mapping the original source
and revision to the archived path and hash, then append a remediation receipt
linking that manifest to the sanitized inventory and resulting remote revision.
Keep local vault paths and internal manifests out of the upload tree. A local
restricted archive needs a separately arranged encrypted backup for off-device
durability. Never use public history or a public report as the private archive.

After independent review passes, add `privacy_review` to the external inventory:
`status` is `PASS`; `policy_sha256` is the exact SHA-256 of `privacy.py`; and
`files_sha256` hashes the UTF-8 encoding of
`json.dumps(inventory["files"], sort_keys=True, separators=(",", ":"))`.
Any file or policy change invalidates that approval. This is a bound review
receipt, not a cryptographic signature proving who reviewed it. Keep the actual
review evidence in the private archive.

Export benchmark diagnostics with explicit numeric field allowlists and reviewed
public task names. Omit authored option strings and fine-grained class, slot,
category and length distributions. Suppress outcome metrics for task cohorts
below 20, retaining their denominators; this is a disclosure policy rather than
differential privacy. Retain truthful overall scores, failed outcomes, quality
gates and limitations. Keep exact internal diagnostics and cohort identities in
the private evidence archive, not as hashed low-entropy labels in public files.

The release's `verification/PRIVACY.json` must distinguish static content checks
from trained-weight privacy. If membership inference, inversion or extraction
has not been measured, record `NOT_MEASURED` explicitly. A candidate-only API,
frozen backbone, LoRA, attribution and safetensors are not privacy guarantees.
Do not clear public weights for a confidentiality claim while that evidence and
the sensitive-source inventory are missing. A formal differential-privacy claim
requires an appropriate training mechanism and valid accounting; ordinary
gradient clipping does not provide one.

Use the canonical gate and publisher for every future model release:

```sh
.venv/bin/python scripts/package/model_release.py check artifacts/model-release \
  --inventory artifacts/reviewed-model-inventory.json
.venv/bin/python scripts/package/model_release.py publish artifacts/model-release \
  --inventory artifacts/reviewed-model-inventory.json \
  --repo-id teilomillet/MODEL --parent-commit REVIEWED_HUB_COMMIT
```

The publisher requires an existing private repository at the reviewed parent
revision. It rejects missing headers, added or changed files, links, bytecode,
unsupported code kinds, prohibited content, missing or stale privacy approval,
a public destination or a changed parent before upload.
It uploads only the frozen inventory and verifies remote privacy, file inventory,
sizes and hashes before reporting success. Keep staging under one writer until
verification finishes. Prepare Hugging Face storage rules before freezing the
inventory; a Hub-added rule is a changed file requiring reconciliation and review.

If verification fails after a commit, inspect the uploaded revision before
retrying. The tool does not automatically retry, change visibility, delete remote
files or create a repository. These commands are release checks and transport;
they do not authorize publication or establish model quality.

Git history cleanup is deferred until the v0.1 public release. Keep both GitHub
and Hugging Face private in the meantime. A later public release must audit the
entire resulting history and retained CI artifacts, including old model-report
revisions; a sanitized new commit does not erase earlier copies. Preserve a
private history archive before that separately reviewed cleanup.
