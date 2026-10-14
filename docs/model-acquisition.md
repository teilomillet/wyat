# Obtain a supported Wyat model

Download the full **Wyat** artifact for
[`teilomillet/wyat-v0.2a`](https://huggingface.co/teilomillet/wyat-v0.2a/tree/0a7b39ccd5521e010c18bae6696bc3776b216767),
revision `0a7b39ccd5521e010c18bae6696bc3776b216767`. This pinned release already
has Wyat's required layout; no export, conversion, adapter merge or downloaded
model Python is needed. Dependency installation does not supply these weights.

Choose a new local model directory. This command explicitly downloads the model
bundle files, about 1.53 GB in total. It performs network I/O; `wyat.load` remains local.
Run it from your application project after [installing Wyat](install.md):

```sh
export WYAT_REPO="$PWD/models/wyat-v0.2a"
export WYAT_MODEL="$WYAT_REPO/model"
uvx --from 'huggingface-hub==1.33.0' hf download \
  teilomillet/wyat-v0.2a \
  LICENSE NOTICE \
  model/native_manifest.json model/decision_head.safetensors \
  model/native-adapters.safetensors model/tokenizer.json \
  model/tokenizer_config.json model/backbone/config.json \
  model/backbone/model.safetensors \
  --revision 0a7b39ccd5521e010c18bae6696bc3776b216767 \
  --local-dir "$WYAT_REPO"
```

The [Hub download guide](https://huggingface.co/docs/huggingface_hub/guides/download)
describes revision pinning, explicit file selection and `--local-dir`. Keep the
revision fixed when repeating this command. Do not mix another release's files
into this directory.

The resulting layout is:

```text
wyat-v0.2a/
├── LICENSE
├── NOTICE
└── model/
    ├── native_manifest.json
    ├── decision_head.safetensors
    ├── native-adapters.safetensors
    ├── tokenizer.json
    ├── tokenizer_config.json
    └── backbone/
        ├── config.json
        └── model.safetensors
```

The downloader also creates `.cache/huggingface/` metadata. It is not part of
inference.

## Verify before inference

Copy [verify.py](../examples/model/verify.py) from this checkout into your
application as `verify_wyat.py`, then run:

```sh
uv run python verify_wyat.py "$WYAT_MODEL"
```

The verifier uses only the Python standard library. It first checks the
manifest's independently pinned SHA-256:
`3db77b2bafdaf536a1398327df08a9b79321e67e3c5526f7f44fbd4a96fd0ffa`.
It then checks every downloaded payload against that manifest and rejects mixed
optional tokenizer files. Missing or changed files fail before inference.
Wyat additionally validates the architecture, tensor storage, configured limits
and consumed file hashes when loading.

Continue with [load, decide and use the ID](install.md#acquire-and-load-a-trained-model).
Keep the loaded model across calls. CPU inference promotes supported storage to
FP32 and needs memory beyond the download's file size.

## Artifact identity and rights

Wyat's Apache-2.0 source license does not replace the model's license. Keep the
downloaded `LICENSE` and `NOTICE` with the artifact and review them before using
or redistributing weights. [NOTICE](../NOTICE) records third-party attributions.
Wyat's package contains no model weights.

The verifier establishes file identity for this pinned release. It does not
establish model quality, calibration, language coverage or suitability for an
application. Evaluate those properties on your own relevant cases.

If acquisition fails, retain its error and complete the missing download or
permission step. Do not substitute an adapter-only checkpoint or silently load a
different revision. No remote service is needed once the verified artifact exists.
