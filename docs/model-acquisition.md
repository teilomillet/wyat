# Obtain the supported Kai model

Download the full **Choice** artifact for
[`vllm-sr/Decision-2.0-Kai-0.6B`](https://huggingface.co/vllm-sr/Decision-2.0-Kai-0.6B/tree/cd49ea3813fd8ba0928a9a23ef6c9a0f2f0cd764),
revision `cd49ea3813fd8ba0928a9a23ef6c9a0f2f0cd764`. This pinned release already
has Wyat's required layout; no export, conversion, adapter merge or downloaded
model Python is needed. Dependency installation does not supply these weights.

Choose a new local model directory. This command explicitly downloads nine data
files, about 1.52 GB in total. It performs network I/O; `wyat.load` remains local.
Run it from your application project after [installing Wyat](install.md):

```sh
export WYAT_MODEL="$PWD/models/Decision-2.0-Kai-0.6B"
python -m pip install 'huggingface-hub==1.33.0'
hf download \
  vllm-sr/Decision-2.0-Kai-0.6B \
  LICENSE MODEL_MANIFEST.json config.json decision_config.json \
  decision_head.safetensors tokenizer.json tokenizer_config.json \
  backbone/config.json backbone/model.safetensors \
  --revision cd49ea3813fd8ba0928a9a23ef6c9a0f2f0cd764 \
  --local-dir "$WYAT_MODEL"
```

The [Hub download guide](https://huggingface.co/docs/huggingface_hub/guides/download)
describes revision pinning, explicit file selection and `--local-dir`. Keep the
revision fixed when repeating this command. Do not mix another release's files
into this directory.

The resulting layout is:

```text
Decision-2.0-Kai-0.6B/
├── LICENSE
├── MODEL_MANIFEST.json
├── config.json
├── decision_config.json
├── decision_head.safetensors
├── tokenizer.json
├── tokenizer_config.json
└── backbone/
    ├── config.json
    └── model.safetensors
```

The downloader also creates `.cache/huggingface/` metadata. It is not part of
inference. The command omits the repository's custom Python and Score offset
files; the ordinary loader admits Choice only.

## Verify before inference

Copy [verify.py](../examples/model/verify.py) from this checkout into your
application as `verify_kai.py`, then run:

```sh
python verify_kai.py "$WYAT_MODEL"
```

The verifier uses only the Python standard library. It first checks the
manifest's independently pinned SHA-256:
`ca7bf2cff1494a5bfc51f4271f52557da0f3ce3b6f0f6e7a2b4118cb4965e92e`.
It then checks every downloaded payload against that manifest and rejects mixed
optional tokenizer files. Missing or changed files fail before inference.
Wyat additionally validates the architecture, tensor storage, configured limits
and consumed file hashes when loading.

Continue with [load, decide and use the ID](install.md#acquire-and-load-a-trained-model).
Keep the loaded model across calls. CPU inference promotes supported storage to
FP32 and needs memory beyond the download's file size.

## Artifact identity and rights

Wyat's Apache-2.0 source license does not replace the model's license. Keep the
downloaded `LICENSE` with the artifact and review it before using or redistributing
weights. [NOTICE](../NOTICE) attributes the adapter to the vLLM Semantic Router
Team's pinned Decision 2.0 Kai release. Wyat's package contains no model weights.

The verifier establishes file identity for this pinned release. It does not
establish model quality, calibration, language coverage or suitability for an
application. Evaluate those properties on your own relevant cases.

If acquisition fails, retain its error and complete the missing download or
permission step. Do not substitute an adapter-only checkpoint or silently load a
different revision. No remote service is needed once the verified artifact exists.
