# Obtain the Wyat model

Wyat's model is [wyat-1-600m](https://huggingface.co/teilomillet/wyat-1-600m),
published on Hugging Face. Installing Wyat does not include it. The release
already has the layout Wyat loads: no export, conversion, adapter merge or
downloaded model code is needed.

Load it by name; the first load downloads it:

```python
import wyat

model = wyat.load("teilomillet/wyat-1-600m")
```

The first call downloads about 1.5 GB into the Wyat cache,
`~/.cache/wyat/wyat-1-600m` (or under `$XDG_CACHE_HOME`), checks every file
against SHA-256 digests pinned in your installed version of Wyat, and then
loads it. Later calls check the cached files' digests locally and need no
network. The CLI accepts the same name: `uv run wyat serve
teilomillet/wyat-1-600m`.

To download ahead of time, for example before going offline, run this from your
application project after [installing Wyat](install.md). The core install is
enough:

```sh
uv run wyat download
```

```text
Wyat model teilomillet/wyat-1-600m @ … -> ~/.cache/wyat/wyat-1-600m
Downloading about 1.5 GB; files already verified are kept.
  verified  LICENSE
  …
  verified  model/tokenizer_config.json
Done. Review the model's LICENSE and NOTICE before redistributing it.
Next: wyat.load("…/.cache/wyat/wyat-1-600m/model") in Python, or 'wyat serve …'.
…/.cache/wyat/wyat-1-600m/model
```

It prints the absolute model path last; that path is the only output on stdout.
Pass a directory as an argument to place the release elsewhere, then load that
directory's `model` path. If a download stops, run the same command, or the same
`wyat.load` call, again: verified files are kept, and only missing or changed
files are fetched. Downloading this release is Wyat's only network access, and
only that exact name triggers it.

The resulting layout is:

```text
wyat-1-600m/
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

Download into a new or previously used Wyat directory. The download refuses a
`model/` directory that also contains tokenizer files from another release
(`special_tokens_map.json`, `added_tokens.json`, `vocab.json`, `merges.txt`).

## Verify a copy made another way

`wyat download` already verified its files. For a copy obtained another way,
such as from a colleague or an offline mirror, copy
[verify.py](../examples/model/verify.py) into your application as
`verify_wyat.py` and run it on the `model` directory:

```sh
uv run python verify_wyat.py ~/.cache/wyat/wyat-1-600m/model
```

The verifier uses only the Python standard library and carries the same release
pins as `wyat download` in the same Wyat version. It first checks the manifest's
pinned SHA-256, then checks every payload against that manifest and rejects
mixed optional tokenizer files.
Missing or changed files fail before inference. Wyat additionally validates the
architecture, tensor storage, configured limits and consumed file hashes when
loading.

Continue with [load, decide and use the ID](install.md#acquire-and-load-a-trained-model).
Keep the loaded model across calls. CPU inference promotes supported storage to
FP32 and needs memory beyond the download's file size.

## Artifact identity and rights

Wyat's Apache-2.0 source license does not replace the model's license. Keep the
downloaded `LICENSE` and `NOTICE` with the artifact and review them before using
or redistributing weights. Wyat's source [NOTICE](../NOTICE) records
third-party attributions for the code; the model's own NOTICE comes with the
download. Wyat's package contains no model weights.

Verification establishes file identity for the pinned release. It does not
establish model quality, calibration, language coverage or suitability for an
application. Evaluate those properties on your own relevant cases.

If the download fails, the message says what to do next: check the connection,
or run the command again. Do not substitute partial model files or load files
that failed verification. No remote service is needed once the verified
model exists.
