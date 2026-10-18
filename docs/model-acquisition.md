# Obtain the wyat model

The upcoming v0.1.1 targets
[wyat-1-800m](https://huggingface.co/teilomillet/wyat-1-800m), which is also the
library default. Its named download is not pinned or available yet.
`wyat.load()` and `wyat download --model teilomillet/wyat-1-800m` report that
pending release. Installing wyat includes no weights.

## Load a prepared 800M artifact

If you have a prepared 800M artifact, load its local model directory with the
PyTorch inference extra installed:

```python
import wyat

model = wyat.load("/path/to/wyat-1-800m/model")
# For a compatible CUDA device: device="cuda".
```

The directory passed to `load` contains `fork_manifest.json`. 800M uses the
`wyat-option-fork-v1` format with a Qwen3.5 backbone, separate adapters and a
scalar decision head. Its required layout is:

```text
model/
├── fork_manifest.json
├── adapters.safetensors
├── head.safetensors
├── tokenizer.json
├── tokenizer_config.json
└── base/
    ├── config.json
    └── model.safetensors
```

An artifact may also contain supported tokenizer files declared in its
manifest. The loader checks file hashes, manifest structure, model identity,
architecture, adapter inventory and tensor shapes before inference. It uses
local files without executing downloaded Python code, converting the model or
merging its adapters. A local manifest establishes internal consistency; until
the release is pinned, it does not authenticate a copy as the public 800M
release.

800M supports PyTorch on CPU by default or a compatible CUDA device. MLX does
not support this format. See [PyTorch inference](torch-inference.md) for runtime
settings. Local loading needs no network; retain the artifact's own license
and notice files alongside the weights.

## Legacy 600M compatibility

The explicit name `teilomillet/wyat-1-600m` remains supported for applications
that already use the legacy `wyat-native-full-v1` artifact. It is a separate
model with its own pinned revision and file hashes; it is not an 800M download
or conversion. Repository access is required until that artifact is public.

For those applications, `wyat.load("teilomillet/wyat-1-600m")` downloads about
2.44 GB of FP32 weights into `~/.cache/wyat/wyat-1-600m` (or under
`$XDG_CACHE_HOME`), verifies each file against the installed package's pins,
and loads it. Later loads verify and reuse the cache. To fetch ahead of time:

```sh
uv run wyat download --model teilomillet/wyat-1-600m
```

The command prints the absolute `model/` directory path last, as its only
stdout output. Passing a destination directory puts the release there. An
interrupted download can resume: verified files are kept, and missing or
changed files are fetched again. Mixed optional tokenizer files are refused.
Only supported release names trigger acquisition; other strings are local
paths.

The standalone [verify.py](../examples/model/verify.py) is **600M-specific**.
For an offline copy of that legacy artifact, copy it into your application as
`verify_wyat.py` and run:

```sh
uv run python verify_wyat.py ~/.cache/wyat/wyat-1-600m/model
```

It uses the Python standard library and the same 600M release pins as the
package. It checks the manifest's pinned SHA-256, payload hashes and mixed
optional tokenizer files before inference. It cannot verify an 800M artifact.

## Artifact identity and rights

wyat's Apache-2.0 source license does not replace the model's license. Keep the
artifact's `LICENSE` and `NOTICE` and review them before using or redistributing
weights. The source [NOTICE](../NOTICE) records third-party code attributions.

File verification establishes identity or internal consistency within the
stated scope. It does not establish model quality, calibration, language
coverage or suitability for an application. Evaluate those properties on your
own relevant cases. CPU inference uses FP32 and needs memory beyond the files'
size.

If a download or validation fails, follow its error message. Do not substitute
partial model files or load files that failed verification. Continue with
[load, decide and use the ID](install.md#acquire-and-load-a-trained-model), and
keep the loaded model across requests.
