# Obtain the wyat model

The default model is
[wyat-1-800m](https://huggingface.co/teilomillet/wyat-1-800m). Installing wyat
includes no weights. The first load downloads its immutable release and
verifies every file before inference:

```python
import wyat

model = wyat.load("teilomillet/wyat-1-800m")  # also the default for wyat.load()
# For a compatible CUDA device: device="cuda".
```

Install the [PyTorch inference extra](install.md) to load the model. 800M runs
on CPU by default or a compatible CUDA device; MLX does not support this
format. Inference itself needs no network or downloaded Python code.

## Download and reuse the verified cache

The download is about **1.82 GB**: the backbone is stored in BF16, with FP32
adapters and decision head. The package pins the immutable Hub revision,
manifest and license hashes in [`wyat/download.py`](../wyat/download.py).
Every payload file is checked against that pinned manifest before loading.
Later loads verify and reuse the files in `~/.cache/wyat/wyat-1-800m` (or under
`$XDG_CACHE_HOME`) without network access.

To fetch ahead of time, for example before going offline, the core CLI is
enough:

```sh
uv run wyat download
# Equivalent: uv run wyat download --model teilomillet/wyat-1-800m
```

The command prints the absolute `model/` directory path last, as its only
stdout output. Pass a destination directory to place the release elsewhere:

```sh
uv run wyat download /path/to/wyat-1-800m
```

An interrupted download can resume: verified files are kept, and missing or
changed files are fetched again. Only supported release names trigger
acquisition; other strings are local directory paths.

## Use an existing local artifact

The released artifact has this layout, including all eight payload files:

```text
wyat-1-800m/
├── LICENSE
├── NOTICE
└── model/
    ├── fork_manifest.json
    ├── adapters.safetensors
    ├── head.safetensors
    ├── tokenizer.json
    ├── tokenizer_config.json
    ├── vocab.json
    ├── merges.txt
    └── base/
        ├── config.json
        └── model.safetensors
```

Pass the directory containing `fork_manifest.json` to `load`:

```python
model = wyat.load("/path/to/wyat-1-800m/model")
```

800M uses the `wyat-option-fork-v1` format with a Qwen3.5 backbone, separate
adapters and a scalar decision head. The loader checks file hashes, manifest
structure, model identity, architecture, adapter inventory and tensor shapes
before inference. No conversion or adapter merge is required. For a copy made
another way, local validation establishes internal consistency; compare it
with the package's release pins to authenticate it as the same public release.
Retain its license and notice files alongside the weights. See
[PyTorch inference](torch-inference.md) for runtime settings.

## Legacy 600M compatibility

The explicit name `teilomillet/wyat-1-600m` remains supported for applications
that already use the legacy `wyat-native-full-v1` artifact. It is a separate
model with its own pinned revision and file hashes; it is not an 800M download
or conversion. Access to its private Hugging Face repository is required.

For those applications, `wyat.load("teilomillet/wyat-1-600m")` downloads about
2.44 GB of FP32 weights into `~/.cache/wyat/wyat-1-600m` (or under
`$XDG_CACHE_HOME`), verifies each file against the installed package's pins,
and loads it. Later loads verify and reuse the cache. To fetch ahead of time:

```sh
uv run wyat download --model teilomillet/wyat-1-600m
```

This legacy download uses the same cache verification and resumable behavior
described above. Mixed optional tokenizer files are refused.

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
The 800M trained adapters and decision head permit non-commercial use and
redistribution with attribution; commercial use is not granted. Its upstream
backbone, tokenizer and runtime retain Apache-2.0 terms.

File verification establishes identity or internal consistency within the
stated scope. It does not establish model quality, calibration, language
coverage or suitability for an application. Evaluate those properties on your
own relevant cases. CPU inference uses FP32 and needs memory beyond the files'
size.

If a download or validation fails, follow its error message. Do not substitute
partial model files or load files that failed verification. Continue with
[load, decide and use the ID](install.md#acquire-and-load-a-trained-model), and
keep the loaded model across requests.
