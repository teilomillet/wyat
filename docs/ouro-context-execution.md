# Ouro context execution (`wyat-ouro-decision-v2`)

A `wyat-ouro-decision-v2` artifact records the context execution its model was
trained and judged with, so wyat runs the model the same way. A v1 artifact
loads and scores exactly as before.

## What v2 adds to `ouro_manifest.json`

Two fields under `execution`, which are also part of the artifact's `model_id`:

```json
"context_prefill": {
  "chunk": 512,
  "single_chunk_max_tokens": 1024,
  "chunked_attention": "flash",
  "host_cache_above_tokens": 12288
},
"numerics": "deterministic-no-tf32-v1"
```

**The context prefill:**
- A context of at most `single_chunk_max_tokens` tokens takes the exact single pass, the v1 path.
- A longer context is prefilled in `chunk`-token pieces with `chunked_attention`:
  - `flash`: fused BF16, not MATH-exact;
  - `lean`: the MATH arithmetic, bit-identical to it.
- Above `host_cache_above_tokens`, the retained context K/V sits in pinned host memory and moves to the GPU one layer at a time. This applies on CUDA with shared-context batches, and the placement never changes a result. When the whole bank fits in free device memory with 3 GiB of headroom, a CUDA scorer keeps it resident on the GPU instead (one check per long context, after returning allocator-cached blocks to the driver); tight-memory devices keep the artifact's host placement. Either way the retained values are the same bf16 tensors, so the logits are identical.

**The numerics:** loading the artifact sets them for the whole process (`wyat.loading.ouro.apply_numerics`):
- TF32 off for matmuls and cuDNN;
- FP32 matmul precision `highest`;
- deterministic algorithms (cuBLAS workspace `:4096:8`), without filling uninitialized memory.

While it scores, the scorer restricts SDPA to the exact MATH kernel, except for the chunked context attention.

**Prefix reuse** (`cache.entries` > 0, `prefix_cache_size` > 0) is not part of this execution, and a v2 artifact refuses it.

## Writing one

```python
from wyat.loading.ouro import write_ouro_artifact

write_ouro_artifact(
    "artifacts/my-ouro-model",
    contract_prompt="canonical-id-description-prefix-isolated-option-v1",
    source_dir=base_dir, tokenizer_dir=base_dir, base_weights=base_dir / "model.safetensors",
    adapters=adapters, head=head, max_length=32768,
    context_prefill={"chunk": 512, "single_chunk_max_tokens": 1024,
                     "chunked_attention": "flash", "host_cache_above_tokens": 12288},
    numerics="deterministic-no-tf32-v1",
)
```

`context_prefill` and `numerics` go together; giving only one is refused.

## What is verified

On CPU, with the same tiny random Ouro model, adapters and head:
- **Single pass:** wyat's single-pass shared-context scoring is bit-identical to the reference implementation that trained these models, for contexts of one chunk.
- **One chunk:** a chunk at least as long as the context is bit-identical to the single pass.
- **`lean` chunks:** match it within `atol=2e-5`.

`tests/test_ouro_artifact.py` covers these, plus the manifest checks and the scorer's choice of prefill.

The `flash` path runs on CUDA only. GPU parity with the reference engine must be measured on the target artifact before it is relied on.

## Runtime acceleration (unchanged execution)

Short shared-context candidate batches (context at most 1,024 tokens, a single
score batch) may be replayed from CUDA graphs. A captured graph records exactly
the kernels the eager batch runs on fixed buffers; its first replay is compared
with the eager logits as FP32 bit patterns before the graph is used, and any
mismatch or capture error permanently falls back to eager scoring. Nothing in
the artifact's `execution` block changes: the same prefill, batching budget,
numerics and readout run, and the values are identical. Capture requires a
`PYTORCH_CUDA_ALLOC_CONF` without `expandable_segments`; elsewhere or on any
error the eager path runs unchanged.
