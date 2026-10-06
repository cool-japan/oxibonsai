# OxiBonsai Model Support Matrix

Which models OxiBonsai loads, in which on-disk formats, and how the formats differ
byte for byte. For flags and environment variables see [`docs/CLI.md`](CLI.md); for
the image model see [`docs/IMAGEN.md`](IMAGEN.md); for the project overview see
[`README.md`](../README.md).

## Contents

- [At a glance](#at-a-glance)
- [Bonsai 2 and the 27B family (`qwen35` hybrid)](#bonsai-2-and-the-27b-family-qwen35-hybrid)
- [The dense Qwen3 family (`qwen3`)](#the-dense-qwen3-family-qwen3)
- [On-disk block layouts](#on-disk-block-layouts)
- [ggml type id 42: three layouts under one id](#ggml-type-id-42-three-layouts-under-one-id)
- [The Hadamard contract (Bonsai 2 files)](#the-hadamard-contract-bonsai-2-files)
- [Where the files live](#where-the-files-live)
- [What has been validated](#what-has-been-validated)

## At a glance

| Family | `general.architecture` | Weight formats | Backends |
|--------|------------------------|----------------|----------|
| Bonsai 2 27B, previous-generation 27B | `qwen35` (hybrid: 48 Gated-DeltaNet + 16 full-attention layers) | `PTQ1_0`, `PQ2_0`, `Q2_0` (id 42), `Q1_0` (id 41) | CPU, Metal (hybrid runner), CUDA\* |
| Bonsai-8B | `qwen3` | `Q1_0_g128` (id 41) | CPU, Metal, CUDA\* |
| Ternary-Bonsai-8B / 4B / 1.7B | `qwen3` | `TQ2_0_g128` (id 42, our own layout) | CPU, Metal, CUDA\* |
| Bonsai-Image (FLUX.2 Klein DiT) | `bonsai-image` | `TQ2_0_g128` + BF16 | CPU, Metal, CUDA\* |
| Generic llama.cpp-style GGUFs | `qwen3` (dense) | `Q4_0`, `Q8_0`, `Q2_K`..`Q8_K`, FP8, `F16`/`BF16`/`F32` | CPU; Metal GEMV for `Q4_0`/`Q8_0`/K-quants |

\* The CUDA backend is **unvalidated in this release**: it was written and reviewed on a
host with no NVIDIA hardware, so it compiles but has never been run. The `cuda`
(scirs2) feature is a CPU fallback, not a GPU build; use `native-cuda` for the NVIDIA
path and expect to validate it yourself.

## Bonsai 2 and the 27B family (`qwen35` hybrid)

Every file below is a `qwen35` hybrid: 64 layers, hidden 5120, FFN 17 408, vocabulary
248 320, context 262 144. Layer `i` is full attention when `(i + 1) % 4 == 0` (16
layers: 3, 7, ..., 63) and Gated-DeltaNet linear attention otherwise (48 layers). The
Bonsai 2 files are stored in a Hadamard-rotated basis (see
[the Hadamard contract](#the-hadamard-contract-bonsai-2-files)); the previous-generation
files are not.

| File | ggml type id | Block | Size (bytes) | Hadamard | Repository |
|------|--------------|-------|-------------:|:--------:|------------|
| `Ternary-Bonsai-2-27B-PTQ1_0.gguf` | **143** `PTQ1_0` | 28 B / 128 weights, `{qs[24], qh[2], d}` (`d` **last**) | 5 946 648 928 | yes | `prism-ml/Ternary-Bonsai-2-27B-gguf` |
| `Ternary-Bonsai-2-27B-PQ2_0.gguf` | **142** `PQ2_0` | 34 B / 128 weights, `{d, qs[32]}` (`d` **first**) | 7 206 168 928 | yes | `prism-ml/Ternary-Bonsai-2-27B-gguf` |
| `Ternary-Bonsai-2-27B-Q2_0-prism-fork-required.gguf` | **42** `Q2_0` (mainline) | 18 B / **64** weights, `{d, qs[16]}` (`d` first) | 7 626 008 928 | yes | `prism-ml/Ternary-Bonsai-2-27B-gguf-dev` |
| `Ternary-Bonsai-27B-PQ2_0.gguf` (previous generation) | **142** `PQ2_0` | 34 B / 128, `d` first | 7 165 121 600 | no | `prism-ml/Ternary-Bonsai-27B-gguf` |
| `Ternary-Bonsai-27B-Q2_0.gguf` (previous generation) | **42** (legacy PrismML g128) | 34 B / 128, `d` first | 7 165 121 600 | no | `prism-ml/Ternary-Bonsai-27B-gguf` |
| `Bonsai-27B-Q1_0.gguf` (previous generation, 1-bit) | **41** `Q1_0` | 18 B / 128, `d` first | 3 803 452 480 | no | `prism-ml/Bonsai-27B-gguf` |
| `Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf` (vision projector, `clip`) | `Q8_0` + `F32`/`F16` | `qwen3vl_merger`, 27 ViT blocks | 629 246 976 | n/a | `prism-ml/Ternary-Bonsai-2-27B-gguf` |

The Bonsai 2 family lives in **two** HuggingFace repositories: the main one holds the
`PTQ1_0`, `PQ2_0` and mmproj files, and the `-dev` repository holds the `Q2_0` file
that only the PrismML llama.cpp fork reads. `oxibonsai pull` knows both
(`OXI_BONSAI2_REPO` / `OXI_BONSAI2_DEV_REPO` override them).

Sampling defaults come from the file's own `general.sampling.*` keys (temperature 1.0,
top-p 0.95, top-k 20); the chat template comes from `tokenizer.chat_template`; the end
of turn is token 248046 (`<|im_end|>`).

**Memory.** The KV cache costs 65 536 bytes per position (16 full-attention layers x 4
KV heads x 256 head dimension x 2 (K and V) x 2 bytes of `f16`), so the default
8192-position window is 512 MiB. The 48 linear layers add a fixed recurrent state of
156 893 184 bytes per sequence (150 994 944 B of Gated-DeltaNet matrices plus 5 898 240
B of convolution windows, about 150 MiB). On the 24 GiB reference host the RAM-derived
context ceiling for `PQ2_0` is 178 176 tokens; `oxibonsai info` prints the plan and the
CLI refuses a request that does not fit.

**Served model id.** `/v1/models` reports the file stem when `general.name` is a
placeholder: the Bonsai 2 27B files carry `general.name = "Hf"`, so the server lists
`Ternary-Bonsai-2-27B-PQ2_0` instead (`/admin` shows both).

### Vision

`--mmproj Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf` plus `--image` (CLI and server) enable
image input for a Bonsai 2 `qwen35` language file (the projector is paired with it).
Decoding images needs no extra tool: PNG and JPEG are decoded in Pure Rust. The vision
tower runs on the GPU on a Metal host (0.87 GiB resident with exact `Q8_0` weights) and
on the CPU otherwise (1.71 GiB of `f32` weights, 23-30 s for a 768x768 image against
2.6-2.7 s on Metal). An image reference is a PNG or JPEG file, a base64 `data:` URI, a
`file://` path inside the server's `--media-path`, or — only with
`--allow-image-url-fetch` — an `http(s)` URL fetched from public addresses (and the
hosts `--image-url-allow-host` names); see [`CLI.md`](CLI.md#remote-images).

## The dense Qwen3 family (`qwen3`)

| Model | Layers | Hidden | FFN | Heads / KV heads | Vocabulary | Context | Format |
|-------|-------:|-------:|----:|-----------------:|-----------:|--------:|--------|
| Bonsai-8B | 36 | 4096 | 12 288 | 32 / 8 | 151 669 | 65 536 (YaRN x4 over 16 384) | `Q1_0_g128` (1.16 GB) |
| Ternary-Bonsai-8B | 36 | 4096 | 12 288 | 32 / 8 | 151 669 | 65 536 (no rope scaling in the file) | `TQ2_0_g128` (2.18 GB) |
| Ternary-Bonsai-4B | 24 | 2560 | 6912 | 20 / 4 | 151 936 | 65 536 | `TQ2_0_g128` |
| Ternary-Bonsai-1.7B | 28 | 2048 | 6144 | 16 / 8 | 151 669 | 32 768 | `TQ2_0_g128` (540 MB) |

The head dimension is 128 everywhere. These are the values in the real
`Bonsai-8B.gguf`, `Ternary-Bonsai-8B.gguf` and `Ternary-Bonsai-1.7B.gguf` headers.
**The built-in configurations were wrong before this release and matched no shipped
file:** `Qwen3Config::bonsai_1_7b()` claimed 16 layers, hidden 1536, FFN 4096, 12 heads,
2 KV heads, vocabulary 151 936 and context 65 536; `bonsai_8b()` claimed FFN 14 336 and
vocabulary 151 936 and did not apply the file's YaRN declaration. All of them now equal
the real headers, and `ModelVariant::from_config` detects the 1.7B from (28 layers,
hidden 2048). The 4B figures come from the published architecture; no 4B file was
available to verify them against.

**YaRN.** A model that declares `rope.scaling.type = yarn` (Bonsai-8B: factor 4,
original context 16 384) decodes with YaRN applied at every position, matching
llama.cpp, so greedy output differs from OxiBonsai 0.2.3 and earlier on such models;
the new output is byte-identical to the PrismML llama.cpp fork. `--rope-scaling off`
(or `[model].rope_scaling = "off"`) reproduces the old behaviour, `auto` (the default)
follows the file, and `on` on a model without scaling keys is refused naming
`<arch>.rope.scaling.*`.

The ternary files are produced by `oxibonsai convert` (safetensors or ONNX
`MatMulNBits` input) or `scripts/download_ternary.sh`; `Bonsai-8B.gguf` and the 27B
files download with `oxibonsai pull`.

## On-disk block layouts

All multi-byte fields are little-endian; `d` is an IEEE half (FP16) scale.

| Format | ggml id | Weights / block | Bytes / block | Field order | Codes |
|--------|--------:|----------------:|--------------:|-------------|-------|
| `Q1_0_g128` | 41 | 128 | 18 | `d` (2 B), then 16 B of sign bits | bit 1 = +scale, bit 0 = -scale |
| `TQ2_0_g128` (OxiBonsai's own writer) | 42 | 128 | 34 | `qs[32]` (32 B), then `d` | 2 bits per weight, LSB first: `00` -1, `01` 0, `10` +1, `11` 0 |
| `Q2_0` legacy PrismML g128 | 42 | 128 | 34 | `d`, then `qs[32]` | as `PQ2_0` |
| `PQ2_0` | **142** | 128 | 34 | `d`, then `qs[32]` | 2 bits LSB first: `00` -1, `01` 0, `10` +1, `11` +2 (ternary data never emits `11`) |
| `Q2_0` mainline `block_q2_0` | 42 | **64** | 18 | `d`, then `qs[16]` | 2 bits LSB first |
| `PTQ1_0` | **143** | 128 | 28 | `qs[24]`, `qh[2]`, then `d` (**last**) | base-3 digits ("trits"): 5 per byte of `qs` (120 weights; the first 16 bytes, then the remaining 8) and 4 per byte of `qh` (8 weights); the element order inside a block is interleaved |

Two things to keep straight: `PQ2_0` stores `d` first while `TQ2_0_g128` stores it
last, and the two 2-bit codecs differ at `0b11` (`PQ2_0` decodes it as +2, `TQ2_0_g128`
as 0). A reader that treats one layout as the other produces plausible-looking garbage,
which is why id 42 is never guessed.

## ggml type id 42: three layouts under one id

Three incompatible layouts ship under tensor type id 42:

| File family | Group | Bytes | Byte order | `general.quantization_version` |
|-------------|------:|------:|------------|-------------------------------|
| `Ternary-Bonsai-{1.7B,8B}` (OxiBonsai's own writer) | 128 | 34 | `qs` first | the **string** `"TQ2_0_G128"` |
| `Ternary-Bonsai-27B-Q2_0` (PrismML previous generation) | 128 | 34 | `d` first | `u32 2` |
| `Ternary-Bonsai-2-27B-Q2_0` (mainline `block_q2_0`) | 64 | 18 | `d` first | `u32 2` |

`general.file_type` is **not** a discriminator (the previous-generation g128 file and the
g64 file both report 41), and neither is the offset delta between two tensors (GGUF pads
offsets). The loader settles the reading in two steps and never guesses:

1. **Group size** by replaying llama.cpp's own offset invariant: walk the tensors in
   offset order, accumulate each tensor's padded byte size for each candidate geometry
   (128 / 34 B and 64 / 18 B), and keep the candidates that reproduce every declared
   offset. On the real files the two candidates differ by hundreds of megabytes, so
   exactly one survives.
2. **Byte order** (`d` first or `qs` first) cannot be read from offsets, since both give
   identical sizes. The legacy string tag settles it for OxiBonsai's own files; for
   every other file a structural check of real block bytes does (a ternary checkpoint
   never contains the reserved `0b11` code, and its scale is a finite non-negative
   half). Without block bytes to look at, the resolver in `oxibonsai-core` reports an
   ambiguous quantization type instead of picking one. When the model loader's sample
   is degenerate (all zeros or too small, which is what the synthetic unit-test
   fixtures look like) it falls back to the legacy `qs`-first reading and logs a
   warning that names the override below; offsets that are internally inconsistent, or
   block bytes that contradict a declared legacy tag, stay hard errors.

**`OXI_FORCE_Q2_LAYOUT`** is the escape hatch. It is read by every loader (dense and
hybrid) and applied before the automatic evidence:

| Value (case-insensitive) | Forces |
|--------------------------|--------|
| `d-first`, `dfirst`, `d_first`, `pq2`, `pq2_0` | group 128, `d` first (PrismML) |
| `qs-first`, `qsfirst`, `qs_first`, `tq2`, `tq2_0`, `legacy` | group 128, `qs` first (OxiBonsai) |
| `g64`, `q2_0_g64`, `q2_0g64` | group 64, mainline `block_q2_0` |
| empty / unset | automatic |

Any other value is rejected with an error that lists the accepted spellings. Ids 142,
143, 41 and the string-tagged files never need it.

## The Hadamard contract (Bonsai 2 files)

The Bonsai 2 language GGUFs carry `prism.hadamard.*` metadata (version 1, block size
1024, the normalized Sylvester-Walsh-Hadamard transform over the input's last
dimension, explicit `+-1` sign vectors for the widths 5120, 6144 and 17 408). Every
folded weight `W` (401 matrices: `output.weight` and every attention, SSM, gate, up and
down projection) was multiplied in the rotated basis, so the forward pass computes
`y = W (H (x * signs))` with a blockwise 1024-point transform; `token_embd.weight`
is stored rotated and the lookup is followed by the inverse transform. Norms,
`ssm_alpha` / `ssm_beta`, the convolution kernels, `ssm_a` and `ssm_dt` are not
folded. The loader validates the metadata the way llama.cpp does and refuses a file
whose contract does not hold. Embeddings returned by `/v1/embeddings` live in the
model's own un-rotated basis.

## Where the files live

| Artefact | Source |
|----------|--------|
| Bonsai 2 27B `PTQ1_0`, `PQ2_0`, mmproj | `prism-ml/Ternary-Bonsai-2-27B-gguf` |
| Bonsai 2 27B `Q2_0` (fork-required) | `prism-ml/Ternary-Bonsai-2-27B-gguf-dev` |
| Previous-generation 27B | `prism-ml/Bonsai-27B-gguf`, `prism-ml/Ternary-Bonsai-27B-gguf` |
| Bonsai-8B | `prism-ml/Bonsai-8B-gguf` |
| Ternary Bonsai 8B / 4B / 1.7B (unpacked safetensors, converted locally) | the `prism-ml` Ternary-Bonsai repositories, via `scripts/download_ternary.sh` |
| Bonsai-Image DiT, text encoder, VAE | `prism-ml/bonsai-image-ternary-4B-mlx-2bit` ([`docs/IMAGEN.md`](IMAGEN.md)) |

`oxibonsai pull <name>` downloads the named GGUFs (`bonsai2-27b` with `--band
pq2|ptq1` and `--vision`, `bonsai2-27b-ptq1_0`, `bonsai2-27b-pq2_0`,
`bonsai2-27b-mmproj`, `bonsai2-27b-q2_0`, `bonsai-27b-q1_0`, `ternary-bonsai-27b-pq2_0`,
`ternary-bonsai-27b-q2_0`, `bonsai-8b`). A named download is verified fail-closed:
GGUF structure (magic, expected `general.architecture`, and `prism.hadamard.version ==
1` for Bonsai 2 language files), then the exact byte size, then the SHA-256 compiled
into the binary, then a cross-check against `scripts/checksums.sha256`; a conflict
between the authorities aborts. `bonsai-8b` verifies against the current upstream
digest `284a335aa3fb2ced3b1b01fcb40b08aa783e3b70832767f0dd2e3fdfa134bd54` (an upstream
re-upload of the same 1 158 654 496 bytes) and accepts the earlier known-good
`ead25897bc034fa52569d0c6d054ce38216f95db09900c8add8f6bbfb370cff1` with a warning.

**Which files have a verified hash.** `oxibonsai pull` has the upstream SHA-256 of
every file it can name compiled in (the HuggingFace LFS object digests), so each named
download is hash-verified. `scripts/checksums.sha256` is the manifest the download
scripts and a manual `shasum -a 256 -c scripts/checksums.sha256 --ignore-missing` use,
and it is less complete:

| Entry | State in `scripts/checksums.sha256` |
|-------|-------------------------------------|
| `Bonsai-8B.gguf`, `tokenizer.json` | hash recorded; a mismatch aborts the download |
| `Ternary-Bonsai-2-27B-PTQ1_0.gguf`, `-PQ2_0.gguf`, `-mmproj-Q8_0.gguf` | hash computed from the real files; a mismatch aborts the download |
| `Ternary-Bonsai-1.7B.gguf`, `Ternary-Bonsai-8B.gguf` | hash recorded, but advisory: the files are converted locally, so a converter change can move the bytes and a mismatch is a warning |
| `Ternary-Bonsai-2-27B-Q2_0-prism-fork-required.gguf`, `Bonsai-27B-Q1_0.gguf`, `Ternary-Bonsai-27B-PQ2_0.gguf`, `Ternary-Bonsai-27B-Q2_0.gguf` | explicit `MISSING` placeholder, no hash invented: the payloads have not been hashed locally, so only `oxibonsai pull` (embedded digest) verifies them |
| `Ternary-Bonsai-4B.gguf` | no entry |

## What has been validated

- **Greedy parity with the reference implementation.** On the two 27B files present on
  the reference host (`PQ2_0`, `PTQ1_0`) greedy token ids equal the PrismML llama.cpp
  fork on three prompts per format on the CPU path, on Metal and through the server and
  CLI; the vision path reproduces both reference prompts token for token. The other
  four language files are covered by header parsing, loader and layout tests on
  synthetic fixtures, not by a real-model run.
- **Dense models.** CPU and Metal produce byte-identical greedy output on the real
  `Ternary-Bonsai-1.7B`; the real `Bonsai-8B` YaRN output is byte-identical to the
  fork.
- **Throughput.** Measured on an M3 with 24 GB, as ranges because the host was under
  background load: Bonsai 2 27B decodes at 6.9-8.4 tok/s on Metal and 1.0-1.3 tok/s on
  the CPU path; see the [README](../README.md#measured-throughput) for the table and
  its caveats.
- **Not validated:** CUDA (see above), the eval logit tasks against a real model, and
  the AVX-512 VNNI INT8 tier on VNNI hardware.
