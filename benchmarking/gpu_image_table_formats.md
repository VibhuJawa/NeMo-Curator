# GPU image-table format benchmark

This benchmark compares the two requested single-GPU workflows on the same
physical image cohort and the same deterministic random row offsets:

1. Parquet → `cudf.read_parquet` → CUDA JPEG decode → CUDA
   variance-of-Laplacian → GPU filter → `cudf.DataFrame.to_parquet`.
2. Lance fragment random take → zero-copy `large_binary`/`large_string` buffer
   view → Arrow-to-cuDF transfer → CUDA JPEG decode → CUDA
   variance-of-Laplacian → GPU filter → Curator `LanceWriter` fragment write
   and checkpointed Lance commit.

The format-specific reader, image processing, and writer are fused into one
Curator GPU stage. A stage boundary would serialize device data through Ray's
object store, so it would not measure the intended workflow. The fused stage
still records a separate duration for every operation.

## Fairness and sampling

Preparation opens the pinned source table at version 4, selects the fragment
nearest the median physical row count (or `--fragment-id`), and copies that
complete fragment into matched Lance and Parquet inputs. Preparation is outside
all timed trials and is reusable through an identity-checked marker.

`ceil(fragment_rows * sample_fraction)` offsets are sampled without replacement
from a fixed seed. The list is sorted only after sampling. Both arms use exactly
that list. Parquet reads only the row groups intersecting the offsets and then
gathers the requested rows in cuDF; Lance directly takes the offsets. The
reported `read_amplification_rows` makes this layout difference explicit.

One persistent GPU actor handles the complete fraction schedule for a storage
backend. Its `setup()` initializes cuDF, CUDA, CuPy, TorchVision/nvJPEG, and a
cached Laplacian kernel once. One unmeasured warmup is then run per format at
the smallest fraction. Only after both warmups finish are three trials at each
of 10%, 20%, 40%, 80%, and 100% recorded. Arm order alternates by trial. The
actor task index must remain contiguous, and output row coordinates and
blur-score sums must match between formats in every trial.

`runtime_setup_s`, warmup time, pipeline wall time, and runner wall time are
reported separately. None is included in measured `end_to_end_s`. All five
fractions share one CUDA/runtime initialization per storage backend.

## GPU-residency contract

For both arms the complete selected table, including encoded image bytes,
resides in cuDF device memory through annotation and filtering. Lance storage
remains Arrow `large_binary`. cuDF 26.6 rejects Arrow `large_binary` in
`DataFrame.from_arrow`, but accepts `large_string`, so the Lance arm creates a
zero-copy Arrow type view over the same offsets and JPEG value buffer before
the cuDF transfer. No UTF-8 conversion or textual accessor is used. At the
required Lance Arrow writer boundary, the filtered JPEG bytes are copied back
and reconstructed as Arrow `large_binary`.

The current APIs still impose two explicit, measured host boundaries:

- Lance exposes Arrow host tables for read and write, measured by
  `arrow_to_device_s` and `writer_prepare_s`.
- TorchVision 0.25 rejects CUDA tensors containing encoded JPEG bitstreams. The
  cuDF image column therefore remains on GPU while one compressed-byte copy is
  made for nvJPEG input, measured by `encoded_device_to_host_s`. Decoded pixels,
  blur scores, threshold mask, and filter gather remain on GPU.

Only the tiny Curator task specification and result metrics cross Ray. Image
tables and image payloads never enter Ray's object store.

## Running through `benchmarking/run.py`

Use a GPU environment containing the Curator image, Lance, and cuDF extras:

```bash
source /path/to/gpu-lance-venv/bin/activate
export PYTHONPATH="$PWD"
python benchmarking/run.py \
  --config benchmarking/gpu-image-table-format-storage-matrix-single-gpu.yaml \
  --strict-config-check
```

The checked-in matrix requests exactly one Ray GPU and runs three sequential
entries on that GPU: node-local NVMe, the site's Weka-backed `/lustre` mount,
and S3. Each entry keeps both its reusable matched inputs and measured outputs
on the backend under test. It loads the `pdx-multimodal` identity from
`~/.config/datamover/storage_locations`; secrets are never written to benchmark
parameters or logs.

Input and output roots accept:

- NVMe: `/local/$USER/...` or `/tmp/...`
- Weka: `/lustre/fsw/...` or `/scratch/fsw/...`
- S3: `s3://bucket/prefix`

The source fragment, seed, batch size, row-group size, heuristic, and GPU stay
fixed; only `--working-root`, `--output-root`, and the descriptive storage label
change. Inputs are reused after preparation, and trials run in one actor in
fraction order, so results represent a warm-runtime, mixed-cache workflow.

## Results

`benchmarking/run.py` captures environment data and GPU utilization. The script
adds `trials.json`, `fraction_summaries.json`, standard `params.json`,
`metrics.json`, and `tasks.pkl`.
Each stage result includes source read, selection, host/device boundary, decode,
heuristic, filter, writer-data, Lance-commit, and end-to-end timings. Summaries
report mean, min, max, and standard deviation rather than the best run. Run
`summarize_gpu_image_table_format_matrix.py SESSION_ROOT` to validate actor
reuse and create a combined `storage_matrix.json` and `storage_matrix.md`.

### Single-H100 storage matrix (2026-08-01)

The final matrix used one H100 80 GB, source version 4, and representative
fragment 208 with 4,941 rows. Every storage backend used the same deterministic
row cohorts, 256-row decode batches, 64×64 heuristic resize, blur threshold
0.10, and three alternating trials per fraction. Both formats produced the
same coordinates, scores, retained rows, and JPEG payload digests in all 30
measured tasks per backend. The full machine-readable result, including every
stage mean/min/max/standard deviation, is checked in at
`benchmarking/results/gpu_image_table_formats/storage_matrix_20260801.json`.

Measured task time excludes actor setup and both warmup tasks. Values below are
mean [minimum, maximum] seconds; `L/P` is Lance divided by Parquet.

| Storage | Fraction | Rows | cuDF Parquet end-to-end | Lance end-to-end | L/P |
| --- | ---: | ---: | ---: | ---: | ---: |
| NVMe | 10% | 495 | 1.369 [1.335, 1.433] | 1.417 [1.389, 1.457] | 1.036 |
| NVMe | 20% | 989 | 2.448 [2.366, 2.603] | 3.094 [3.024, 3.152] | 1.264 |
| NVMe | 40% | 1,977 | 3.791 [3.735, 3.850] | 5.631 [5.542, 5.690] | 1.485 |
| NVMe | 80% | 3,953 | 6.794 [6.595, 6.997] | 11.147 [11.055, 11.274] | 1.641 |
| NVMe | 100% | 4,941 | 8.074 [8.028, 8.150] | 14.503 [14.474, 14.540] | 1.796 |
| Lustre/Weka | 10% | 495 | 1.657 [1.603, 1.759] | 1.536 [1.518, 1.571] | 0.927 |
| Lustre/Weka | 20% | 989 | 2.817 [2.715, 2.953] | 3.263 [3.219, 3.347] | 1.158 |
| Lustre/Weka | 40% | 1,977 | 4.289 [4.222, 4.404] | 5.960 [5.812, 6.134] | 1.390 |
| Lustre/Weka | 80% | 3,953 | 7.420 [7.284, 7.632] | 11.741 [11.658, 11.893] | 1.582 |
| Lustre/Weka | 100% | 4,941 | 8.769 [8.723, 8.851] | 14.809 [14.536, 15.249] | 1.689 |
| S3 | 10% | 495 | 93.581 [48.013, 165.341] | 46.438 [41.595, 48.865] | 0.496 |
| S3 | 20% | 989 | 120.317 [92.045, 141.278] | 59.375 [56.627, 62.472] | 0.493 |
| S3 | 40% | 1,977 | 74.207 [49.754, 92.636] | 80.725 [79.082, 83.672] | 1.088 |
| S3 | 80% | 3,953 | 65.975 [53.854, 73.277] | 85.246 [77.974, 92.639] | 1.292 |
| S3 | 100% | 4,941 | 67.894 [60.939, 74.702] | 90.304 [85.054, 94.263] | 1.330 |

CUDA/runtime initialization was measured only for the audit below. Actor task
indexes 0 and 1 are warmups; every reported task is index 2 through 31 from the
same process and CUDA context.

| Storage | Setup excluded | Warmups excluded | Measured tasks | Pipeline wall |
| --- | ---: | ---: | ---: | ---: |
| NVMe | 10.323 s | 5.341 s | 30 | 204.509 s |
| Lustre/Weka | 10.367 s | 5.712 s | 30 | 217.323 s |
| S3 | 10.617 s | 89.414 s | 30 | 2,466.457 s |

The following representative stage means show where steady-state time goes.
`Boundary` is the encoded-image GPU-to-decoder-host copy plus Arrow-to-GPU for
Lance. `Write` includes Lance Arrow preparation and data write. Small metadata,
row-selection, filter, and validation stages explain the remainder.

| Storage | Rows | Format | Read | Boundary | nvJPEG | Blur | Write | Commit | End-to-end |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| NVMe | 495 | Parquet | 0.205 | 0.095 | 0.565 | 0.012 | 0.434 | — | 1.369 |
| NVMe | 495 | Lance | 0.169 | 0.107 | 0.561 | 0.012 | 0.520 | 0.005 | 1.417 |
| NVMe | 4,941 | Parquet | 0.195 | 0.961 | 5.330 | 0.118 | 1.194 | — | 8.074 |
| NVMe | 4,941 | Lance | 2.542 | 1.094 | 5.287 | 0.117 | 5.178 | 0.005 | 14.503 |
| Lustre/Weka | 495 | Parquet | 0.391 | 0.096 | 0.577 | 0.011 | 0.507 | — | 1.657 |
| Lustre/Weka | 495 | Lance | 0.191 | 0.102 | 0.566 | 0.011 | 0.566 | 0.036 | 1.536 |
| Lustre/Weka | 4,941 | Parquet | 0.400 | 0.931 | 5.402 | 0.117 | 1.704 | — | 8.769 |
| Lustre/Weka | 4,941 | Lance | 2.744 | 1.045 | 5.384 | 0.116 | 5.277 | 0.030 | 14.809 |
| S3 | 495 | Parquet | 82.077 | 0.110 | 0.555 | 0.012 | 10.165 | — | 93.581 |
| S3 | 495 | Lance | 13.904 | 0.107 | 0.559 | 0.011 | 22.235 | 8.786 | 46.438 |
| S3 | 4,941 | Parquet | 28.875 | 0.970 | 5.311 | 0.117 | 31.613 | — | 67.894 |
| S3 | 4,941 | Lance | 23.602 | 1.066 | 5.231 | 0.116 | 46.513 | 12.785 | 90.304 |

### Findings and pitfalls

- Runtime initialization was the earlier false bottleneck. The persistent actor
  removes it from trial time and changes the 10% NVMe conclusion: warmed
  Parquet is 3.6% faster, not slower. Runtime setup remains visible as a
  separate audit metric.
- The actual blur computation is already cheap: approximately 0.012 seconds
  for 495 images and 0.117 seconds for 4,941. Batched nvJPEG decode is the main
  compute cost at about 5.3 seconds for the complete fragment. A 512-row spot
  check reduced decode time by about 2.3% but did not improve end-to-end time
  reliably and increased peak device memory, so the matrix keeps 256 rows.
- A direct cuDF-device-buffer prototype was also exercised against the real
  cohort, but this installed nvJPEG backend rejected its first batch with
  `JPEG_NOT_SUPPORTED`; nvImageCodec was not installed in the selected venv.
  The validated path therefore remains TorchVision's batched nvJPEG API and
  reports its required compressed-byte copy instead of hiding it.
- Every random cohort touched all five Parquet row groups. Parquet therefore
  reads 4,941 physical rows even at 10% (9.982× amplification), while Lance
  reads exactly the selected offsets. That makes Lance about 2× faster on S3
  at 10% and 20%. Once enough rows are selected, Parquet's much cheaper bulk
  write wins: Parquet is 8.8% faster at 40% and 24.8% faster at 100% on S3.
- S3 is a mixed remote-cache result, not a cold-cache bandwidth test. Fractions
  run in ascending order against the same objects; server/backend caching and
  network variance make Parquet read time non-monotonic. The very wide
  min-to-max ranges, especially 10% and 20%, must be considered with the means.
- One S3 development run exposed delayed/missing visibility after a successful
  Lance create commit. The Curator Lance checkpoint committer now polls with
  bounded exponential backoff and republishes a missing create/overwrite
  transaction once. Append is never republished because that could duplicate
  rows. The repaired path recovered the exact failed checkpoint; the final S3
  matrix then completed without invoking the retry.
- `large_string` is only an in-memory cuDF compatibility view. Lance input and
  output remain Arrow `large_binary`, and output byte digests are validated.
- Parquet physically stores the image as BYTE_ARRAY. The prepared input and
  every cuDF output were footer-validated with image compression disabled,
  dictionary encoding disabled, and `PLAIN` value encoding. Prepared input
  metadata reports `UNCOMPRESSED` and `RLE, PLAIN`; `RLE` encodes definition
  levels, not values. cuDF outputs report `UNCOMPRESSED` and `PLAIN` only.
- Results favor random sparse rows. Contiguous ranges or smaller Parquet row
  groups would reduce Parquet read amplification and should be measured before
  generalizing beyond this workflow.
