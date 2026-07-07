# Annotate interleaved text in a Lance dataset

`LancePartitioningStage`, a thin private reader adapter, and
`LanceAnnotationWriter` can wrap a private, batch-to-batch annotation stage
without coupling the classifier to Lance. This example scores both the
concatenated text of each document and each original text segment, then writes
the annotations back to the source Lance dataset.

The thin reader adapter receives one Lance fragment and returns one
`InterleavedBatch`. The private classifier then returns one annotation-only
`DocumentBatch` containing every metadata and text row from that fragment and
preserves `__lance_rowaddr` and `__lance_fragid`. It may create internal model
microbatches, but it must not duplicate, reorder, or emit additional tasks.
Image rows may be omitted from the sparse update; newly created nullable
annotation columns remain null on those rows.

This layout assumes that a document's metadata row and all rows with the same
`sample_id` belong to one fragment. The stage may sort a working view by
`position` while assembling text, but must restore the source row order before
returning the batch.

## Annotation columns

Use one nullable set of classifier columns for both scoring granularities. The
existing `modality` column identifies the meaning of a score:

| Column | Arrow type | `metadata` row | `text` row | `image` row |
| --- | --- | --- | --- | --- |
| `text_quality_scored` | `bool` | Document was scored | Segment was scored | null |
| `text_quality_truncated_512` | `bool` | Concatenated input was truncated | Segment input was truncated | null |
| `text_quality_nemotron_score` | `float32` | Document score | Segment score | null |
| `text_quality_mistral_score` | `float32` | Document score | Segment score | null |
| `text_quality_fasttext_score` | `float32` | Document score | Segment score | null |
| `text_quality_nemotron_bin` | `uint8` | Document bin | Segment bin | null |
| `text_quality_mistral_bin` | `uint8` | Document bin | Segment bin | null |
| `text_quality_fasttext_bin` | `uint8` | Document bin | Segment bin | null |
| `text_quality_max_bin` | `uint8` | Document maximum bin | Segment maximum bin | null |
| `text_quality_mean_bin` | `float32` | Concatenated document score | Independent segment score | null |
| `text_quality_segment_observed_token_weighted_mean_bin` | `float32` | Combined segment score | null | null |
| `text_quality_segment_observed_token_fraction` | `float32` | Segment coverage | null | null |

For blank text, or a metadata row with no non-empty text segments, set
`text_quality_scored` to `false` and leave its remaining annotations null. An
inference error should raise and retry the fragment rather than be recorded as
an unscored row.

For a classifier that observes at most 510 content tokens, calculate the
metadata-row segment aggregate as

```text
weight_i = min(segment_token_count_i, 510)
combined_score = sum(weight_i * segment_score_i) / sum(weight_i)
observed_fraction = sum(weight_i) / sum(segment_token_count_i)
```

This weights a segment only by tokens the classifier actually observed. Keep
the concatenated document score as the primary, backward-compatible annotation;
the aggregate is an additional view.

## Pipeline

The classifier implementation below remains private. It is an ordinary
`ProcessingStage[InterleavedBatch, DocumentBatch]`: load the model once in
`setup()`, assemble document and segment inputs inside `process()`, use
`large_string` for working text, and return only the two Lance coordinates and
annotation columns. The thin reader subclass only changes the task container
from `DocumentBatch` to `InterleavedBatch`; it does not classify data.

```python
import pyarrow as pa

from nemo_curator.backends.ray_data import RayDataExecutor
from nemo_curator.core.client import RayClient
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.resources import Resources
from nemo_curator.stages.text.io.reader.lance import LancePartitioningStage
from nemo_curator.stages.text.io.writer import (
    LanceAnnotationWriter,
    commit_lance_annotation_checkpoint,
)
from your_private_package import (
    InterleavedLanceReaderStage,
    InterleavedTextQualityAnnotationStage,
)

DATASET_URI = "<existing-lance-dataset>"
COMMIT_PATH = "<fresh-lance-commit-checkpoint-directory>"
CHECKPOINT_PATH = "<fresh-curator-checkpoint-directory>"

SOURCE_FIELDS = ["sample_id", "position", "modality", "text_content"]
ANNOTATION_SCHEMA = pa.schema(
    [
        pa.field("text_quality_scored", pa.bool_(), nullable=True),
        pa.field("text_quality_truncated_512", pa.bool_(), nullable=True),
        pa.field("text_quality_nemotron_score", pa.float32(), nullable=True),
        pa.field("text_quality_mistral_score", pa.float32(), nullable=True),
        pa.field("text_quality_fasttext_score", pa.float32(), nullable=True),
        pa.field("text_quality_nemotron_bin", pa.uint8(), nullable=True),
        pa.field("text_quality_mistral_bin", pa.uint8(), nullable=True),
        pa.field("text_quality_fasttext_bin", pa.uint8(), nullable=True),
        pa.field("text_quality_max_bin", pa.uint8(), nullable=True),
        pa.field("text_quality_mean_bin", pa.float32(), nullable=True),
        pa.field(
            "text_quality_segment_observed_token_weighted_mean_bin",
            pa.float32(),
            nullable=True,
        ),
        pa.field(
            "text_quality_segment_observed_token_fraction",
            pa.float32(),
            nullable=True,
        ),
    ]
)

writer = LanceAnnotationWriter(
    path=DATASET_URI,
    commit_path=COMMIT_PATH,
    schema=ANNOTATION_SCHEMA,
    fields=ANNOTATION_SCHEMA.names,
    create_columns=True,
)

# Add missing nullable columns and pin the exact Lance version to annotate.
read_version = writer.prepare()

annotator = InterleavedTextQualityAnnotationStage(
    sample_id_field="sample_id",
    position_field="position",
    modality_field="modality",
    text_field="text_content",
    document_separator="\n\n",
    max_content_tokens=510,
    model_inference_batch_size=512,
).with_(
    resources=Resources(cpus=4, gpus=1),
    num_workers=8,
)

pipeline = Pipeline(
    name="lance_interleaved_text_quality_annotation",
    stages=[
        LancePartitioningStage(
            path=DATASET_URI,
            fragments_per_partition=1,
            read_kwargs={"version": read_version},
        ),
        InterleavedLanceReaderStage(
            path=DATASET_URI,
            fields=SOURCE_FIELDS,
            read_kwargs={"version": read_version},
            include_lance_metadata=True,
        ),
        annotator,
        writer,
    ],
)

with RayClient():
    pipeline.run(
        executor=RayDataExecutor(),
        checkpoint_path=CHECKPOINT_PATH,
    )

# Run this once, after every fragment task succeeds.
committed_version = commit_lance_annotation_checkpoint(DATASET_URI, COMMIT_PATH)
print(f"Committed Lance version {committed_version}")
```

`fragments_per_partition=1` makes the Lance partitioning source emit exactly
one task per source fragment. Curator checkpointing can therefore skip complete
fragments on retry, while the annotation stage can still classify model-sized
microbatches internally. The writer produces one update record per fragment.

Keep `COMMIT_PATH` separate from `CHECKPOINT_PATH`. Use fresh directories for a
new model or pipeline configuration, and reuse both only when retrying the
identical run. Do not modify the dataset between `writer.prepare()` and the
final commit. For a Slurm array, every worker runs only the pipeline against the
shared checkpoint paths; one coordinator commits after every array task
succeeds. See the [Slurm tutorial](../../slurm/README.md) for execution and
retry patterns.

## Keep run metadata outside the table

Do not add separate document/segment score families, copied text, row
addresses, token spans, input hashes, model identifiers, run identifiers,
timings, or checkpoint state as annotation columns. Store model provenance,
configuration, and runtime metrics in a run manifest. Lance row addresses are
temporary writer coordinates supplied by the reader, not durable annotations.
