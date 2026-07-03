# Annotate a Lance dataset with an existing Curator stage

`LanceReader` and `LanceAnnotationWriter` can wrap any existing batch-to-batch
Curator stage. The classifier does not need to know that its input came from
Lance: it receives a `DocumentBatch`, appends its annotation column, and returns
the batch.

The stage must preserve `__lance_rowaddr` and `__lance_fragid`, must not filter
or duplicate rows, must raise if annotation fails, and should load its model
once in `setup()`. Its `process_batch()` must also return exactly one task per
input task and leave task IDs to Curator. The example below keeps the classifier
private and uses only public Curator plumbing.

```python
import pyarrow as pa

from nemo_curator.backends.ray_data import RayDataExecutor
from nemo_curator.core.client import RayClient
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.resources import Resources
from nemo_curator.stages.text.io.reader import LanceReader
from nemo_curator.stages.text.io.writer import (
    LanceAnnotationWriter,
    commit_lance_annotation_checkpoint,
)
from your_private_package import ImageQualityStage

DATASET_URI = "<existing-lance-dataset>"
COMMIT_PATH = "<fresh-annotation-checkpoint-directory>"
CHECKPOINT_PATH = "<shared-POSIX-resumability-directory>"
IMAGE_FIELD = "image"
SCORE_FIELD = "image_quality_score"

classifier = ImageQualityStage(
    image_field=IMAGE_FIELD,
    score_field=SCORE_FIELD,
    model_inference_batch_size=128,
).with_(
    resources=Resources(cpus=4, gpus=1),
    num_workers=8,
)

writer = LanceAnnotationWriter(
    path=DATASET_URI,
    commit_path=COMMIT_PATH,
    schema=pa.schema([pa.field(SCORE_FIELD, pa.float32(), nullable=True)]),
    fields=[SCORE_FIELD],
    create_columns=True,
)

# Create the annotation column if missing and pin the version being read.
read_version = writer.prepare()

pipeline = Pipeline(
    name="lance_image_annotation",
    stages=[
        LanceReader(
            path=DATASET_URI,
            fragments_per_partition=1,
            fields=[IMAGE_FIELD],
            read_kwargs={"version": read_version},
            include_lance_metadata=True,
        ),
        classifier,
        writer,
    ],
)

# Exceptions stop the workflow here, before the commit.
with RayClient():
    pipeline.run(
        executor=RayDataExecutor(),
        checkpoint_path=CHECKPOINT_PATH,
    )
committed_version = commit_lance_annotation_checkpoint(DATASET_URI, COMMIT_PATH)
print(f"Committed Lance version {committed_version}")
```

Use fresh `COMMIT_PATH` and `CHECKPOINT_PATH` directories for each new run.
Reuse both only when retrying the identical model and pipeline configuration. `LanceReader`
already decomposes to a partitioning source stage whose task IDs include the
dataset URI, pinned version, and fragment IDs, so completed fragment partitions
are skipped automatically. Do not modify the Lance dataset between
`writer.prepare()` and the final commit. For a Slurm array, workers should run
only the pipeline with a shared checkpoint path; one coordinator calls
`commit_lance_annotation_checkpoint` after every shard succeeds. See the
existing [Slurm tutorial](../../slurm/README.md) for cluster execution and
retries.
