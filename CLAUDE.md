# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

NVIDIA NeMo Curator is a GPU-accelerated data curation framework for training AI models across text, image, video, and audio modalities. It uses Ray for distributed processing and provides a unified API for defining pipelines that can run on different backends (Xenna, Ray Actors, Ray Data) without changing pipeline logic.

**Key Technologies**: Python 3.10-3.12, Ray, PyTorch, RAPIDS (cuDF, cuML), uv package manager

## Development Setup

### Environment Activation (Always Run First)

```bash
# Always run these commands first when starting a new session
source /home/nfs/vjawa/.bashrc
cd /raid/vjawa/NeMo-Curator && source .venv/bin/activate
```

### Initial Setup (One-Time)

```bash
# Install uv package manager (if not already installed)
pip3 install uv

# Install base dependencies
uv sync

# Install with modality-specific extras (common options)
uv sync --extra text_cuda12      # Text curation with GPU support
uv sync --extra image_cuda12     # Image curation with GPU support
uv sync --extra video_cuda12     # Video curation with GPU support
uv sync --extra audio_cuda12     # Audio curation with GPU support
uv sync --extra all              # All features with CUDA 12.x support

# Install pre-commit hooks (REQUIRED)
pre-commit install --install-hooks
```

## Common Commands

### Testing
```bash
# Interleaved module tests (verbose output)
python -m pytest tests/stages/interleaved/ -v

# Full test suite (quiet mode)
python -m pytest tests/ -q

# Run all tests (CPU-only by default)
uv run pytest

# On shared machines where another user's Ray cluster is already running on
# ports 6379-6399, the autouse shared_ray_cluster fixture tries to start a NEW
# cluster and hangs for ~5 minutes before failing. Work around this by pointing
# Ray at the existing cluster BEFORE invoking pytest:
#   export RAY_ADDRESS=<host:port>   # e.g. export RAY_ADDRESS=10.184.206.10:6399
# The conftest explicitly deletes RAY_ADDRESS, so this env var does NOT skip
# Ray startup — instead, run tests that skip Ray entirely with:
#   python -m pytest tests/stages/interleaved/ -p no:randomly --ignore=... -v

# Run tests excluding GPU tests
uv run pytest -m "not gpu"

# Run GPU tests only (requires CUDA environment)
uv run pytest -m gpu

# Run specific test module
uv run pytest tests/stages/text/
uv run pytest tests/utils/test_nvcodec_utils.py

# Run with coverage
uv run pytest --cov=nemo_curator
```

### Linting and Formatting
```bash
# Lint check (use this for quick verification)
python -m ruff check nemo_curator/ tests/

# Run linting on entire project
uv run ruff check .

# Auto-fix linting issues
uv run ruff check . --fix

# Format code
uv run ruff format .

# IMPORTANT: Always verify lint-clean after editing any file
python -m ruff check <file>
```

### Documentation
```bash
# Build HTML documentation
make docs-html

# Live-reload documentation server
make docs-live

# Setup documentation environment
make docs-env
```

### Dependency Management
```bash
# Update dependencies
uv lock --upgrade

# Sync dependencies with lock file
uv sync
```

## Test-Driven Development (Interleaved Module)

For any change to `nemo_curator/stages/interleaved/` or `nemo_curator/tasks/interleaved.py`, follow this workflow:

1. **Inspect real data first** — Examine actual data at the available dataset paths (e.g., MINT-1T tars) to understand structure, edge cases, and real-world behavior. Ask the user which dataset to use if unclear. Fall back to synthetic data only when no real data is available.

2. **Write tests** — Design test cases informed by the real data inspection. Use synthetic fixtures in `tests/stages/interleaved/` that mimic real data structure (e.g., multi-frame TIFFs, interleaved text/image samples, tar archives).

3. **Implement** — Write the minimum code to make tests pass.

## Benchmarking

### Single Shard Smoke Test (No Materialization)
```bash
python benchmarking/scripts/multimodal_mint1t_benchmark.py \
  --benchmark-results-path .tmp_multimodal_runs/run_name \
  --input-path /datasets/vjawa/MINT-1T-PDF-CC-2024-18-10gb/CC-MAIN-2024-18-shard-0/CC-MAIN-20240412101354-20240412131354-00000.tar \
  --output-path .tmp_multimodal_runs/run_name/output \
  --no-materialize-on-write --no-materialize-on-read --mode overwrite
```

### Full 10GB Benchmark (90 Shards)
```bash
python benchmarking/scripts/multimodal_mint1t_benchmark.py \
  --benchmark-results-path .tmp_multimodal_runs/run_name \
  --input-path /datasets/vjawa/MINT-1T-PDF-CC-2024-18-10gb/CC-MAIN-2024-18-shard-0/ \
  --output-path .tmp_multimodal_runs/run_name/output \
  --mode overwrite
```

## Available Datasets

Use these dataset paths for testing and benchmarking:

- **Single shard (79MB)**: `/datasets/vjawa/MINT-1T-PDF-CC-2024-18-10gb/CC-MAIN-2024-18-shard-0/CC-MAIN-20240412101354-20240412131354-00000.tar`
- **10GB MINT1T (90 tar shards)**: `/datasets/vjawa/MINT-1T-PDF-CC-2024-18-10gb/CC-MAIN-2024-18-shard-0/`
- **20GB interleaved sample (17 tars)**: `/raid/vjawa/mint_interleaved_100mb_sample/webdataset/`
- **173GB parquet subset (997 files)**: `/datasets/vjawa/nvmint_mint1t_parquet_1k_subset/`

## Architecture Overview

NeMo Curator uses a **task-centric architecture** where data flows through pipelines as Tasks:

### Core Components

1. **Tasks** (`nemo_curator/tasks/`): Fundamental units of data that flow through pipelines
   - `DocumentBatch`: Text documents (data: pd.DataFrame or pa.Table)
   - `ImageBatch`: Images (data: list[ImageObject])
   - `VideoTask`: Videos (data: Video)
   - `AudioBatch`: Audio files (data: dict or list[dict])
   - `FileGroupTask`: File paths (data: list[str])
   - `_EmptyTask`: Singleton for data-generating stages (readers, file discovery)

2. **Stages** (`nemo_curator/stages/`): Processing operations that transform tasks
   - Inherit from `ProcessingStage[InputType, OutputType]`
   - Implement `process(task)` → task | list[task] | None
   - Declare resource requirements via `resources` attribute
   - Define input/output requirements via `inputs()` and `outputs()` methods
   - Organized by modality: `text/`, `image/`, `video/`, `audio/`, `interleaved/`

3. **Pipelines** (`nemo_curator/pipeline/`): Sequences of stages
   - Compose stages with `pipeline.add_stage(stage)`
   - Execute with `pipeline.run(executor, initial_tasks)`
   - Automatically decomposes composite stages

4. **Executors/Backends** (`nemo_curator/backends/`): Run pipelines on different Ray backends
   - `XennaExecutor`: NVIDIA Cosmos orchestration (default)
   - `RayActorExecutor`: Ray actor pool execution
   - `RayDataExecutor`: Ray Data framework execution
   - Unified interface allows switching backends without changing pipeline logic

### Key Design Principles

- **Fault Tolerance**: All stages MUST be fault-tolerant and retry-safe (Xenna can preempt tasks)
- **Backend Agnostic**: Same pipeline runs on any executor
- **Type Safety**: Generic types ensure input/output compatibility
- **Map-Style Processing**: Stages operate on individual tasks in parallel
- **Modality Organized**: Code organized by data type (text, image, video, audio)

## Implementation Patterns

### Creating a Processing Stage

```python
from dataclasses import dataclass
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import DocumentBatch

@dataclass
class MyStage(ProcessingStage[DocumentBatch, DocumentBatch]):
    name: str = "MyStage"  # Override default name
    resources: Resources = Resources(cpus=1.0, gpu_memory_gb=4.0)
    batch_size: int = 1  # Number of tasks to process at once

    def inputs(self) -> tuple[list[str], list[str]]:
        """Define required task and data attributes."""
        return ["data"], ["text"]  # Requires task.data with "text" column

    def outputs(self) -> tuple[list[str], list[str]]:
        """Define output task and data attributes."""
        return ["data"], ["processed_text"]  # Adds "processed_text" column

    def process(self, task: DocumentBatch) -> DocumentBatch:
        """Process a single task."""
        # Transform task.data
        return DocumentBatch(
            task_id=f"{task.task_id}_{self.name}",
            dataset_name=task.dataset_name,
            data=transformed_data,
            _metadata=task._metadata,
            _stage_perf=task._stage_perf,
        )
```

### Filtering Pattern (Empty Batches, Never `None`)

**Stages must never return `None` to filter data.** Always return the task with filtered rows (an empty DataFrame/Table is fine). This is the established pattern in text, interleaved, and all other modules.

```python
def process(self, task: DocumentBatch) -> DocumentBatch:
    df = task.to_pandas()

    # Early exit for already-empty tasks
    if df.empty:
        return task

    # Apply row-level filtering
    keep_mask = df["score"] > self.threshold
    df = df[keep_mask]

    if len(df) == 0:
        logger.info("All rows filtered out for task {}", task.task_id)

    # Always return a batch — even if 0 rows
    return DocumentBatch(
        task_id=f"{task.task_id}_{self.name}",
        dataset_name=task.dataset_name,
        data=df,
        _metadata=task._metadata,
        _stage_perf=task._stage_perf,
    )
```

**Why not `None`?**
- Empty batches are visible and debuggable; `None` silently vanishes
- Consistent with text module, interleaved module, and all existing filter stages
- Fault-tolerant: Xenna can safely retry a stage that returned an empty batch
- The backend has a `None` safety-net filter, but stages should never rely on it

### Optional Stage Lifecycle Methods

- `setup_on_node(node_info, worker_metadata)`: Node-level initialization (e.g., download models)
- `setup(worker_metadata)`: Worker-level initialization (e.g., load models into memory)
- `teardown()`: Cleanup after processing
- `process_batch(tasks)`: Vectorized batch processing (more efficient than individual `process()`)

### Building a Pipeline

```python
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.text.io.reader.jsonl import JsonlReader
from nemo_curator.stages.text.io.writer.jsonl import JsonlWriter

# Create pipeline
pipeline = Pipeline(name="my_pipeline")
pipeline.add_stage(JsonlReader(input_file_path, files_per_partition=1))
pipeline.add_stage(MyProcessingStage())
pipeline.add_stage(JsonlWriter(output_file_path))

# Run with default executor (Xenna)
result = pipeline.run()

# Or specify executor explicitly
from nemo_curator.backends.ray_data import RayDataExecutor
result = pipeline.run(executor=RayDataExecutor())
```

### Resource Specification

```python
from nemo_curator.stages.resources import Resources

# CPU-only stage
resources = Resources(cpus=1.0)

# Single GPU stage
resources = Resources(cpus=1.0, gpu_memory_gb=8.0)

# Multi-GPU stage
resources = Resources(cpus=2.0, gpus=2.0)

# Entire GPU allocation
resources = Resources(cpus=1.0, entire_gpu=True)
```

## Testing Guidelines

### Test Organization
- Tests mirror source structure: `tests/` matches `nemo_curator/`
- Mark GPU tests with `@pytest.mark.gpu`
- 80% coverage requirement for code in `nemo_curator/`

### Writing Tests

```python
import pytest

# CPU test (runs by default)
def test_cpu_processing():
    stage = MyStage()
    task = DocumentBatch(...)
    result = stage.process(task)
    assert result is not None

# GPU test (requires CUDA)
@pytest.mark.gpu
def test_gpu_processing():
    import cudf
    # GPU-specific test logic
```

### Running Specific Tests

```bash
# Run single test file
uv run pytest tests/stages/text/test_my_stage.py

# Run single test function
uv run pytest tests/stages/text/test_my_stage.py::test_cpu_processing

# Run tests matching pattern
uv run pytest -k "test_reader"
```

## Code Standards

### Linting (Ruff)
- Line length: 119 characters
- Type annotations required (except `*args`, `**kwargs`, special methods)
- Use `loguru` for logging: `from loguru import logger`
- **Always use loguru `{}` format, never f-strings in log calls** — loguru evaluates arguments lazily so the string is only built if the level is active:
  ```python
  # correct
  logger.warning("schema_overrides ignored because schema= is set; got overrides={}", overrides)
  logger.info("Processed {} rows in task {}", len(df), task.task_id)
  # wrong — f-string is always evaluated even if WARNING is disabled
  logger.warning(f"schema_overrides ignored because schema= is set; got overrides={overrides}")
  ```
- Print statements allowed
- Boolean arguments in functions allowed
- **NEVER add `# noqa:` comments** in mainline code (`nemo_curator/`) to suppress lint warnings. Fix the underlying issue instead (refactor the code, reduce arguments, etc.)
- **In test files (`tests/`)**, `# noqa:` is permitted for rules that are legitimately impractical to fix structurally — for example `PLR0913` (too many arguments) on fixture factory functions that need many parameters by design. Always prefer structural fixes first; use `# noqa:` in tests only as a last resort.
- **Always run `python -m ruff check <file>`** after writing or editing any Python file to verify there are no lint errors before considering the task complete

### Git Commits and Pre-Commit Workflow

**⚠️ CRITICAL: Pre-commit hooks are REQUIRED before pushing any code.**

**Do NOT add `Co-Authored-By: Claude ...` lines to commit messages.** The user does not want Claude authorship attribution in commits.

The repository has pre-commit hooks that automatically check:
- Code formatting and linting (Ruff)
- Large file detection
- YAML validity
- Private key detection
- Trailing whitespace
- Signed-off-by in commit messages
- Dependency lock file sync (uv)

#### Workflow

1. **Make changes** to code

2. **Stage changes**
   ```bash
   git add <files>
   ```

3. **Commit with sign-off** (REQUIRED)
   ```bash
   git commit -s -m "Your commit message"
   ```
   - The `-s` flag adds `Signed-off-by: Author Name <author@example.com>`
   - Required by the DCO (Developer Certificate of Origin) check on PRs
   - Pre-commit hooks run automatically on commit

4. **If pre-commit hooks fail**:
   - Review the failures
   - If hooks auto-fix files, stage the fixes:
     ```bash
     git add <auto-fixed-files>
     git commit --amend -s --no-edit
     ```
   - If manual fixes needed, fix them, stage, and amend:
     ```bash
     # Fix issues manually
     git add <fixed-files>
     git commit --amend -s --no-edit
     ```

5. **Verify hooks pass** by running manually:
   ```bash
   pre-commit run --from-ref HEAD~1 --to-ref HEAD
   ```
   - Repeat steps 4-5 until all hooks pass cleanly

6. **Before pushing**, always verify pre-commit passes:
   ```bash
   pre-commit run --all-files  # Run on all files (thorough check)
   # OR
   pre-commit run --from-ref origin/main --to-ref HEAD  # Run on changed files
   ```

7. **Push to remote**
   ```bash
   git push
   ```

### GitHub API — Updating PR Descriptions

`gh pr edit` requires `read:org` scope which is not available on NVIDIA-NeMo org tokens. Use the REST API directly instead:

```bash
gh api repos/NVIDIA-NeMo/Curator/pulls/<PR_NUMBER> \
  --method PATCH \
  --field body="$(cat <<'EOF'
## Summary
...your description here...
EOF
)" --jq '.html_url'
```

#### Quick Reference

```bash
# Standard commit workflow
git add .
git commit -s -m "feat: add new feature"
pre-commit run --from-ref HEAD~1 --to-ref HEAD
# If failed, fix and amend
git add .
git commit --amend -s --no-edit
# Push when clean
git push
```

### Copyright Header
All non-empty Python files must include:

```python
# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
```

### Import Patterns
```python
# Standard error handling for optional dependencies
try:
    import required_library
except ImportError as e:
    logger.error("Required dependency not found: {}", e)
    raise ImportError("Please install the required dependencies")
```

## File Structure

```
nemo_curator/
├── stages/              # Processing stages organized by modality
│   ├── text/           # Text processing (classifiers, filters, readers, writers)
│   ├── image/          # Image processing and filtering
│   ├── audio/          # Audio processing and ASR
│   ├── video/          # Video processing and analysis
│   ├── interleaved/    # Multimodal processing
│   ├── math/           # Mathematical reasoning
│   ├── synthetic/      # Synthetic data generation
│   └── base.py         # ProcessingStage base class
├── tasks/              # Task definitions (DocumentBatch, ImageBatch, etc.)
├── backends/           # Execution adapters (Xenna, Ray Actors, Ray Data)
├── pipeline/           # Pipeline orchestration
├── utils/              # Shared utilities
├── datasets/           # Dataset loading and manipulation
├── modules/            # Core processing algorithms
├── classifiers/        # Pre-built classifier modules
├── filters/            # Pre-built filter modules
└── modifiers/          # Text modification utilities

tests/                  # Mirrors nemo_curator/ structure
tutorials/              # Example scripts and notebooks by modality
docs/                   # Sphinx documentation
```

## GPU Development Notes

### Memory Management
- Use context managers for GPU memory allocation
- Monitor GPU memory during development with `gpustat`
- Implement graceful degradation when GPU unavailable

### CUDA Compatibility
- CUDA 12.x required for GPU features
- All GPU code must handle missing CUDA gracefully
- Test both single-GPU and multi-GPU scenarios

### Performance
- Profile GPU kernels for bottlenecks
- Use batch processing (`process_batch()`) for efficiency
- Leverage Ray for distributed GPU workloads

## Common Gotchas

1. **Stage Fault Tolerance**: Stages must handle interruption (Xenna can preempt tasks). Always check if partial work exists before starting.

2. **Task ID Naming**: Use `f"{task.task_id}_{self.name}"` pattern for output task IDs to maintain traceability.

3. **Resource Specification**: Don't request both `gpus` and `gpu_memory_gb` - use `gpu_memory_gb` for single GPU, `gpus` for multi-GPU.

4. **Empty Task Usage**: Use `_EmptyTask()` as input for reader stages and file discovery stages that generate data rather than transform it.

5. **Test Coverage**: Changes to `nemo_curator/` require corresponding tests to maintain 80% coverage.

6. **Type Annotations**: Processing stages must declare input/output types: `ProcessingStage[InputType, OutputType]`.

## Interleaved IO Module Conventions

### Schema Parameter Naming

**IMPORTANT**: For interleaved readers and writers, always use `schema` (not `input_schema` or `output_schema`).

This follows the Spark convention where a single `schema` parameter is used for both read and write operations.

**Default Behavior (Zero Config):**

```python
# ✓ MOST COMMON - No schema needed (defaults to INTERLEAVED_SCHEMA)
reader = InterleavedParquetReader(input_path="data.parquet")
writer = InterleavedParquetWriter(output_path="output.parquet")

# Also works for WebDataset
reader = InterleavedWebdatasetReader(input_path="data.tar")
writer = InterleavedWebdatasetWriter(output_path="output.tar")
```

**Custom Fields (schema_overrides):**

```python
# ✓ RECOMMENDED - Add/override specific fields only
reader = InterleavedParquetReader(
    input_path="data.parquet",
    schema_overrides={
        "url": pa.string(),           # Add custom field
        "timestamp": pa.int64(),      # Add custom field
        "text_content": pa.string(),  # Override (use string instead of large_string)
    }
)

# Internally merges with INTERLEAVED_SCHEMA:
# - All standard fields (sample_id, modality, text_content, etc.)
# - Plus your custom fields (url, timestamp)
# - With your type overrides (text_content as string)
```

**Full Custom Schema (Power Users):**

```python
# ✓ ADVANCED - Complete custom schema (no merge with INTERLEAVED_SCHEMA)
custom_schema = pa.schema([
    pa.field("sample_id", pa.string()),
    pa.field("custom_field", pa.int64()),
    # ... your complete schema
])

reader = InterleavedParquetReader(
    input_path="data.parquet",
    schema=custom_schema,  # Uses this schema exactly
)
```

**Wrong Usage:**

```python
# ✗ WRONG - Don't use 'output_schema' or 'input_schema'
reader = InterleavedParquetReader(
    input_path="data.parquet",
    output_schema=INTERLEAVED_SCHEMA,  # Don't use this
)
```

**Rationale:**
- Simpler API (one parameter name instead of two)
- Follows industry standard (Spark)
- Zero config for 95% of use cases (defaults to INTERLEAVED_SCHEMA)
- Easy customization with `schema_overrides` dict
- Readers and writers both work with the same schema

**This is new functionality** - no deprecation warnings needed, just use `schema` directly.

### Supported Modalities

**WebDataset Writer** supports only these modalities:
- `"metadata"` - Sample-level metadata
- `"text"` - Text content
- `"image"` - Image binary data

**Unsupported modalities will raise `ValueError`** (fail fast instead of silent data loss):
```python
# This will raise ValueError
df = pd.DataFrame({
    "sample_id": ["s1"],
    "modality": ["video"],  # ← Not supported!
    "text_content": [None],
})
# ValueError: Unsupported modality 'video'. Supported: 'metadata', 'text', 'image'
```

### Null Handling in Interleaved Data

When working with interleaved data (text, images, metadata):

1. **Missing Columns**: If a requested field is missing from source data:
   - With `schema` parameter: Filled with typed nulls from schema
   - Without `schema` parameter: Filled with `pa.null()` (untyped)
   - **Best Practice**: Always provide `schema` for predictable typing

2. **Null Binary Content**:
   - `materialize_on_write=True`: Binary content is fetched from source
   - `materialize_on_write=False`: May have null `binary_content`
   - Use `on_materialize_error` parameter to control error handling

3. **Empty Batches (Filtering Pattern)**:
   - **Never return `None` from a stage to filter data** — return the task with filtered rows instead
   - An empty DataFrame/Table (0 rows) is the correct way to signal "nothing to process"
   - All existing filter stages (text, interleaved) follow this pattern
   - The backend has a `None` safety-net filter but stages must not rely on it
   - Example: `if df.empty: return task` (pass through); `df = df[mask]` then return even if 0 rows

## References

- Main Documentation: https://docs.nvidia.com/nemo/curator/latest/
- API Reference: https://docs.nvidia.com/nemo/curator/latest/apidocs/index.html
- GitHub Copilot Instructions: `.github/copilot-instructions.md` (detailed architecture)
- Cursor Rules: `.cursor/rules/*.mdc` (implementation patterns)
- Quickstart Example: `tutorials/quickstart.py`
