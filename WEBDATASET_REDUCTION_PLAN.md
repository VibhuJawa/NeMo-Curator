# WebDataset Reader Reduction Plan

## Scope
Target file: `nemo_curator/stages/multimodal/io/readers/webdataset.py`

Goal: reduce density and repeated logic while preserving behavior and test outcomes.

## Principles
1. No behavior changes unless explicitly called out and reviewed.
2. Keep exception semantics stable (`raise`/`skip`/`log`).
3. Run tests after each phase, not only at the end.
4. Prefer removing duplicated branches over adding many small helpers.

## Current Pain Points
1. Interleaved row creation has repeated text/image branch logic.
2. Text-member parsing combines multiple concerns in one path.
3. Fallback/error handling around interleaved parsing is harder to scan than needed.
4. Member-type dispatch has repeated suffix handling patterns.

## Reduction Phases

### Phase 1: Control-flow cleanup (low risk)
1. Keep current behavior, but collapse redundant exception branches where invariant checks already exist.
2. Keep one clear fallback path for `sample_format != "interleaved"`.
3. Ensure all error messages that tests rely on remain unchanged.

Acceptance:
1. Existing multimodal tests pass unchanged.
2. No schema/output changes.

### Phase 2: Duplicate row-construction removal (low-medium risk)
1. Centralize interleaved segment row construction into one reusable function.
2. Reuse that function from all interleaved segment paths.
3. Keep modality checks and metadata JSON population exactly as today.

Acceptance:
1. Interleaved reader/writer roundtrip tests pass.
2. Element metadata JSON behavior remains identical.

### Phase 3: Text-member path decomposition (medium risk)
1. Split JSON text-member path from plain text-member path for readability.
2. Keep shared row finalization in one place to avoid re-duplicating code.
3. Retain existing sample-id/position assignment rules.

Acceptance:
1. Non-interleaved JSON fallback behavior remains unchanged.
2. Metadata sidecar population stays first-wins and identical to current tests.

### Phase 4: Optional line-count pass (optional)
1. Re-evaluate helper count vs readability.
2. Inline helpers that only wrap one call and do not improve clarity.
3. Keep final structure flat and easy to follow.

Acceptance:
1. Net reduction in repeated branches and lines where practical.
2. No decrease in maintainability/readability.

## Test Gate Per Phase
Run:
1. `pytest -q tests/stages/multimodal/test_writer_output_formats.py`
2. `pytest -q tests/stages/multimodal/test_parquet_reader.py`

## Deliverables
1. One commit per phase (or grouped Phase 1+2 if very small).
2. Updated diff summary with:
1. duplicated branches removed
2. net insertions/deletions
3. test results
