# Scheduler incident 402755

## Summary

On 2026-07-08 the `fragment-amort` Slurm array triggered the cluster's
defunct-job detector. Six one-GPU array elements completed in
43 seconds to 2 minutes 32 seconds. The scheduler correctly identified this as
an inefficient submission shape: every microbenchmark point paid independent
queue, allocation, container, and teardown overhead.

No element of `402755` remains pending, held, or running. There is therefore no
job to release with `scontrol release` or cancel.

## Application audit

| Point | Slurm state | Application result |
| --- | --- | --- |
| 1 row/fragment | `COMPLETED` | Correct |
| 10 rows/fragment | `COMPLETED` | Correct |
| 25 rows/fragment | `COMPLETED` | Correct |
| 100 rows/fragment | `COMPLETED` | Correct |
| 500 rows/fragment | `COMPLETED` | Setup failed |
| 1,000 rows/fragment | `COMPLETED` | Correct |

The 500-row point encountered a live-source constructor mismatch during setup.
The old benchmark harness retained top-level status `completed` and exit code
zero even though the arm status was `setup_failed`. Its measurement is invalid
and is excluded from the performance report.

## Corrective actions

1. Do not submit sub-ten-minute parameter points as Slurm array elements.
   Reserve one exclusive allocation, run the points sequentially in that
   allocation, persist each point independently, then release the allocation.
2. Treat every requested benchmark arm that does not reach `completed` as a
   failed run. Setup, unavailable-arm, warmup, repeat, validation, and teardown
   failures must produce a nonzero process exit.
3. Run a CPU-only preflight for argument parsing, manifest construction, and
   output-path creation before allocating a GPU.
4. Point `REPO_ROOT` and `PYTHONPATH` at an immutable clean commit or dedicated
   benchmark worktree, record that commit in every artifact, and never execute
   a batch allocation against a shared worktree being edited.
5. Use arrays for independent production-size shards whose useful runtime is
   comfortably above the cluster's short-job threshold, not for latency-scale
   sweeps.
6. Keep partial artifacts for diagnosis, but never aggregate them into a
   speedup or scale model.

Every checked-in launcher now rejects `SLURM_ARRAY_JOB_ID` before it creates an
output or starts `srun`. This guards against repeating the exact inline
`sbatch --array ... run_gpu_lance_scaling_job.sh` launch shape that caused this
incident. A shared preflight also rejects restarted, requeueable, non-running,
or non-exclusive allocations; clears inherited Ray cluster discovery; validates
all rank manifests and immutable dataset/sidecar pins; and refuses existing
output paths. New allocations must use `--no-requeue`, exclusive nodes, and a
fresh run ID. These runtime guards do not replace submission-time review.

The follow-on queue work in that interactive allocation is finished. No
replacement fragment-amortization array was submitted, and no saturation or
scaling job from this incident remains in flight. New live saturation runs also
fail before output creation unless the caller supplies a minimum remaining-time
floor and a numeric allocation end time; this prevents a repeat set from
starting when the allocation cannot plausibly finish all requested repeats.
