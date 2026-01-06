# Hyperpolyglot External Dependency

This directory contains scripts to fetch, patch, and build the hyperpolyglot library
for programming language detection in NeMo Curator.

## Overview

[Hyperpolyglot](https://github.com/monkslc/hyperpolyglot) is a fast programming language detector
written in Rust. It uses multiple detection strategies:

1. **Filename detection** - Exact filename matches (e.g., `Makefile` → Make)
2. **Extension detection** - File extension matching (e.g., `.py` → Python)
3. **Shebang detection** - Interpreter detection from shebang lines
4. **Heuristics** - Content-based pattern matching
5. **Classifier** - Naive Bayes classification for ambiguous cases

## Version Pinning

| Component | Version/Commit | Source |
|-----------|---------------|--------|
| hyperpolyglot | `v0.5.6` (or specific commit) | https://github.com/monkslc/hyperpolyglot |

## Patches Applied

The following patches are applied to the upstream hyperpolyglot:

1. **001-add-python-bindings.patch** - Adds PyO3 bindings for Python integration
2. **002-expose-detector-functions.patch** - Exposes additional detector functions for NeMo Curator

## Building

### Prerequisites

- Rust toolchain (1.70+)
- Python 3.10+
- maturin (for building Python wheels)

### Build Steps

```bash
# From the NeMo-Curator root directory
./external/hyperpolyglot/fetch.sh
./external/hyperpolyglot/build.sh
```

Or use the master build script:
```bash
./scripts/build_code_annotation.sh
```

## License

hyperpolyglot is dual-licensed under MIT and Apache-2.0.
See the original repository for full license details.
