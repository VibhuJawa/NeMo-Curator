# Code Annotation Library

This directory contains NVIDIA's custom Rust code for fast code annotation in NeMo Curator.

## Overview

The code_annotation library provides high-performance code analysis capabilities:

- **Basic annotations** - Byte counts, line statistics, pattern detection (base64, hex, unicode)
- **Language detection** - Programming language identification via hyperpolyglot
- **Software metrics** - Code quality metrics via rust-code-analysis wrapper
- **Tokenization** - BPE tokenization with customizable vocabularies
- **Decontamination** - N-gram matching for benchmark contamination detection

## Architecture

```
code_annotation/
├── src/
│   ├── rust/
│   │   ├── lib.rs              # PyO3 module entry point
│   │   ├── annotations.rs      # Core annotation functions
│   │   └── annotations/
│   │       └── vendored_loc.rs # Custom LOC counting
│   ├── macros/
│   │   └── src/lib.rs          # Proc-macros for function registration
│   └── python/
│       └── code_annotation/    # Python package
├── Cargo.toml.template         # Cargo template (references external deps)
├── pyproject.toml              # Python build configuration
└── build.sh                    # Build script
```

## Dependencies

This library depends on external components fetched at build time:

| Dependency | Purpose | License |
|------------|---------|---------|
| hyperpolyglot | Language detection | MIT/Apache-2.0 |
| rust-code-analysis | Code metrics | MPL-2.0 |
| bpe-openai | BPE tokenization | MIT |
| polars | DataFrame processing | Apache-2.0 |

## Building

### Prerequisites

- Rust toolchain (1.70+)
- Python 3.10+
- maturin (for building Python wheels)

### Build Steps

The build script orchestrates fetching external dependencies and building:

```bash
# From the NeMo-Curator root directory
./external/code_annotation/build.sh
```

Or use the master build script:
```bash
./scripts/build_code_annotation.sh
```

## Python API

```python
import code_annotation

# Annotate a DataFrame with compressed source code
df = code_annotation.annotate({
    "basic": {"xml_header_search_length": 100},
    "detect_language": {},
    "software_metrics": {},
}, df)
```

## Registered Functions

| Function | Description | Arguments |
|----------|-------------|-----------|
| `basic` | Basic stats, pattern detection | `xml_header_search_length`, `max_decompressed_byte_size` |
| `detect_language` | Language detection | (none) |
| `software_metrics` | Code quality metrics | (none) |
| `tokenize` | BPE tokenization | `tokenizer_name`, `vocab`, `pretokenizer_patterns` |
| `opencoder_rs` | Comment fraction analysis | (none) |
| `decontaminate` | N-gram contamination check | `ngrams`, `ngram_order` |

## License

This code is licensed under Apache-2.0.
