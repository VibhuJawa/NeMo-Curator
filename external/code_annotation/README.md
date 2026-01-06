# Code Annotation Library

This directory contains NVIDIA's custom Rust code for fast code annotation in NeMo Curator.

## Overview

The code_annotation library provides high-performance code analysis capabilities:

- **Basic annotations** - Byte counts, line statistics, pattern detection (base64, hex, unicode)
- **Language detection** - Programming language identification via hyperpolyglot
- **Software metrics** - Code quality metrics via rust-code-analysis wrapper (optional feature)
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
│   ├── software_metrics/       # Optional software metrics wrapper
│   └── python/
│       └── code_annotation/    # Python package
├── Cargo.toml.template         # Cargo template (references external deps)
├── pyproject.toml              # Python build configuration
└── build.sh                    # Build script
```

## Dependencies

This library depends on external components:

| Dependency | Purpose | License |
|------------|---------|---------|
| hyperpolyglot | Language detection | MIT/Apache-2.0 |
| rust-code-analysis | Code metrics (optional) | MPL-2.0 |
| bpe-openai | BPE tokenization | MIT |
| tiktoken-rs | Tiktoken tokenization | MIT |

**Note**: No Python dependencies required - pure Rust extension module.

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

To build with software metrics support:
```bash
cd external/code_annotation
cargo build --release --features software_metrics
maturin build --release --features software_metrics
```

## Python API

The library provides simple function-based APIs that operate on lists of strings:

```python
from code_annotation import (
    compute_basic_stats,
    compute_language_detection,
    compute_tokenization,
    compute_opencoder_rs,
    compute_decontamination,
    compute_ngram_matches,
)

# Input: plain list of code strings
codes = ["def hello(): print('world')", "int main() { return 0; }"]
filenames = ["hello.py", "main.c"]

# Basic statistics
stats = compute_basic_stats(codes, xml_header_search_length=1024, max_byte_size=None)
# Returns: [{"num_bytes": 27, "num_lines": 1, "alpha_percent": 0.74, ...}, ...]

# Language detection
langs = compute_language_detection(codes, filenames)
# Returns: [{"language": "Python", "language_detector": "Extension"}, ...]

# Tokenization
tokens = compute_tokenization(codes, "github_o200k_base")
# Returns: [{"tokens": [123, 456, ...], "num_tokens": 7}, ...]

# Comment analysis (requires language detection first)
languages = [lang["language"] for lang in langs]
comments = compute_opencoder_rs(codes, languages)
# Returns: [{"comment_lines_frac": 0.0, "comment_chars_frac": 0.0}, ...]

# Decontamination check
ngrams = {"test_set": ["hello world", "foo bar"]}
contamination = compute_decontamination(codes, ngrams, ngram_order=2)
# Returns: {"test_set": [0, 0]}  # counts per code string

# Find matching n-grams
matches = compute_ngram_matches(codes, ngrams, ngram_order=2)
# Returns: {"test_set": [[], []]}  # matched ngrams per code string
```

## Available Functions

| Function | Description | Arguments |
|----------|-------------|-----------|
| `compute_basic_stats` | Basic stats, pattern detection | `codes`, `xml_header_search_length=1024`, `max_byte_size=None` |
| `compute_language_detection` | Language detection | `codes`, `filenames` |
| `compute_software_metrics` | Code quality metrics (optional feature) | `codes`, `languages` |
| `compute_tokenization` | BPE tokenization | `codes`, `tokenizer_name`, `vocab=None`, `pretokenizer_patterns=None` |
| `compute_opencoder_rs` | Comment fraction analysis | `codes`, `languages` |
| `compute_decontamination` | N-gram contamination counts | `codes`, `ngrams`, `ngram_order` |
| `compute_ngram_matches` | Find matching n-grams | `codes`, `ngrams`, `ngram_order` |

### Basic Stats Output Fields

| Field | Type | Description |
|-------|------|-------------|
| `num_bytes` | int | Total byte count |
| `valid_utf8` | bool | Whether the string is valid UTF-8 |
| `max_line_length` | int | Maximum line length |
| `num_lines` | int | Total line count |
| `average_line_length` | float | Average line length |
| `contains_xml_header` | bool | Whether XML header is present |
| `alpha_percent` | float | Fraction of alphabetic characters |
| `alnum_percent` | float | Fraction of alphanumeric characters |
| `base64_percent` | float | Fraction matching base64 patterns |
| `hex_percent` | float | Fraction matching hex patterns |
| `unicode_percent` | float | Fraction matching unicode escape patterns |
| `base64_match_lengths` | list[int] | Lengths of base64 pattern matches |
| `hex_match_lengths` | list[int] | Lengths of hex pattern matches |
| `unicode_match_lengths` | list[int] | Lengths of unicode pattern matches |

### Tokenizer Names

- `"tiktoken_o200k_base"` - OpenAI's tiktoken o200k tokenizer
- `"github_o200k_base"` - GitHub's o200k tokenizer (faster)
- Custom tokenizer - provide `vocab` parameter with tiktoken-format vocabulary

## Software Metrics (Optional)

Software metrics require the `software_metrics` feature flag during build. This feature depends on rust-code-analysis which has specific tree-sitter version requirements.

```python
from code_annotation import compute_software_metrics, HAS_SOFTWARE_METRICS

if HAS_SOFTWARE_METRICS:
    metrics = compute_software_metrics(codes, languages)
    # Returns cyclomatic complexity, cognitive complexity, maintainability index, etc.
```

## License

This code is licensed under Apache-2.0.
