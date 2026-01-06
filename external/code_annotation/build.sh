#!/bin/bash
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

# =============================================================================
# Code Annotation Build Script
# =============================================================================
# This script builds the code_annotation library with its external dependencies.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXTERNAL_DIR="$(dirname "${SCRIPT_DIR}")"
HYPERPOLYGLOT_SRC="${EXTERNAL_DIR}/hyperpolyglot/src"
SOFTWARE_METRICS_DIR="${SCRIPT_DIR}/src/software_metrics"

echo "=== Building code_annotation ==="

# Check dependencies
if [ ! -d "${HYPERPOLYGLOT_SRC}" ]; then
    echo "Error: hyperpolyglot not found. Run hyperpolyglot/fetch.sh first."
    exit 1
fi

# Create software_metrics wrapper directory if it doesn't exist
mkdir -p "${SOFTWARE_METRICS_DIR}"

# Generate Cargo.toml from template
echo "Generating Cargo.toml..."
sed \
    -e "s|{{HYPERPOLYGLOT_PATH}}|${HYPERPOLYGLOT_SRC}|g" \
    -e "s|{{SOFTWARE_METRICS_PATH}}|${SOFTWARE_METRICS_DIR}|g" \
    "${SCRIPT_DIR}/Cargo.toml.template" > "${SCRIPT_DIR}/Cargo.toml"

# Generate software_metrics Cargo.toml
# Using rust-code-analysis from crates.io (uses tree-sitter 0.20.9)
echo "Generating software_metrics wrapper..."
cat > "${SOFTWARE_METRICS_DIR}/Cargo.toml" << EOF
[package]
name = "software-metrics"
version = "0.1.0"
edition = "2021"
license = "Apache-2.0"

[lib]
name = "software_metrics"
crate-type = ["rlib", "cdylib"]

[dependencies]
pyo3 = "0.22.0"
lazy-regex = "*"
# Use rust-code-analysis from crates.io
rust-code-analysis = "0.0.25"
# Pin tree-sitter versions to match rust-code-analysis 0.0.25 (uses 0.20.9)
tree-sitter = "=0.20.9"
tree-sitter-python = "=0.20.2"
tree-sitter-rust = "=0.20.3"
tree-sitter-java = "=0.20.0"
tree-sitter-javascript = "=0.20.0"
tree-sitter-typescript = "=0.20.1"
EOF

cd "${SCRIPT_DIR}"

# Build the library
echo "Building Rust library..."
cargo build --release

# Build Python wheel
echo "Building Python wheel..."
if command -v maturin &> /dev/null; then
    maturin build --release
else
    echo "Warning: maturin not found. Skipping wheel build."
    echo "Install with: pip install maturin"
fi

echo "=== code_annotation built successfully ==="
