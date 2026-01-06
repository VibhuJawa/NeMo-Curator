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
# Hyperpolyglot Build Script
# =============================================================================
# This script applies patches and builds hyperpolyglot as a Rust library.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="${SCRIPT_DIR}/src"
PATCHES_DIR="${SCRIPT_DIR}/patches"
# Must match the commit in fetch.sh
COMMIT_SHA="a55a3b5"

echo "=== Building hyperpolyglot ==="

# Check if source exists
if [ ! -d "${SRC_DIR}" ]; then
    echo "Error: Source directory not found. Run fetch.sh first."
    exit 1
fi

cd "${SRC_DIR}"

# Reset to the pinned commit to ensure clean state before patching
echo "Resetting to commit ${COMMIT_SHA}..."
git checkout -- .
git checkout "${COMMIT_SHA}" 2>/dev/null || true

# Apply patches
echo "Applying patches..."
for patch in "${PATCHES_DIR}"/*.patch; do
    if [ -f "$patch" ]; then
        echo "  Applying: $(basename "$patch")"
        git apply "$patch" || {
            echo "Error: Patch $(basename "$patch") failed to apply"
            exit 1
        }
    fi
done

# Build the library (library only, not as a Python package directly)
echo "Building Rust library..."
cargo build --release --lib

echo "=== Hyperpolyglot built successfully ==="
echo "Library location: ${SRC_DIR}/target/release/"
