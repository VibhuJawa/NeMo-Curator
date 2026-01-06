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
# Hyperpolyglot Fetch Script
# =============================================================================
# This script fetches the hyperpolyglot repository at a specific commit/tag.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_URL="https://github.com/monkslc/hyperpolyglot.git"
# Pin to a specific commit for reproducibility
# Using the main branch head as of 2025-01
COMMIT_SHA="a55a3b5"
TARGET_DIR="${SCRIPT_DIR}/src"

echo "=== Fetching hyperpolyglot ==="
echo "Repository: ${REPO_URL}"
echo "Commit: ${COMMIT_SHA}"
echo "Target: ${TARGET_DIR}"

# Clean up any existing source
if [ -d "${TARGET_DIR}" ]; then
    echo "Removing existing source directory..."
    rm -rf "${TARGET_DIR}"
fi

# Clone the repository (shallow clone of main branch)
echo "Cloning repository..."
git clone --depth 100 "${REPO_URL}" "${TARGET_DIR}"

# Checkout specific commit
cd "${TARGET_DIR}"
git checkout "${COMMIT_SHA}" 2>/dev/null || {
    echo "Commit ${COMMIT_SHA} not found, using HEAD"
    git checkout HEAD
}

echo "=== Hyperpolyglot fetched successfully ==="
echo "Source directory: ${TARGET_DIR}"
