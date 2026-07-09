# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

import subprocess
import sys


def test_reader_exports_are_lazy() -> None:
    script = """
import importlib
import sys

package_name = "nemo_curator.stages.text.io.reader"
readers = importlib.import_module(package_name)
reader_modules = {f"{package_name}.{name}" for name in ("jsonl", "lance", "parquet")}

assert readers.__all__ == ["JsonlReader", "LanceReader", "ParquetReader"]
assert reader_modules.isdisjoint(sys.modules)
assert "lance" not in sys.modules

readers.JsonlReader
readers.ParquetReader
assert f"{package_name}.jsonl" in sys.modules
assert f"{package_name}.parquet" in sys.modules
assert f"{package_name}.lance" not in sys.modules
assert "lance" not in sys.modules

readers.LanceReader
assert f"{package_name}.lance" in sys.modules
assert "lance" in sys.modules
"""
    subprocess.run([sys.executable, "-c", script], check=True)  # noqa: S603
