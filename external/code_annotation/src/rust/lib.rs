// Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Code Annotation Library for NeMo Curator
//!
//! This library provides fast code annotation capabilities including:
//! - Basic statistics (byte counts, line stats, pattern detection)
//! - Language detection via hyperpolyglot
//! - Software metrics via rust-code-analysis
//! - BPE tokenization
//! - Decontamination via n-gram matching

use pyo3::{prelude::*, types::PyDict};
use pyo3_polars::PyDataFrame;
use std::collections::HashMap;

mod annotations;
use annotations::call_functions;

/// Annotate a DataFrame with code analysis results.
///
/// # Arguments
/// * `function_dict` - Dictionary mapping function names to their arguments
/// * `pydf` - Input DataFrame with compressed source code
///
/// # Returns
/// DataFrame with annotation columns added
#[pyfunction]
fn annotate(function_dict: &Bound<'_, PyDict>, pydf: PyDataFrame) -> PyResult<PyDataFrame> {
    let mut function_args = Vec::new();
    let function_names: Vec<String> = function_dict
        .iter()
        .map(|(k, _)| k.extract::<String>().unwrap())
        .collect();

    for func_name in &function_names {
        if let Ok(args_any) = function_dict.get_item(func_name) {
            if let Ok(args_dict) = args_any.expect("Expected dict").downcast::<PyDict>() {
                let mut args_map = HashMap::new();
                for (key, value) in args_dict.iter() {
                    let key_str = key.extract::<String>()?;
                    args_map.insert(key_str.clone(), value);
                }
                function_args.push(args_map);
            }
        }
    }

    call_functions(function_names, function_args, &mut pydf.into())
}

/// Python module definition
#[pymodule]
fn _code_annotation(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Add the annotate function
    m.add_function(wrap_pyfunction!(annotate, m)?)?;

    // Add constants for column names
    m.add("CODE_COL_NAME", "content")?;
    m.add("COMPRESSED_SRC_COL_NAME", "compressed_content")?;
    m.add("LANGUAGE_COL_NAME", "language")?;
    m.add("FILENAME_COL_NAME", "representative_filename")?;
    m.add("TOKENS_COL_NAME", "tokenized_content")?;

    Ok(())
}
