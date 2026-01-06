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

//! Software Metrics wrapper for rust-code-analysis.
//!
//! This module provides a thin wrapper around Mozilla's rust-code-analysis
//! library, adding language name to parser mapping and Python bindings.

use lazy_regex::Regex;
use pyo3::prelude::*;
use pyo3::types::{IntoPyDict, PyDict};
use rust_code_analysis::{metrics, FuncSpace, ParserTrait};
use std::path::Path;
use tree_sitter::{Language, Parser};

// Language parsers - tree-sitter 0.22.x API
use tree_sitter_java::language as java_language;
use tree_sitter_javascript::language as javascript_language;
use tree_sitter_python::language as python_language;
use tree_sitter_rust::language as rust_language;
use tree_sitter_typescript::language_typescript as typescript_language;

// Note: C++ parsing uses rust-code-analysis's internal parser, not tree-sitter-cpp

const LANGUAGE_PYTHON: &str = "Python";
const LANGUAGE_RUST: &str = "Rust";
const LANGUAGE_CPP: &str = "C++";
const LANGUAGE_C: &str = "C";
const LANGUAGE_JAVA: &str = "Java";
const LANGUAGE_JAVASCRIPT: &str = "JavaScript";
const LANGUAGE_TYPESCRIPT: &str = "TypeScript";

/// Get regex for extracting code blocks from markdown.
fn get_language_regex(language_name: &str) -> Option<Regex> {
    match language_name {
        LANGUAGE_PYTHON => Some(Regex::new(r"(?si)```(?:python|py|\n)\s*(.*?)```").unwrap()),
        LANGUAGE_CPP => Some(Regex::new(r"(?si)```(?:cpp|c\+\+|\n)\s*(.*?)```").unwrap()),
        LANGUAGE_C => Some(Regex::new(r"(?si)```(?:c\n)\s*(.*?)```").unwrap()),
        LANGUAGE_RUST => Some(Regex::new(r"(?si)```(?:rust|rs\n)\s*(.*?)```").unwrap()),
        LANGUAGE_JAVA => Some(Regex::new(r"(?si)```(?:java\n)\s*(.*?)```").unwrap()),
        LANGUAGE_JAVASCRIPT => Some(Regex::new(r"(?si)```(?:javascript|js\n)\s*(.*?)```").unwrap()),
        LANGUAGE_TYPESCRIPT => Some(Regex::new(r"(?si)```(?:typescript|ts\n)\s*(.*?)```").unwrap()),
        _ => None,
    }
}

/// Extract code blocks from markdown text.
pub fn extract_code_block(text: &str, language_name: &str) -> Option<String> {
    if let Some(re) = get_language_regex(language_name) {
        if let Some(captures) = re.captures(text) {
            if let Some(block) = captures.get(1) {
                return Some(block.as_str().to_string());
            }
        }
    }
    None
}

/// Check if code has parsing errors.
pub fn has_parsing_errors(code: &str, language_name: &str) -> bool {
    match get_ts_grammar(language_name) {
        Some(lang) => {
            let mut parser = Parser::new();
            parser.set_language(&lang).expect("Error loading language");
            match parser.parse(code, None) {
                Some(tree) => tree.root_node().has_error(),
                None => true,
            }
        }
        None => true,
    }
}

/// Get tree-sitter grammar for a language.
pub fn get_ts_grammar(language_name: &str) -> Option<Language> {
    match language_name {
        LANGUAGE_PYTHON => Some(python_language()),
        LANGUAGE_RUST => Some(rust_language()),
        LANGUAGE_JAVA => Some(java_language()),
        LANGUAGE_JAVASCRIPT => Some(javascript_language()),
        LANGUAGE_TYPESCRIPT => Some(typescript_language()),
        // C/C++ use rust-code-analysis's internal parser
        LANGUAGE_CPP | LANGUAGE_C => None,
        _ => None,
    }
}

/// Get metrics for source code.
pub fn get_metrics(language_name: &str, source_code: &str) -> Option<FuncSpace> {
    let path = Path::new("foo.c");
    let source_as_vec = source_code.as_bytes().to_vec();

    match language_name {
        LANGUAGE_PYTHON => {
            let parser = rust_code_analysis::PythonParser::new(source_as_vec, path, None);
            metrics(&parser, path)
        }
        LANGUAGE_CPP | LANGUAGE_C => {
            let parser = rust_code_analysis::CppParser::new(source_as_vec, path, None);
            metrics(&parser, path)
        }
        LANGUAGE_JAVA => {
            let parser = rust_code_analysis::JavaParser::new(source_as_vec, path, None);
            metrics(&parser, path)
        }
        LANGUAGE_JAVASCRIPT => {
            let parser = rust_code_analysis::JavascriptParser::new(source_as_vec, path, None);
            metrics(&parser, path)
        }
        LANGUAGE_TYPESCRIPT => {
            let parser = rust_code_analysis::TypescriptParser::new(source_as_vec, path, None);
            metrics(&parser, path)
        }
        _ => None,
    }
}

/// Get all metrics as a list of key-value pairs.
pub fn get_mz_metrics(language_name: &str, source_code: &str) -> Vec<(&'static str, f64)> {
    let mut result: Vec<(&'static str, f64)> = Vec::new();

    if let Some(mzmetrics) = get_metrics(language_name, source_code) {
        let metrics = mzmetrics.metrics;
        result.push(("cyclomatic", metrics.cyclomatic.cyclomatic_average()));
        result.push(("cognitive_complexity", metrics.cognitive.cognitive_average()));
        result.push(("exits average", metrics.nexits.exit_average()));
        result.push(("maintainability_index", metrics.mi.mi_visual_studio()));
        result.push(("halstead_difficulty", metrics.halstead.difficulty()));
        result.push(("# comments", metrics.loc.cloc()));
        result.push(("# comments per space", metrics.loc.cloc_average()));
        result.push(("# blank lines", metrics.loc.blank()));
        result.push(("# blank lines per space", metrics.loc.blank_average()));
        result.push(("# args average", metrics.nargs.nargs_average()));
        result.push(("functions/closures per space", metrics.nom.average()));

        let cda = metrics.npa.total_cda();
        if !cda.is_nan() {
            result.push(("total cda", cda));
            result.push(("total_wmc", metrics.wmc.total_wmc()));
        }

        let coa = metrics.npm.total_coa();
        if !coa.is_nan() {
            result.push(("total coa", coa));
        }

        if has_parsing_errors(source_code, language_name) {
            result.push(("parsed_ok", 0.));
        } else {
            result.push(("parsed_ok", 1.));
        }
    }

    result
}

/// Python binding for get_all_metrics.
#[pyfunction]
#[pyo3(name = "get_all_metrics")]
fn get_mz_metrics_py<'a>(
    py: Python<'a>,
    language_name: &'a str,
    source_code: &'a str,
) -> Bound<'a, PyDict> {
    let metrics = get_mz_metrics(language_name, source_code);
    let dict = PyDict::new_bound(py);
    for (k, v) in &metrics {
        let _ = dict.set_item(k, v.into_py(py));
    }
    dict.into_py_dict_bound(py)
}

/// Python module definition.
#[pymodule]
fn software_metrics(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(get_mz_metrics_py, m)?)?;
    Ok(())
}
