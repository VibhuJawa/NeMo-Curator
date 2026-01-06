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

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

mod annotations;
use annotations::{
    basic_stats, decontaminate, detect_language, ngrams_matches, opencoder_rs_stats,
    tokenize,
};
#[cfg(feature = "software_metrics")]
use annotations::software_metrics;

/// Compute basic statistics for a list of code strings.
///
/// # Arguments
/// * `codes` - List of source code strings
/// * `xml_header_search_length` - Number of bytes to search for XML header
/// * `max_byte_size` - Optional maximum byte size (code larger than this returns None for stats)
///
/// # Returns
/// List of dictionaries with statistics for each code string
#[pyfunction]
#[pyo3(signature = (codes, xml_header_search_length=1024, max_byte_size=None))]
fn compute_basic_stats<'py>(
    py: Python<'py>,
    codes: Vec<String>,
    xml_header_search_length: usize,
    max_byte_size: Option<usize>,
) -> PyResult<Bound<'py, PyList>> {
    let results = basic_stats(&codes, xml_header_search_length, max_byte_size);

    let py_list = PyList::empty_bound(py);
    for result in results {
        let dict = PyDict::new_bound(py);
        dict.set_item("num_bytes", result.num_bytes)?;
        dict.set_item("valid_utf8", result.valid_utf8)?;
        dict.set_item("max_line_length", result.max_line_length)?;
        dict.set_item("num_lines", result.num_lines)?;
        dict.set_item("average_line_length", result.average_line_length)?;
        dict.set_item("contains_xml_header", result.contains_xml_header)?;
        dict.set_item("alpha_percent", result.alpha_percent)?;
        dict.set_item("alnum_percent", result.alnum_percent)?;
        dict.set_item("base64_percent", result.base64_percent)?;
        dict.set_item("hex_percent", result.hex_percent)?;
        dict.set_item("unicode_percent", result.unicode_percent)?;
        dict.set_item("base64_match_lengths", result.base64_match_lengths.clone())?;
        dict.set_item("hex_match_lengths", result.hex_match_lengths.clone())?;
        dict.set_item("unicode_match_lengths", result.unicode_match_lengths.clone())?;
        py_list.append(dict)?;
    }
    Ok(py_list)
}

/// Detect programming languages for a list of code strings.
///
/// # Arguments
/// * `codes` - List of source code strings
/// * `filenames` - List of filenames (used for language detection hints)
///
/// # Returns
/// List of dictionaries with language and detector variant for each code string
#[pyfunction]
fn compute_language_detection<'py>(
    py: Python<'py>,
    codes: Vec<String>,
    filenames: Vec<String>,
) -> PyResult<Bound<'py, PyList>> {
    let results = detect_language(&codes, &filenames);

    let py_list = PyList::empty_bound(py);
    for result in results {
        let dict = PyDict::new_bound(py);
        dict.set_item("language", result.language)?;
        dict.set_item("language_detector", result.detector)?;
        py_list.append(dict)?;
    }
    Ok(py_list)
}

/// Compute software metrics for a list of code strings.
///
/// Requires the `software_metrics` feature to be enabled.
///
/// # Arguments
/// * `codes` - List of source code strings
/// * `languages` - List of detected languages (from compute_language_detection)
///
/// # Returns
/// List of dictionaries with software metrics for each code string
#[cfg(feature = "software_metrics")]
#[pyfunction]
fn compute_software_metrics<'py>(
    py: Python<'py>,
    codes: Vec<String>,
    languages: Vec<Option<String>>,
) -> PyResult<Bound<'py, PyList>> {
    let results = software_metrics(&codes, &languages);

    let py_list = PyList::empty_bound(py);
    for result in results {
        let dict = PyDict::new_bound(py);
        dict.set_item("cyclomatic_complexity", result.cyclomatic_complexity)?;
        dict.set_item("cognitive_complexity", result.cognitive_complexity)?;
        dict.set_item("exits_average", result.exits_average)?;
        dict.set_item("maintainability_index", result.maintainability_index)?;
        dict.set_item("halstead_difficulty", result.halstead_difficulty)?;
        dict.set_item("comment_lines", result.comment_lines)?;
        dict.set_item("comment_lines_frac", result.comment_lines_frac)?;
        dict.set_item("comment_lines_per_space", result.comment_lines_per_space)?;
        dict.set_item("blank_lines", result.blank_lines)?;
        dict.set_item("blank_lines_per_space", result.blank_lines_per_space)?;
        dict.set_item("args_average", result.args_average)?;
        dict.set_item("functions_closures_per_space", result.functions_closures_per_space)?;
        dict.set_item("total_cda", result.total_cda)?;
        dict.set_item("total_wmc", result.total_wmc)?;
        dict.set_item("total_coa", result.total_coa)?;
        dict.set_item("parsed_ok", result.parsed_ok)?;
        py_list.append(dict)?;
    }
    Ok(py_list)
}

/// Tokenize a list of code strings.
///
/// # Arguments
/// * `codes` - List of source code strings
/// * `tokenizer_name` - Name of the tokenizer ("tiktoken_o200k_base" or "github_o200k_base")
/// * `vocab` - Optional custom vocabulary for tokenization
/// * `pretokenizer_patterns` - Optional pretokenizer patterns (list of (pattern, is_lookahead) tuples)
///
/// # Returns
/// List of dictionaries with tokens and token count for each code string
#[pyfunction]
#[pyo3(signature = (codes, tokenizer_name, vocab=None, pretokenizer_patterns=None))]
fn compute_tokenization<'py>(
    py: Python<'py>,
    codes: Vec<String>,
    tokenizer_name: String,
    vocab: Option<String>,
    pretokenizer_patterns: Option<Vec<(String, bool)>>,
) -> PyResult<Bound<'py, PyList>> {
    let results = tokenize(&codes, &tokenizer_name, vocab.as_deref(), pretokenizer_patterns.as_deref());

    let py_list = PyList::empty_bound(py);
    for result in results {
        let dict = PyDict::new_bound(py);
        dict.set_item("tokens", result.tokens.clone())?;
        dict.set_item("num_tokens", result.num_tokens)?;
        py_list.append(dict)?;
    }
    Ok(py_list)
}

/// Compute OpenCoder-RS comment statistics.
///
/// # Arguments
/// * `codes` - List of source code strings
/// * `languages` - List of detected languages
///
/// # Returns
/// List of dictionaries with comment line and char fractions
#[pyfunction]
fn compute_opencoder_rs<'py>(
    py: Python<'py>,
    codes: Vec<String>,
    languages: Vec<Option<String>>,
) -> PyResult<Bound<'py, PyList>> {
    let results = opencoder_rs_stats(&codes, &languages);

    let py_list = PyList::empty_bound(py);
    for result in results {
        let dict = PyDict::new_bound(py);
        dict.set_item("comment_lines_frac", result.comment_lines_frac)?;
        dict.set_item("comment_chars_frac", result.comment_chars_frac)?;
        py_list.append(dict)?;
    }
    Ok(py_list)
}

/// Check for n-gram contamination in code strings.
///
/// # Arguments
/// * `codes` - List of source code strings
/// * `ngrams` - Dictionary mapping labels to lists of n-grams to search for
/// * `ngram_order` - The n-gram order (e.g., 3 for trigrams)
///
/// # Returns
/// Dictionary mapping labels to lists of match counts per code string
#[pyfunction]
fn compute_decontamination<'py>(
    py: Python<'py>,
    codes: Vec<String>,
    ngrams: std::collections::HashMap<String, Vec<String>>,
    ngram_order: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let results = decontaminate(&codes, &ngrams, ngram_order);

    let py_dict = PyDict::new_bound(py);
    for (label, counts) in results {
        py_dict.set_item(label, counts)?;
    }
    Ok(py_dict)
}

/// Find matching n-grams in code strings.
///
/// # Arguments
/// * `codes` - List of source code strings
/// * `ngrams` - Dictionary mapping labels to lists of n-grams to search for
/// * `ngram_order` - The n-gram order (e.g., 3 for trigrams)
///
/// # Returns
/// Dictionary mapping labels to lists of matched n-grams per code string
#[pyfunction]
fn compute_ngram_matches<'py>(
    py: Python<'py>,
    codes: Vec<String>,
    ngrams: std::collections::HashMap<String, Vec<String>>,
    ngram_order: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let results = ngrams_matches(&codes, &ngrams, ngram_order);

    let py_dict = PyDict::new_bound(py);
    for (label, matches) in results {
        let py_matches = PyList::empty_bound(py);
        for match_list in matches {
            py_matches.append(match_list)?;
        }
        py_dict.set_item(label, py_matches)?;
    }
    Ok(py_dict)
}

/// Python module definition
#[pymodule]
fn _code_annotation(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compute_basic_stats, m)?)?;
    m.add_function(wrap_pyfunction!(compute_language_detection, m)?)?;
    #[cfg(feature = "software_metrics")]
    m.add_function(wrap_pyfunction!(compute_software_metrics, m)?)?;
    m.add_function(wrap_pyfunction!(compute_tokenization, m)?)?;
    m.add_function(wrap_pyfunction!(compute_opencoder_rs, m)?)?;
    m.add_function(wrap_pyfunction!(compute_decontamination, m)?)?;
    m.add_function(wrap_pyfunction!(compute_ngram_matches, m)?)?;

    Ok(())
}
